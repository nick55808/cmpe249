# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
from isaaclab.utils.string import list_intersection, string_to_callable

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="Pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--use_skillgen",
    action="store_true",
    default=False,
    help="Use skillgen to generate motion trajectories",
)
parser.add_argument(
    "--disable_dataset_compression",
    action="store_true",
    default=False,
    help="Disables dataset compression",
)
parser.add_argument(
    "--external_callback",
    default=None,
    help="Fully qualified path to an externally defined callback.",
)

# --- TeleopQ v2: scripted-operator degradation ---
parser.add_argument("--latency_steps", type=int, default=0, help="Operator feedback delay in env steps (20 Hz: 2 = 100 ms, 5 = 250 ms).")
parser.add_argument("--op_gain", type=float, default=0.4, help="Operator correction gain per step on the perceived position error.")
parser.add_argument("--jitter_mm", type=float, default=0.0, help="Std of Gaussian noise (mm) on the operator's perceived hand position.")
parser.add_argument("--tq_log", type=str, default=None, help="Where to write per-step operator log (npz).")
parser.add_argument("--occlusion", type=str, default="none", choices=["none", "partial", "heavy"],
                    help="Virtual occluder between the operator's head view and the object (perception only, no physics).")
parser.add_argument("--occ_thresh", type=float, default=0.5, help="Below this visible-corner fraction the operator's hand view freezes.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, remaining_args = parser.parse_known_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Call an external callback if requested.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# Error on unrecognized arguments.
unrecognized_args = list_intersection(remaining_args, remaining_args_env_registration)
if unrecognized_args:
    parser.error(f"unrecognized arguments: {' '.join(unrecognized_args)}")

"""Rest everything follows."""

import asyncio
import inspect
import logging
import random

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401

# import logger
logger = logging.getLogger(__name__)


class TeleopQOperator:
    """Scripted teleoperator: follows Mimic's intended hand target and corrects errors it sees on a degraded feed.

    Command each step:  c_t = T_t + e_t,  with the correction updated from what the operator *perceives*
        e_t = clip(e_{t-1} + K * (T_{t-k} - P~_{t-k}), +/- e_max)
    T: intended target (from Mimic), P~: hand position in the video frame shown to the operator.
      - latency: the frame is k env steps old (feed delay)
      - jitter: Gaussian noise (jitter_mm) on the perceived hand position
      - occlusion: a virtual plate between the head and the object (perception only, no physics). Each step the
        operator gets a fresh frame with probability v_t (visible fraction of the object's 8 bounding-box corners);
        otherwise its view of the hand stays frozen at the last fresh frame. (occ_thresh is kept for the CLI but unused.)
    Only commands are degraded; recorded observations stay the true simulator state. Orientation and fingers pass through.
    Quaternions are x,y,z,w (Isaac Lab 3.0).
    """

    HALF = (0.143, 0.143, 0.042)   # steering wheel half-extents (m), from the scene AABB (scale 0.75)
    CZ = 0.038                     # bbox centre height above the object origin (m)
    OCC_FRAC = 0.6                 # plate position along the head->object line
    OCC_HALF = {"partial": 0.09, "heavy": 0.12}   # plate half-size (m)
    OCC_SHIFT = {"partial": 0.09, "heavy": 0.0}   # sideways offset of the plate centre (m)

    def __init__(self, env, latency_steps, gain, jitter_mm, occlusion="none", occ_thresh=0.5, e_max=0.03):
        self.env, self.k, self.K, self.jit, self.e_max = env, int(latency_steps), float(gain), float(jitter_mm) / 1000.0, e_max
        self.occ, self.occ_thresh = occlusion, float(occ_thresh)
        self.robot = env.scene["robot"]; self.obj = env.scene["object"]
        self.head_idx = list(self.robot.body_names).index("head_yaw_link")
        self.state = {}
        # rows: env, ep_step, eef, target3, perceived3, cmd3, true3, frame_age, v_t, frozen, head3, objcentre3, objquat4
        self.log = []

    @staticmethod
    def _rot(q):  # xyzw -> 3x3
        x, y, z, w = (float(v) for v in q)
        return torch.tensor([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                             [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                             [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])

    def _scene(self, env_id):
        o = self.env.scene.env_origins[env_id].cpu()
        head = (self.robot.data.body_pos_w[env_id, self.head_idx].cpu() - o)
        opos = self.obj.data.root_pos_w[env_id].cpu() - o
        oq = self.obj.data.root_quat_w[env_id].cpu()
        R = self._rot(oq)
        centre = opos + R @ torch.tensor([0.0, 0.0, self.CZ])
        hx, hy, hz = self.HALF
        offs = torch.tensor([[sx*hx, sy*hy, sz*hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        corners = centre + offs @ R.T
        return head, centre, oq, corners

    def _place_occluder(self, st, head, centre):
        if self.occ == "none":
            st["plate"] = None; return
        d = centre - head; n = d / d.norm()
        up = torch.tensor([0.0, 0.0, 1.0]); u = torch.linalg.cross(n, up); u = u / u.norm(); v = torch.linalg.cross(u, n)
        p0 = head + self.OCC_FRAC * d + self.OCC_SHIFT[self.occ] * u
        st["plate"] = (p0, n, u, v, self.OCC_HALF[self.occ])

    def _visible_fraction(self, st, head, corners):
        if st["plate"] is None:
            return 1.0
        p0, n, u, v, s = st["plate"]; vis = 0
        for q in corners:
            r = q - head; denom = float(r @ n)
            blocked = False
            if abs(denom) > 1e-9:
                t = float((p0 - head) @ n) / denom
                if 0.0 < t < 1.0:
                    x = head + t * r - p0
                    blocked = abs(float(x @ u)) <= s and abs(float(x @ v)) <= s
            vis += 0 if blocked else 1
        return vis / 8.0

    def _reset(self, env_id, head, centre):
        st = {"T": {"left": [], "right": []}, "P": {"left": [], "right": []}, "e": {}, "last_len": -1,
              "last_seen": {"left": None, "right": None}}
        self._place_occluder(st, head, centre)
        self.state[env_id] = st

    def __call__(self, orig, target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=0):
        ep_len = int(self.env.episode_length_buf[env_id].item())
        head, centre, oq, corners = self._scene(env_id)
        st = self.state.get(env_id)
        if st is None or ep_len < st["last_len"] or ep_len == 0:
            self._reset(env_id, head, centre); st = self.state[env_id]
        st["last_len"] = ep_len
        v_t = self._visible_fraction(st, head, corners)
        new_targets = {}
        for i, eef in enumerate(("left", "right")):
            T = target_eef_pose_dict[eef].clone()
            t_now = T[:3, 3].clone()
            true_pos = self.env.get_robot_eef_pose(eef, env_ids=[env_id])[0, :3, 3].clone()
            Th, Ph = st["T"][eef], st["P"][eef]
            Th.append(t_now); Ph.append(true_pos)
            if len(Ph) > self.k + 1: Ph.pop(0); Th.pop(0)
            frame, intended_then = Ph[0].clone(), Th[0]
            age = len(Ph) - 1
            frozen = 0
            # graded occlusion: each step the operator gets a fresh frame with probability v_t (fraction of visible
            # object corners); otherwise the view of the hand stays frozen at the last fresh frame
            if eef == "left":
                st["fresh"] = bool(torch.rand(1).item() < v_t)
            if st["fresh"] or st["last_seen"][eef] is None:
                st["last_seen"][eef] = frame
            else:
                frame = st["last_seen"][eef].clone(); frozen = 1
            perceived = frame + self.jit * torch.randn_like(frame) if self.jit > 0 else frame
            e = st["e"].get(eef, torch.zeros_like(t_now))
            e = torch.clamp(e + self.K * (intended_then - perceived), -self.e_max, self.e_max)
            st["e"][eef] = e
            cmd = t_now + e
            T[:3, 3] = cmd
            new_targets[eef] = T
            self.log.append([env_id, ep_len, i, *t_now.tolist(), *perceived.tolist(), *cmd.tolist(), *true_pos.tolist(), age,
                             v_t, frozen, *head.tolist(), *centre.tolist(), *oq.tolist()])
        return orig(target_eef_pose_dict=new_targets, gripper_action_dict=gripper_action_dict, action_noise_dict=action_noise_dict, env_id=env_id)


def install_teleopq_operator(env, args):
    op = TeleopQOperator(env, args.latency_steps, args.op_gain, args.jitter_mm, args.occlusion, args.occ_thresh)
    orig = env.target_eef_pose_to_action

    def target_eef_pose_to_action(target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=0):
        return op(orig, target_eef_pose_dict, gripper_action_dict, action_noise_dict, env_id)

    env.target_eef_pose_to_action = target_eef_pose_to_action
    env._teleopq_operator = op
    print(f"[TeleopQ] operator installed: latency_steps={op.k} gain={op.K} jitter_mm={args.jitter_mm} occlusion={op.occ} thresh={op.occ_thresh}")
    try:  # one-time scene info for the visibility (V) computation; read-only
        robot = env.scene["robot"]
        print("[TeleopQ] robot body names:", list(robot.body_names))
        print("[TeleopQ] scene sensors:", list(getattr(env.scene, "sensors", {}).keys()))
        obj = env.scene["object"]
        print("[TeleopQ] object cfg spawn:", type(obj.cfg.spawn).__name__, getattr(obj.cfg.spawn, "usd_path", None), "scale", getattr(obj.cfg.spawn, "scale", None))
        try:
            from pxr import UsdGeom, Usd
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(obj.cfg.prim_path.replace("env_.*", "env_0"))
            bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeLocalBound(prim)
            r = bb.ComputeAlignedRange(); print("[TeleopQ] object local AABB min", list(r.GetMin()), "max", list(r.GetMax()))
        except Exception as e:
            print("[TeleopQ] bbox lookup failed:", e)
    except Exception as e:
        print("[TeleopQ] scene info failed:", e)
    return op


def main():
    num_envs = args_cli.num_envs

    # Setup output paths and get env name
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task
    if task_name:
        task_name = args_cli.task.split(":")[-1]
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    # Configure environment
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
        dataset_compression=not args_cli.disable_dataset_compression,
    )

    # Create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    install_teleopq_operator(env, args_cli)

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    # Check if the mimic API from this environment contains decprecated signatures
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        logger.warning(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    # Set seed for generation
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # Reset before starting
    env.reset()

    motion_planners = None
    try:
        if args_cli.use_skillgen:
            from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
            from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

            # Create one motion planner per environment
            motion_planners = {}
            for env_id in range(num_envs):
                print(f"Initializing motion planner for environment {env_id}")
                # Create a config instance from the task name
                planner_config = CuroboPlannerCfg.from_task_name(env_name)

                # Ensure visualization is only enabled for the first environment
                # If not, sphere and plan visualization will be too slow in isaac lab
                # It is efficient to visualize the spheres and plan for the first environment in rerun
                if env_id != 0:
                    planner_config.visualize_spheres = False
                    planner_config.visualize_plan = False

                motion_planners[env_id] = CuroboPlanner(
                    env=env,
                    robot=env.scene["robot"],
                    config=planner_config,  # Pass the config object
                    env_id=env_id,  # Pass environment ID
                )

            env.cfg.datagen_config.use_skillgen = True

        # Setup and run async data generation
        async_components = setup_async_generation(
            env=env,
            num_envs=args_cli.num_envs,
            input_file=args_cli.input_file,
            success_term=success_term,
            pause_subtask=args_cli.pause_subtask,
            motion_planners=motion_planners,  # Pass the motion planners dictionary
        )

        try:
            data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
            env_loop(
                env,
                async_components["reset_queue"],
                async_components["action_queue"],
                async_components["info_pool"],
                async_components["event_loop"],
                data_gen_tasks=data_gen_tasks,
            )
        except asyncio.CancelledError:
            print("Tasks were cancelled.")
        finally:
            # Cancel all async tasks when env_loop finishes
            data_gen_tasks.cancel()
            try:
                # Wait for tasks to be cancelled
                async_components["event_loop"].run_until_complete(data_gen_tasks)
            except asyncio.CancelledError:
                print("Remaining async tasks cancelled and cleaned up.")
            except Exception as e:
                print(f"Error cancelling remaining async tasks: {e}")
            # Cleanup of motion planners and their visualizers
            if motion_planners is not None:
                for env_id, planner in motion_planners.items():
                    if getattr(planner, "plan_visualizer", None) is not None:
                        print(f"Closing plan visualizer for environment {env_id}")
                        planner.plan_visualizer.close()
                        planner.plan_visualizer = None
                motion_planners.clear()
    finally:
        # TeleopQ: save the operator log before closing (atexit may not run after Kit shuts down)
        if args_cli.tq_log and hasattr(env, "_teleopq_operator"):
            np.save(args_cli.tq_log, np.asarray(env._teleopq_operator.log, dtype=np.float32))
            print(f"[TeleopQ] saved {len(env._teleopq_operator.log)} operator log rows to {args_cli.tq_log}")
        # Close env after async tasks are done so success_term is never called on a closed env
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # Close sim app
    simulation_app.close()
