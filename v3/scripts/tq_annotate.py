"""TeleopQ: add Mimic datagen_info to operator-generated demos so they can be used as Mimic seeds (v3).

Isaac Lab's annotate_demos.py --auto needs env.get_subtask_term_signals(), which the GR-1 pick-place Mimic env does not
implement. Everything Mimic needs is already in the recorded data, so we build it offline:
  eef_pose        <- obs/{left,right}_eef_pos + _quat
  target_eef_pose <- actions (wrist pos + quat, as env.action_to_target_eef_pose)
  object_pose     <- obs/object_pos + object_rot
  idle_right      <- True from (first step the object is >1 cm above its start) + OFFSET
OFFSET is calibrated on NVIDIA's 5 annotated seeds (boundary - first lift = 22..29 steps, mean ~26).
Usage: python tq_annotate.py IN.hdf5 OUT.hdf5 [--check-seeds]
"""
import sys, h5py, numpy as np

OFFSET = 26
LIFT_M = 0.01

def quat_to_mat(q):  # (T,4) x,y,z,w (Isaac Lab 3.0 convention; verified against the seeds' datagen_info) -> (T,3,3)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2*(y*y + z*z); R[:, 0, 1] = 2*(x*y - z*w); R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w); R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w); R[:, 2, 1] = 2*(y*z + x*w); R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R

def pose(pos, quat):
    T = np.zeros((len(pos), 4, 4), dtype=np.float32); T[:, :3, :3] = quat_to_mat(quat); T[:, :3, 3] = pos; T[:, 3, 3] = 1
    return T

def boundary_from_lift(obj_z):
    lift = np.nonzero(obj_z - obj_z[0] > LIFT_M)[0]
    if len(lift) == 0: return None
    return int(min(lift[0] + OFFSET, len(obj_z) - 2))

def check_seeds(path):
    f = h5py.File(path, "r"); errs = []; qerr = []
    for k in sorted(f["data"].keys()):
        d = f["data"][k]
        b = int(np.argmax(d["obs/datagen_info/subtask_term_signals/idle_right"][:].reshape(-1)))
        z = d["obs/datagen_info/object_pose/object"][:, 2, 3]
        errs.append(boundary_from_lift(z) - b)
        a = d["actions"][:]
        Tt = pose(a[:, 7:10], a[:, 10:14]); ref = d["obs/datagen_info/target_eef_pose/right"][:]
        qerr.append(float(np.abs(Tt - ref).max()))
    print("seed boundary error (pred - true, steps):", errs, "| max |target pose - datagen| :", max(qerr))

def annotate(src, dst):
    fi = h5py.File(src, "r"); fo = h5py.File(dst, "w")
    gi = fi["data"]; go = fo.create_group("data")
    for k, v in gi.attrs.items(): go.attrs[k] = v
    n_ok = n_skip = 0; info = {}
    for k in sorted(gi.keys(), key=lambda s: int(s.split("_")[-1])):
        d = gi[k]; ob = d["obs"]
        b = boundary_from_lift(ob["object_pos"][:, 2])
        if b is None:
            n_skip += 1; continue
        gi.copy(d, go, name=k)
        dg = go[k]["obs"].create_group("datagen_info")
        a = d["actions"][:]
        dg.create_dataset("eef_pose/left", data=pose(ob["left_eef_pos"][:], ob["left_eef_quat"][:]))
        dg.create_dataset("eef_pose/right", data=pose(ob["right_eef_pos"][:], ob["right_eef_quat"][:]))
        dg.create_dataset("target_eef_pose/left", data=pose(a[:, 0:3], a[:, 3:7]))
        dg.create_dataset("target_eef_pose/right", data=pose(a[:, 7:10], a[:, 10:14]))
        dg.create_dataset("object_pose/object", data=pose(ob["object_pos"][:], ob["object_rot"][:]))
        sig = np.zeros((1, len(a)), dtype=bool); sig[0, b:] = True
        dg.create_dataset("subtask_term_signals/idle_right", data=sig)
        info[k] = b; n_ok += 1
    fo.close(); fi.close()
    print(f"annotated {n_ok} demos, skipped {n_skip} (object never lifted) -> {dst}; boundaries: {info}")

if __name__ == "__main__":
    if "--check-seeds" in sys.argv:
        check_seeds("/workspace/isaaclab/datasets/dataset_annotated_gr1.hdf5")
    else:
        annotate(sys.argv[1], sys.argv[2])
