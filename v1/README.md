# TeleopQ v1 — Baseline reproduction report

**Goal:** reproduce NVIDIA's documented Isaac Lab Mimic → robomimic BC pipeline on the Fourier GR-1 humanoid
and record a reference success rate before any TeleopQ research code is added (course step: *Reproduce*).

**Status:** complete (Nov 14, 2026). **Reference success rate: 48/50 = 96%** (95% Wilson interval 86.5–98.9%).

## Environment

| Item | Version / value |
|---|---|
| Machine | NVIDIA Brev "Isaac Launchable" on AWS g6e.4xlarge: 1× L40S 48 GB, 16 vCPU, 128 GB RAM, 256 GB disk |
| Isaac Sim | 6.0.1 (build `6.0.1-rc.7`, see `logs/environment.txt`) |
| Isaac Lab | 3.0.0 (repo VERSION file; the Brev launchable lists it as 3.0.0-beta2-post1) |
| robomimic | 0.4.0 |
| PyTorch | 2.10.0+cu128 |
| NVIDIA driver | 595.71 |
| Task (train / eval) | `Isaac-PickPlace-GR1T2-Abs-v0` |
| Task (generation) | `Isaac-PickPlace-GR1T2-Abs-Mimic-v0` |
| Control rate | sim dt 1/120 s × decimation 6 = **0.05 s per env step (20 Hz)** |
| Action space | 36-D: left wrist pose (pos 3 + quat 4), right wrist pose (7), 22 hand-joint targets |

Note: an L4 (24 GB) instance with a 100 GB disk failed during the launchable's setup (the setup needs about
100 GB); the default 256 GB disk on L40S worked.

## Pipeline and commands

All commands run from the Isaac Lab root (`/workspace/isaaclab`). The scripts in [`scripts/`](scripts/) contain
them exactly as run.

| Step | Script | What it does |
|---|---|---|
| 0 | `00_download_seed.sh` | Downloads NVIDIA's 5 annotated GR-1 pick-and-place seed demos (`dataset_annotated_gr1.hdf5`, 0.9 MB) |
| 1 | `01_smoke_test.sh` | Generates 10 demos with Mimic to check the install |
| 2 | `02_generate.sh` | Mimic amplifies the 5 seeds into 300 successful demos (20 parallel envs) |
| 3 | `03_train.sh` | robomimic BC-RNN (low-dim observations), 2000 epochs, actions normalized to [−1, 1] |
| 4 | `04_eval.sh` | 50 closed-loop rollouts of the final checkpoint, horizon 400 steps |

## Results

| Stage | Result |
|---|---|
| Smoke test | 10/12 attempts succeeded (83.3%); 4.8 MB for 10 demos (~0.5 MB/demo) |
| Generation | **300/388 attempts succeeded = 77.3%** (95% Wilson 72.9–81.2%); `v1_generated_gr1.hdf5` = 145 MB, 39,662 samples |
| Training | 2000 epochs, ~1.1 s/epoch on L40S (~37 min); L1 loss 1.19e-2 (epoch 1) → 7.2e-5 (105) → 1.8e-5 (676) → 6.6e-6 (1272) → 4.6e-6 (2000); no errors (0 tracebacks in `logs/v1_train.log`) |
| Action normalization | min −1.7409957647323608, max 1.3797252178192139 |
| **Evaluation** | **48/50 rollouts succeeded = 96%** (95% Wilson 86.5–98.9%), checkpoint `model_epoch_2000.pth` |

Machine-readable numbers: [`results/v1_summary.json`](results/v1_summary.json).

## Exit test

> One command reproduces a non-trivial success rate over 50 rollouts; record it as the v1 reference.

Passed. 96% is the clean-pipeline reference that v2–v4 (degraded operators, curated seed sets) are compared against.

## Limitations

- One task (GR-1 pick-and-place), one training seed, one evaluation of 50 rollouts. With 50 rollouts the
  interval is about ±6–10 points, so v4 uses 100 rollouts per arm.
- Only the final checkpoint was evaluated; earlier checkpoints may score differently.
- Seeds are NVIDIA's clean demos; no degradation yet (that starts in v2).
- Evaluation runs on CPU physics (`--device cpu`), as in NVIDIA's instructions.

## Files

| Path | Contents |
|---|---|
| `scripts/` | The exact commands for steps 0–4 |
| `results/v1_summary.json` | All v1 numbers |
| `logs/` | Unedited run logs, training config, checksums and environment snapshot copied from the GPU machine |
| `PROOF.md` | Maps every claim to the log line that proves it |
| `MODEL_CARD.md` | Model card for the v1 BC policy |
| `EXPERIMENT_LOG.md` | Timestamped log of the v1 run |

Large files (the 145 MB generated dataset and the checkpoints) are not stored in git; see `logs/README.md`.
