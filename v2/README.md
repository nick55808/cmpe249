# TeleopQ v2 — Scripted operator and degradation injectors

**Goal:** create demonstrations whose quality problems are *known exactly*, by degrading what a scripted
teleoperator perceives while it collects them. The degradation changes the operator's commands; nothing is added
to the recorded observations afterwards.

**Status (Nov 16, 2026):** latency and jitter injectors built and measured; the labeling step that lets Mimic reuse
these demos as seeds (needed for v3) works. The occlusion injector and visibility score are built in v3.

## How the scripted operator works

`scripts/tq_generate.py` is Isaac Lab's `generate_dataset.py` plus a `TeleopQOperator` hook on
`env.target_eef_pose_to_action`. Mimic still decides *where* each hand should go (target `T_t`); the operator decides
*what command is sent*, using a degraded view of its own hand:

```
c_t = T_t + e_t
e_t = clip( e_{t-1} + K · (T_{t-k} − P̃_{t-k}), ±3 cm )
```

- `P̃_{t-k}`: hand position in the video frame the operator sees, `k` env steps old (latency), plus Gaussian noise
  (jitter). One env step = 0.05 s, so k = 0 / 2 / 5 → 0 / 100 / 250 ms.
- `K`: how strongly the operator corrects per step. With no delay this is smooth tracking; with delay the correction
  acts on stale errors and overshoots, like a person reacting to a lagging video feed.
- Orientation and finger commands pass through unchanged. Recorded observations stay the true simulator state.

**Why not delay the object pose instead?** Mimic reads the object pose only once at the start of each subtask, while
the object is still on the table, so delaying it would change nothing. This is documented in the experiment log.

## Results

20 successful demos per condition, 10 parallel envs. Yield = successful / attempted generations (95% Wilson interval).
SPARC is the smoothness of the right-hand speed profile (closer to 0 = smoother), over successful demos only.
Tracking error is over all attempts.

| Batch | Operator gain K | Latency | Jitter | Yield | SPARC (mean ± sd) | Tracking error mean / p95 |
|---|---|---|---|---|---|---|
| 1 | 0.3 | 0 ms | 0 | 20/26 = **76.9%** (57.9–89.0) | −6.32 ± 0.96 | 14.4 / 44.1 mm |
| 1 | 0.3 | 100 ms | 0 | 20/29 = **69.0%** (50.8–82.7) | −5.36 ± 0.95 | 16.4 / 40.5 mm |
| 1 | 0.3 | 250 ms | 0 | 20/166 = **12.0%** (7.9–17.9) | −4.35 ± 0.77 | 80.4 / 337.9 mm |
| 2 | 0.2 | 0 ms | 0 | 20/30 = **66.7%** (48.8–80.8) | −6.24 ± 0.89 | 15.8 / 47.5 mm |
| 2 | 0.2 | 100 ms | 0 | 20/33 = **60.6%** (43.7–75.3) | −5.66 ± 0.48 | 19.4 / 64.8 mm |
| 2 | 0.2 | 250 ms | 0 | 20/78 = **25.6%** (17.3–36.3) | −5.71 ± 0.65 | 27.3 / 77.6 mm |
| 2 | 0.2 | 0 ms | 5 mm | 20/38 = **52.6%** (37.3–67.5) | −5.87 ± 1.03 | 15.9 / 49.6 mm |
| 2 | 0.2 | 0 ms | 15 mm | 20/32 = **62.5%** (45.3–77.1) | −6.10 ± 1.04 | 19.8 / 59.6 mm |

Reference: clean Isaac Lab Mimic (v1, no operator) = 300/388 = 77.3%.

**One-seed amplification test:** one operator demo (0 ms, labeled with `tq_annotate.py`) given to Mimic as the only
seed → **13/15 = 86.7%** (62.1–96.3) successful generations. This is how v3 measures each seed's value.

## Findings

1. **The operator reproduces clean Mimic when undegraded.** K=0.3, 0 ms: 76.9% vs 77.3% for clean Mimic.
2. **Latency degrades demos in graded steps.** With K=0.2: 67% → 61% → 26% yield, tracking error 16 → 19 → 27 mm.
   K=0.3 collapses at 250 ms (12%): it is past the loop's stability limit, so attempts oscillate and fail. **K=0.2 is
   used from here on.**
3. **Jitter is a weak factor here.** The operator's gradual correction smooths most of the perception noise. The
   5 mm vs 15 mm ordering is within sampling noise at n ≈ 30–40 attempts.
4. **Smoothness can mislead (survivor bias).** Successful demos under latency score as *smoother* (SPARC closer to 0),
   because only calm attempts succeed. This is preliminary evidence relevant to H1: action smoothness alone may
   mis-rank seed quality.

## Labeling operator demos for Mimic (v3 prerequisite)

Isaac Lab's `annotate_demos.py --auto` needs `env.get_subtask_term_signals()`, which the GR-1 pick-place Mimic env
does not implement. `scripts/tq_annotate.py` rebuilds Mimic's `datagen_info` from the recorded data instead:

| Field | Built from |
|---|---|
| `eef_pose` | `obs/{left,right}_eef_pos` + `_quat` |
| `target_eef_pose` | `actions` (wrist position + quaternion) |
| `object_pose` | `obs/object_pos` + `object_rot` |
| `subtask_term_signals/idle_right` | True from (first step the object is > 1 cm above its start) + 26 steps |

Validated against NVIDIA's 5 annotated seeds (`logs/batch2_gain0.2/annotate.txt`): target poses match to 5e-7;
boundaries within −3…+4 steps. **Isaac Lab 3.0 stores quaternions as x, y, z, w** (`logs/batch2_gain0.2/framecheck.txt`).

## Limitations

- Scripted operator, not a human; one task (GR-1 pick-and-place, steering wheel).
- 20 successful demos per condition; yield intervals are ±15–20 points.
- Latency and jitter act on hand position only, not orientation.
- The subtask boundary rule is calibrated on 5 seeds.

## Files

| Path | Contents |
|---|---|
| `scripts/tq_generate.py` | Mimic generation + scripted operator (latency, jitter) |
| `scripts/tq_annotate.py`, `tq_subset.py` | Label operator demos for Mimic; pick single seeds |
| `scripts/tq_analyze_latency.py`, `tq_analyze_v2.py` | Analysis behind the tables |
| `scripts/tq_seedsig.py`, `tq_boundary.py`, `tq_framecheck.py` | Checks used to design the annotator |
| `scripts/run_*.sh` | The exact batch commands |
| `logs/batch1_gain0.3/`, `logs/batch2_gain0.2/` | Unedited run logs |
| `logs/checksums.txt` | SHA-256 of every dataset, operator log, script and log |
| `results/v2_summary.json` | All numbers above |
| `PROOF.md` | Claim → evidence index |

Datasets (`.hdf5`) and per-step operator logs (`.npy`) are not in git; they are identified by hash in
`logs/checksums.txt`.
