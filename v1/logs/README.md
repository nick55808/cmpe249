# v1 logs (unedited)

Copied byte-for-byte from the GPU machine (Brev L40S, `/workspace/isaaclab/`) on Nov 16, 2026. The SHA-256 of each
log matches the value recorded on the machine in `checksums.txt`.

| File | What it is |
|---|---|
| `smoke.log` | Day-1 smoke test: 10/12 demos generated |
| `v1_gen.log` | Mimic generation: 300/388 (77.3%) successful demos |
| `v1_train.log` | robomimic BC training, epochs 1–2000 |
| `v1_eval.log` | 50 evaluation rollouts: 48 successful, success rate 0.96 |
| `config.json` | robomimic training config saved by the run (run folder `bc_rnn_low_dim_gr1t2/20261114225335`) |
| `normalization_params.txt` | Action min/max used at evaluation |
| `checksums.txt` | SHA-256 and timestamps of the logs, the seed and generated datasets, and `model_epoch_2000.pth` |
| `environment.txt` | GPU (`nvidia-smi`), OS, CPU/RAM, Isaac Sim / Isaac Lab versions, Python package versions |

Large artifacts are **not** in git (GitHub rejects files over 100 MB): `v1_generated_gr1.hdf5` (145 MB) and the
checkpoints stay on the GPU machine / SJSU HPC. Their hashes in `checksums.txt` identify them. The seed file can be
re-downloaded with `../scripts/00_download_seed.sh`.
