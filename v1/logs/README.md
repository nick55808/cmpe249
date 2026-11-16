# v1 logs and large artifacts

Small text logs from the GPU machine go in this folder:

| File | From (on the Brev machine, `/workspace/isaaclab/`) |
|---|---|
| `smoke.log` | smoke test output |
| `v1_gen.log` | 300-demo generation output |
| `v1_train.log` | BC training output |
| `v1_eval.log` | 50-rollout evaluation output |
| `config.json` | `logs/robomimic/Isaac-PickPlace-GR1T2-Abs-v0/bc_rnn_low_dim_gr1t2/20261114225335/config.json` |
| `normalization_params.txt` | same run folder, `logs/normalization_params.txt` |

Large artifacts are **not** stored in git (GitHub rejects files over 100 MB):

| File | Size | Where it is kept |
|---|---|---|
| `datasets/v1_generated_gr1.hdf5` | 145 MB | Brev machine; copy to the SJSU HPC project workspace |
| `models/model_epoch_*.pth` | per checkpoint | Brev machine; copy the final checkpoint to the HPC |
| `datasets/dataset_annotated_gr1.hdf5` | 0.9 MB | Re-download with `../scripts/00_download_seed.sh` |
