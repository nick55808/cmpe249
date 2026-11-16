#!/bin/bash
# TeleopQ v1 - step 3: train a behavior-cloning policy (robomimic BC-RNN, low-dim obs) on the 300 demos.
# Result on Nov 14, 2026: 2000 epochs, ~1.1 s/epoch on L40S (~37 min), final L1 loss ~4.8e-6.
# Output: logs/robomimic/Isaac-PickPlace-GR1T2-Abs-v0/bc_rnn_low_dim_gr1t2/<timestamp>/
#         (models/model_epoch_*.pth every 100 epochs, logs/normalization_params.txt)
set -euo pipefail
nohup ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-PickPlace-GR1T2-Abs-v0 --algo bc --normalize_training_actions \
  --dataset ./datasets/v1_generated_gr1.hdf5 > v1_train.log 2>&1 &
echo "started training, PID $!  (follow: grep -o 'Train Epoch [0-9]*' v1_train.log | tail -1)"
