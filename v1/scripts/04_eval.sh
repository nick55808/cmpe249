#!/bin/bash
# TeleopQ v1 - step 4: closed-loop evaluation of the trained policy (50 rollouts, horizon 400).
# Result on Nov 14, 2026: 48/50 successful = 96% (95% Wilson interval 86.5%-98.9%).
# The norm factors come from <run folder>/logs/normalization_params.txt written by training.
set -euo pipefail
RUN=$(ls -td logs/robomimic/Isaac-PickPlace-GR1T2-Abs-v0/bc_rnn_low_dim_gr1t2/*/ | head -1)
MIN=$(grep -i '^min' "$RUN/logs/normalization_params.txt" | awk '{print $2}')
MAX=$(grep -i '^max' "$RUN/logs/normalization_params.txt" | awk '{print $2}')
echo "run: $RUN  norm min=$MIN max=$MAX"
# Values used on Nov 14, 2026: min -1.7409957647323608, max 1.3797252178192139
nohup ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --device cpu --headless --task Isaac-PickPlace-GR1T2-Abs-v0 \
  --num_rollouts 50 --horizon 400 \
  --norm_factor_min "$MIN" --norm_factor_max "$MAX" \
  --checkpoint "${RUN}models/model_epoch_2000.pth" > v1_eval.log 2>&1 &
echo "started evaluation, PID $!  (result: grep -a 'Success rate' v1_eval.log)"
