#!/bin/bash
# TeleopQ v1 - step 2: amplify the 5 seed demos into 300 successful demos with Isaac Lab Mimic.
# Result on Nov 14, 2026: 300/388 attempts succeeded (77.3%), v1_generated_gr1.hdf5 = 145 MB.
set -euo pipefail
nohup ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu --headless --num_envs 20 --generation_num_trials 300 \
  --input_file ./datasets/dataset_annotated_gr1.hdf5 \
  --output_file ./datasets/v1_generated_gr1.hdf5 > v1_gen.log 2>&1 &
echo "started generation, PID $!  (follow: grep -a 'successful demos' v1_gen.log | tail -1)"
