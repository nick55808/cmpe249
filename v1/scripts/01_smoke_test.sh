#!/bin/bash
# TeleopQ v1 - step 1: day-1 smoke test. Generate 10 demos with Isaac Lab Mimic to check the install.
# Result on Nov 14, 2026: 10/12 attempts succeeded (83.3%), smoke_gr1.hdf5 = 4.8 MB.
set -euo pipefail
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu --headless --num_envs 10 --generation_num_trials 10 \
  --input_file ./datasets/dataset_annotated_gr1.hdf5 \
  --output_file ./datasets/smoke_gr1.hdf5 > smoke.log 2>&1
grep -a "successful demos" smoke.log | tail -1
