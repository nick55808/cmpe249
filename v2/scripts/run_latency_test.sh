#!/bin/bash
# TeleopQ v2 latency sanity test: 20 generated demos per latency setting (0 / 2 / 5 steps = 0 / 100 / 250 ms)
cd /workspace/isaaclab
mkdir -p /workspace/teleopq/out datasets/v2
for K in 0 2 5; do
  echo "=== latency_steps=$K start $(date +%T)"
  ./isaaclab.sh -p /workspace/teleopq/tq_generate.py --device cpu --headless --num_envs 10 --generation_num_trials 20 \
    --input_file ./datasets/dataset_annotated_gr1.hdf5 --output_file ./datasets/v2/lat${K}.hdf5 \
    --latency_steps $K --op_gain 0.3 --tq_log /workspace/teleopq/out/oplog_lat${K}.npy > /workspace/teleopq/out/gen_lat${K}.log 2>&1
  echo "=== latency_steps=$K done $(date +%T): $(grep -a 'successful demos' /workspace/teleopq/out/gen_lat${K}.log | tail -1)"
done
echo ALL_DONE
