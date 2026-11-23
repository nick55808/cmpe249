#!/bin/bash
# TeleopQ v2 batch 2: operator gain K=0.2 at all three latencies, then jitter at zero latency.
# 20 successful demos per condition, 10 parallel envs. Output: datasets/v2b/<tag>.hdf5
cd /workspace/isaaclab
mkdir -p /workspace/teleopq/out2 datasets/v2b
run () {  # tag latency jitter
  echo "=== $1 (latency_steps=$2 jitter_mm=$3 K=0.2) start $(date +%T)"
  ./isaaclab.sh -p /workspace/teleopq/tq_generate.py --device cpu --headless --num_envs 10 --generation_num_trials 20 \
    --input_file ./datasets/dataset_annotated_gr1.hdf5 --output_file ./datasets/v2b/$1.hdf5 \
    --latency_steps $2 --jitter_mm $3 --op_gain 0.2 --tq_log /workspace/teleopq/out2/oplog_$1.npy > /workspace/teleopq/out2/gen_$1.log 2>&1
  echo "=== $1 done $(date +%T): $(grep -a 'successful demos' /workspace/teleopq/out2/gen_$1.log | tail -1)"
}
run lat0_j0 0 0
run lat2_j0 2 0
run lat5_j0 5 0
run lat0_j5 0 5
run lat0_j15 0 15
echo ALL_DONE
