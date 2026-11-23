#!/bin/bash
# Re-run jobs that crashed when two simulators started at once (Pink IK mesh load clash). Strictly sequential.
cd /workspace/isaaclab
while pgrep -f run_v2_batch2.sh > /dev/null; do sleep 15; done
echo "=== lat2_j0 retry start $(date +%T)"
./isaaclab.sh -p /workspace/teleopq/tq_generate.py --device cpu --headless --num_envs 10 --generation_num_trials 20 \
  --input_file ./datasets/dataset_annotated_gr1.hdf5 --output_file ./datasets/v2b/lat2_j0.hdf5 \
  --latency_steps 2 --jitter_mm 0 --op_gain 0.2 --tq_log /workspace/teleopq/out2/oplog_lat2_j0.npy > /workspace/teleopq/out2/gen_lat2_j0.log 2>&1
echo "=== lat2_j0 done $(date +%T): $(grep -a 'successful demos' /workspace/teleopq/out2/gen_lat2_j0.log | tail -1)"
echo "=== one-seed amplification test start $(date +%T)"
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --device cpu --headless --num_envs 5 --generation_num_trials 10 \
  --input_file ./datasets/v2/oneseed_test.hdf5 --output_file ./datasets/v2/oneseed_amplified.hdf5 > /workspace/teleopq/out2/oneseed.log 2>&1
echo "=== one-seed done $(date +%T): $(grep -a 'successful demos' /workspace/teleopq/out2/oneseed.log | tail -1)"
echo QUEUE_DONE
