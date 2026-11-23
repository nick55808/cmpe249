#!/bin/bash
# TeleopQ v3 seed pool: scripted operator over the full condition grid
# latency {0,2,5 steps} x occlusion {none,partial,heavy} x jitter {0,5,15 mm}; >=2 successful seeds per condition.
# Strictly one simulator at a time.
cd /workspace/isaaclab
/isaac-sim/python.sh -m py_compile /workspace/teleopq/tq_generate.py && echo COMPILE_OK || { echo COMPILE_FAIL; exit 1; }
mkdir -p /workspace/teleopq/out_pool datasets/v3/pool
for L in 0 2 5; do for O in none partial heavy; do for J in 0 5 15; do
  TAG=L${L}_O${O}_J${J}
  echo "=== $TAG start $(date +%T)"
  ./isaaclab.sh -p /workspace/teleopq/tq_generate.py --device cpu --headless --num_envs 4 --generation_num_trials 2 \
    --input_file ./datasets/dataset_annotated_gr1.hdf5 --output_file ./datasets/v3/pool/$TAG.hdf5 \
    --latency_steps $L --jitter_mm $J --op_gain 0.2 --occlusion $O --tq_log /workspace/teleopq/out_pool/oplog_$TAG.npy > /workspace/teleopq/out_pool/gen_$TAG.log 2>&1
  echo "=== $TAG done $(date +%T): $(grep -a 'successful demos' /workspace/teleopq/out_pool/gen_$TAG.log | tail -1) $(grep -c Traceback /workspace/teleopq/out_pool/gen_$TAG.log) tracebacks"
done; done; done
echo POOL_DONE
