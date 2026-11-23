#!/bin/bash
# v3 prep: occlusion injector smoke test (10 successful demos per level, K=0.2, no latency/jitter). One simulator at a time.
cd /workspace/isaaclab
/isaac-sim/python.sh -m py_compile /workspace/teleopq/tq_generate.py && echo COMPILE_OK || { echo COMPILE_FAIL; exit 1; }
mkdir -p /workspace/teleopq/out3 datasets/v3test
for O in none partial heavy; do
  echo "=== occ_$O start $(date +%T)"
  ./isaaclab.sh -p /workspace/teleopq/tq_generate.py --device cpu --headless --num_envs 10 --generation_num_trials 10 \
    --input_file ./datasets/dataset_annotated_gr1.hdf5 --output_file ./datasets/v3test/occ_$O.hdf5 \
    --latency_steps 0 --jitter_mm 0 --op_gain 0.2 --occlusion $O --tq_log /workspace/teleopq/out3/oplog_occ_$O.npy > /workspace/teleopq/out3/gen_occ_$O.log 2>&1
  echo "=== occ_$O done $(date +%T): $(grep -a 'successful demos' /workspace/teleopq/out3/gen_occ_$O.log | tail -1) $(grep -c Traceback /workspace/teleopq/out3/gen_occ_$O.log) tracebacks"
done
echo OCC_TEST_DONE
