#!/bin/bash
# TeleopQ v3 step 4: per-seed Mimic yield. Usage: run_v3_yield.sh <lane> <n_lanes>
# Each seed alone is given to Mimic; stop after >= 20 attempts; yield = successes / attempts.
# Lanes run in parallel, but a startup lock makes sure only one simulator initializes at a time.
LANE=$1; NL=$2; MAX_ATT=20
cd /workspace/isaaclab
OUT=/workspace/teleopq/out_v3; mkdir -p $OUT/yield_logs datasets/v3/yield_out
RES=$OUT/yield_lane$LANE.csv; [ -f $RES ] || echo "seed_id,successes,attempts,status,start,end" > $RES
tail -n +2 $OUT/scores.csv | cut -d, -f1 | awk -v l=$LANE -v n=$NL 'NR % n == l' | while read SID; do
  grep -q "^$SID," $RES && continue
  LOG=$OUT/yield_logs/$SID.log; T0=$(date +%T)
  exec 9>/tmp/tq_start.lock; flock 9
  setsid ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --device cpu --headless --num_envs 5 \
    --generation_num_trials $MAX_ATT --input_file ./datasets/v3/seeds/$SID.hdf5 --output_file ./datasets/v3/yield_out/$SID.hdf5 > $LOG 2>&1 &
  PID=$!; W=0
  while kill -0 $PID 2>/dev/null && ! grep -qa "successful demos" $LOG && [ $W -lt 240 ]; do sleep 3; W=$((W+3)); done
  flock -u 9; exec 9>&-
  STATUS=done; W=0
  while kill -0 $PID 2>/dev/null; do
    N=$(grep -ao "[0-9]*/[0-9]* ([0-9.]*%) successful" $LOG | tail -1 | cut -d/ -f2 | cut -d' ' -f1)
    [ -n "$N" ] && [ "$N" -ge $MAX_ATT ] && { STATUS=capped; break; }
    [ $W -ge 600 ] && { STATUS=timeout; break; }
    sleep 3; W=$((W+3))
  done
  kill -9 -- -$PID 2>/dev/null; sleep 2
  KN=$(grep -ao "[0-9]*/[0-9]* ([0-9.]*%) successful" $LOG | tail -1 | cut -d' ' -f1)
  K=${KN%/*}; N=${KN#*/}; [ -z "$KN" ] && { K=0; N=0; STATUS=failed; }
  grep -q Traceback $LOG && STATUS="$STATUS+traceback"
  echo "$SID,$K,$N,$STATUS,$T0,$(date +%T)" >> $RES
done
echo "LANE_$LANE_DONE"
