# TeleopQ v3 — evidence index

All logs are copied unedited from the GPU machine; `logs/checksums.txt` lists the SHA-256 recorded there for every
file, plus the datasets and operator logs that are too large for git.

| Claim | Where to look | What you will see |
|---|---|---|
| Occlusion v1 had no effect at "partial" | `logs/occlusion_test/occ_test.log`, `occ_check.txt` | none 10/17, partial 10/17, heavy 10/50; partial v_t early = 0.50 |
| 27 conditions, ≥ 2 seeds each | `logs/seed_pool/pool.log` | One `done` line per condition, 0 tracebacks |
| 56 seeds, each matched to its operator episode | `logs/yield/scores.txt` | `56 seeds from 27 conditions`, `max episode-match distance (mm): 0.0` |
| Every score per seed | `results/scores.csv` | V, L, S, q and all baselines, one row per seed |
| Each seed's yield | `logs/yield/yield_lane{0,1,2}.csv` + `logs/yield/yield_logs/<seed>.log` | e.g. `L0_Oheavy_J0_demo_0,8,20,capped`; last progress line in that seed's log |
| H1 table and key test | `results/h1_table.md`, `logs/h1.txt` | ρ(q) − ρ(S) = 0.543, CI [0.240, 0.833] |
| Smoothness ranks backwards | `results/h1_results.json` | S: ρ = −0.293, CI [−0.528, −0.044] |
| Plot | `results/q_vs_yield.png` | q vs yield (ρ = 0.25) and S vs yield (ρ = −0.29) |

## Quick checks

```bash
cat v3/results/h1_table.md
grep -c . v3/logs/yield/yield_lane*.csv            # 56 seed rows + 3 headers
grep -a "successful demos" v3/logs/yield/yield_logs/L0_Oheavy_J0_demo_0.log | tail -1
grep "done" v3/logs/seed_pool/pool.log | wc -l     # 27
```

## Questions to expect

| Question | Answer |
|---|---|
| Is H1 real or a statistics trick? | The test was fixed before the data: the interval of ρ(q) − ρ(S) must exclude 0. It does ([0.24, 0.83]). We also report that much of the gap is S being negative, and that latency alone beats q. |
| Why is smoothness negative? | Survivor bias: under latency/occlusion only calm attempts succeed and become seeds, so degraded seeds look smooth. First seen in v2, confirmed here against measured seed value. |
| Why not drop S from q now? | Changing the score after seeing the result would be overfitting. It is reported as a finding and becomes a pre-declared variant in v4. |
| How is "seed value" measured? | Mimic is given one seed alone and attempts ~20 new demos; yield = successes / attempts. |
| Isn't V just the occlusion label? | V is measured from geometry around the contact moments, not read from the setting; its correlation with yield (0.22) is weaker than the oracle's (0.44). |
