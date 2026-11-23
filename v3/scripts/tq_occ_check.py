"""Check the occlusion injector from operator logs: visible fraction v_t, how often the hand view froze, scene geometry."""
import re, numpy as np
np.set_printoptions(precision=3, suppress=True)
for O in ("none", "partial", "heavy"):
    try:
        a = np.load(f"/workspace/teleopq/out3/oplog_occ_{O}.npy")
    except Exception as e:
        print(O, "no log", e); continue
    r = a[a[:, 2] == 1]   # right hand rows
    v = r[:, 16]; fz = r[:, 17]
    head = r[:, 18:21]; cen = r[:, 21:24]
    log = open(f"/workspace/teleopq/out3/gen_occ_{O}.log", errors="ignore").read()
    m = re.findall(r"(\d+)/(\d+) \(([\d.]+)%\) successful demos", log)
    print(f"{O:8s} yield {m[-1][0]}/{m[-1][1]} ({m[-1][2]}%) | v_t mean {v.mean():.2f}, share of steps v_t<0.5 {np.mean(v < 0.5):.2f}, "
          f"frozen share {fz.mean():.2f} | head mean {head.mean(0)} | object centre mean {cen.mean(0)} | dist {np.linalg.norm(cen - head, axis=1).mean():.2f} m")
    # v_t over episode progress (first 40 steps ~ reach/grasp, later ~ carry/place)
    early = r[r[:, 1] < 40][:, 16].mean(); late = r[r[:, 1] >= 40][:, 16].mean()
    print(f"          v_t early (steps<40) {early:.2f}  later {late:.2f}")
