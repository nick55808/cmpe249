"""TeleopQ v2 analysis: yield, smoothness (SPARC), jerk and operator tracking error per condition.
Batch 1 = operator gain K=0.3 (latency only); batch 2 = K=0.2 (latency + jitter). Writes v2_summary.json."""
import re, json, os, numpy as np, h5py, math

DT = 0.05

def sparc(speed, fs=1 / DT, padlevel=4, fc=10.0, amp_th=0.05):
    if len(speed) < 4 or np.max(speed) < 1e-9: return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft); Mf = np.abs(np.fft.fft(speed, nfft)); Mf = Mf / Mf.max()
    sel = f <= fc; f_sel, M_sel = f[sel], Mf[sel]
    idx = np.nonzero(M_sel >= amp_th)[0]
    if len(idx) == 0: return float("nan")
    f_sel, M_sel = f_sel[idx[0]: idx[-1] + 1], M_sel[idx[0]: idx[-1] + 1]
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0] + 1e-12); dM = np.diff(M_sel)
    return float(-np.sum(np.sqrt(df ** 2 + dM ** 2)))

def wilson(k, n, z=1.96):
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [round(100*(c-h), 1), round(100*(c+h), 1)]

def one(tag, K, lat, jit, log, h5, op):
    r = {"tag": tag, "op_gain": K, "latency_steps": lat, "latency_ms": lat * 50, "jitter_mm": jit}
    m = re.findall(r"(\d+)/(\d+) \(([\d.]+)%\) successful demos", open(log, errors="ignore").read())
    k, n = int(m[-1][0]), int(m[-1][1]); r.update(successes=k, attempts=n, yield_pct=round(100*k/n, 1), yield_wilson95_pct=wilson(k, n))
    with h5py.File(h5, "r") as f:
        S, J, L = [], [], []
        for d in f["data"].values():
            p = d["obs/right_eef_pos"][:]; v = np.linalg.norm(np.diff(p, axis=0), axis=1) / DT
            S.append(sparc(v)); J.append(float(np.mean(np.linalg.norm(np.diff(p, n=3, axis=0) / DT**3, axis=1)))); L.append(len(p))
    r.update(n_demos=len(S), sparc_mean=round(float(np.nanmean(S)), 3), sparc_sd=round(float(np.nanstd(S)), 3),
             jerk_mean=round(float(np.mean(J)), 1), len_mean_steps=round(float(np.mean(L)), 1))
    a = np.load(op); right = a[a[:, 2] == 1]
    err = np.linalg.norm(right[:, 12:15] - right[:, 3:6], axis=1)
    r.update(track_err_mm_mean=round(float(err.mean()*1000), 1), track_err_mm_p95=round(float(np.percentile(err, 95)*1000), 1),
             perceived_age_steps_mean=round(float(right[:, 15].mean()), 2),
             perceived_vs_true_mm_mean=round(float(np.linalg.norm(right[:, 6:9] - right[:, 12:15], axis=1).mean()*1000), 1))
    return r

T = "/workspace/teleopq/"; D = "/workspace/isaaclab/datasets/"
rows = []
for lat in (0, 2, 5):
    rows.append(one(f"b1_lat{lat}", 0.3, lat, 0, T+f"out/gen_lat{lat}.log", D+f"v2/lat{lat}.hdf5", T+f"out/oplog_lat{lat}.npy"))
for tag, lat, jit in (("lat0_j0",0,0),("lat2_j0",2,0),("lat5_j0",5,0),("lat0_j5",0,5),("lat0_j15",0,15)):
    rows.append(one("b2_"+tag, 0.2, lat, jit, T+f"out2/gen_{tag}.log", D+f"v2b/{tag}.hdf5", T+f"out2/oplog_{tag}.npy"))
m = re.findall(r"(\d+)/(\d+) \(([\d.]+)%\) successful demos", open(T+"out2/oneseed.log", errors="ignore").read())
summary = {"conditions": rows, "one_seed_amplification": {"seed": "v2/lat0_annotated.hdf5 demo_0", "successes": int(m[-1][0]), "attempts": int(m[-1][1]),
           "yield_pct": float(m[-1][2]), "yield_wilson95_pct": wilson(int(m[-1][0]), int(m[-1][1]))}}
json.dump(summary, open(T+"out2/v2_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
