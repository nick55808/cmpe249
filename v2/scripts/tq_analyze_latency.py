"""TeleopQ v2 latency sanity check: compare yield, smoothness (SPARC) and tracking error across latency settings."""
import re, json, numpy as np, h5py, os

DT = 0.05  # env step (20 Hz)

def sparc(speed, fs=1 / DT, padlevel=4, fc=10.0, amp_th=0.05):
    """Spectral arc length (Balasubramanian et al. 2015). More negative = less smooth."""
    if len(speed) < 4 or np.max(speed) < 1e-9:
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(speed, nfft)); Mf = Mf / Mf.max()
    sel = f <= fc; f_sel, M_sel = f[sel], Mf[sel]
    idx = np.nonzero(M_sel >= amp_th)[0]
    if len(idx) == 0:
        return float("nan")
    f_sel, M_sel = f_sel[idx[0]: idx[-1] + 1], M_sel[idx[0]: idx[-1] + 1]
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0] + 1e-12); dM = np.diff(M_sel)
    return float(-np.sum(np.sqrt(df ** 2 + dM ** 2)))

out = {}
for K in (0, 2, 5):
    r = {"latency_steps": K, "latency_ms": K * DT * 1000}
    log = "/workspace/teleopq/out/gen_lat%d.log" % K
    if os.path.exists(log):
        m = re.findall(r"(\d+)/(\d+) \(([\d.]+)%\) successful demos", open(log, errors="ignore").read())
        if m:
            r["successes"], r["attempts"], r["yield_pct"] = int(m[-1][0]), int(m[-1][1]), float(m[-1][2])
    h5 = "/workspace/isaaclab/datasets/v2/lat%d.hdf5" % K
    if os.path.exists(h5):
        with h5py.File(h5, "r") as f:
            S, L, J = [], [], []
            for d in f["data"].values():
                p = d["obs/right_eef_pos"][:]
                v = np.linalg.norm(np.diff(p, axis=0), axis=1) / DT
                S.append(sparc(v)); L.append(len(p))
                jerk = np.diff(p, n=3, axis=0) / DT ** 3
                J.append(float(np.mean(np.linalg.norm(jerk, axis=1))))
            r["n_demos"] = len(S)
            r["sparc_mean"], r["sparc_std"] = float(np.nanmean(S)), float(np.nanstd(S))
            r["len_mean_steps"] = float(np.mean(L)); r["jerk_mean"] = float(np.mean(J))
    op = "/workspace/teleopq/out/oplog_lat%d.npy" % K
    if os.path.exists(op):
        a = np.load(op)  # cols: env, step, eef, target3, perceived3, cmd3, true3, age
        right = a[a[:, 2] == 1]
        err = np.linalg.norm(right[:, 12:15] - right[:, 3:6], axis=1)
        r["track_err_mm_mean"] = float(err.mean() * 1000); r["track_err_mm_p95"] = float(np.percentile(err, 95) * 1000)
        r["perceived_age_steps_mean"] = float(right[:, 15].mean())
        # check: degradation hits commands, not observations -> perceived != true when K>0, cmd != target
        r["perceived_vs_true_mm_mean"] = float(np.linalg.norm(right[:, 6:9] - right[:, 12:15], axis=1).mean() * 1000)
    out[K] = r
print(json.dumps(out, indent=1))
json.dump(out, open("/workspace/teleopq/out/latency_test_summary.json", "w"), indent=1)
