"""TeleopQ v3 step 1-3: build the seed pool, label each seed for Mimic, compute every score.

For every pool file datasets/v3/pool/<tag>.hdf5 (tag = L<k>_O<occ>_J<mm>):
  - label demos for Mimic (tq_annotate), split each successful demo into its own seed file datasets/v3/seeds/<seed_id>.hdf5
  - match each demo to its operator-log episode (initial object position, then length check)
  - scores per seed:
      V   mean visible fraction v_t over the contact windows (+-10 steps = +-0.5 s around right-hand close and open)
      L   mean age (steps) of the hand frame the operator saw (feed latency), from the operator log
      S   SPARC of the right-hand speed profile (closer to 0 = smoother; higher = better)
      q   z(V) - z(L) + z(S)      (z over the pool)
    baselines: q_noV = z(S) - z(L); psd_hf = -(share of action power above 2 Hz); knn = -(mean distance to 3 nearest seeds'
    object-relative right-hand trajectories); neg_len = -(episode length); oracle = -(z(lat)+z(occ)+z(jit)) from injected levels
Writes /workspace/teleopq/out_v3/scores.csv
"""
import sys, os, re, glob, csv, json, numpy as np, h5py
sys.path.insert(0, "/workspace/teleopq")
from tq_annotate import annotate, quat_to_mat

DT = 0.05; WIN = 10; CZ = 0.038
POOL = "/workspace/isaaclab/datasets/v3/pool"; SEEDS = "/workspace/isaaclab/datasets/v3/seeds"
OUT = "/workspace/teleopq/out_v3"; os.makedirs(SEEDS, exist_ok=True); os.makedirs(OUT, exist_ok=True)
OCC_LEVEL = {"none": 0, "partial": 1, "heavy": 2}

def sparc(speed, fs=1 / DT, padlevel=4, fc=10.0, amp_th=0.05):
    if len(speed) < 4 or np.max(speed) < 1e-9: return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft); Mf = np.abs(np.fft.fft(speed, nfft)); Mf = Mf / Mf.max()
    sel = f <= fc; f_sel, M_sel = f[sel], Mf[sel]; idx = np.nonzero(M_sel >= amp_th)[0]
    if len(idx) == 0: return float("nan")
    f_sel, M_sel = f_sel[idx[0]: idx[-1] + 1], M_sel[idx[0]: idx[-1] + 1]
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0] + 1e-12); dM = np.diff(M_sel)
    return float(-np.sum(np.sqrt(df ** 2 + dM ** 2)))

def hf_share(x, fs=1 / DT, f0=2.0):
    x = x - x.mean(0); P = np.abs(np.fft.rfft(x, axis=0)) ** 2; f = np.fft.rfftfreq(len(x), 1 / fs)
    tot = P.sum(); return float(P[f > f0].sum() / tot) if tot > 0 else 0.0

def episodes(oplog):
    """Split operator-log rows (right hand only) into episodes per env."""
    r = oplog[oplog[:, 2] == 1]; eps = []
    for env in np.unique(r[:, 0]):
        e = r[r[:, 0] == env]; cut = np.nonzero(np.diff(e[:, 1]) < 0)[0] + 1
        eps += [seg for seg in np.split(e, cut) if len(seg) > 5]
    return eps

def contact_steps(rh):
    c = np.linalg.norm(rh - rh[0], axis=1); m = c.max()
    if m < 1e-6: return []
    closed = c > 0.5 * m; ev = [int(np.argmax(closed))]
    after = np.nonzero(~closed[ev[0]:])[0]
    if len(after): ev.append(ev[0] + int(after[0]))
    return ev

rows = []
for path in sorted(glob.glob(POOL + "/L*_O*_J*.hdf5")):
    tag = os.path.basename(path)[:-5]
    m = re.match(r"L(\d+)_O(\w+)_J(\d+)", tag); lat, occ, jit = int(m[1]), m[2], int(m[3])
    ann = path.replace(".hdf5", "_annotated.hdf5"); annotate(path, ann)
    op = np.load(f"/workspace/teleopq/out_pool/oplog_{tag}.npy"); eps = episodes(op)
    starts = np.array([e[0, 21:24] for e in eps])
    with h5py.File(ann, "r") as f:
        for k in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1])):
            d = f["data"][k]; T = len(d["actions"])
            p0 = d["initial_state/rigid_object/object/root_pose"][0]
            c0 = p0[:3] + quat_to_mat(p0[None, 3:7])[0] @ np.array([0, 0, CZ])
            dist = np.linalg.norm(starts - c0, axis=1); j = int(np.argmin(dist))
            ep = eps[j]; match_mm = float(dist[j] * 1000)
            a = d["actions"][:]; rh = a[:, 25:]
            ev = contact_steps(rh)
            steps = ep[:, 1].astype(int); v = ep[:, 16]
            win = np.zeros(len(ep), bool)
            for t in ev: win |= np.abs(steps - t) <= WIN
            V = float(v[win].mean()) if win.any() else float(v.mean())
            L = float(ep[:, 15].mean())
            pos = d["obs/right_eef_pos"][:]; speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / DT
            S = sparc(speed)
            rel = pos - d["obs/object_pos"][0]
            traj = np.stack([np.interp(np.linspace(0, T - 1, 50), np.arange(T), rel[:, i]) for i in range(3)], 1)
            seed_id = f"{tag}_{k}"
            with h5py.File(f"{SEEDS}/{seed_id}.hdf5", "w") as fo:
                go = fo.create_group("data")
                for kk, vv in f["data"].attrs.items(): go.attrs[kk] = vv
                f["data"].copy(d, go, name="demo_0"); go.attrs["total"] = T
            rows.append(dict(seed_id=seed_id, tag=tag, latency_steps=lat, occlusion=occ, jitter_mm=jit, length=T,
                             V=V, L=L, S=S, psd_hf=-hf_share(a[:, 7:10]), match_mm=round(match_mm, 2),
                             frozen_share=float(ep[:, 17].mean()), contact_events=len(ev), _traj=traj))

z = lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
V = np.array([r["V"] for r in rows]); L = np.array([r["L"] for r in rows]); S = np.array([r["S"] for r in rows])
q = z(V) - z(L) + z(S); q_noV = z(S) - z(L)
lat = np.array([r["latency_steps"] for r in rows], float); occ = np.array([OCC_LEVEL[r["occlusion"]] for r in rows], float)
jit = np.array([r["jitter_mm"] for r in rows], float); oracle = -(z(lat) + z(occ) + z(jit))
trajs = np.stack([r["_traj"] for r in rows]); D = np.linalg.norm((trajs[:, None] - trajs[None]).reshape(len(rows), len(rows), -1), axis=2)
np.fill_diagonal(D, np.inf); knn = -np.sort(D, axis=1)[:, :3].mean(1)
for i, r in enumerate(rows):
    r.pop("_traj"); r.update(q=float(q[i]), q_noV=float(q_noV[i]), oracle=float(oracle[i]), knn=float(knn[i]), neg_len=-r["length"])
keys = list(rows[0].keys())
with open(OUT + "/scores.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} seeds from {len(set(r['tag'] for r in rows))} conditions -> {OUT}/scores.csv")
print("max episode-match distance (mm):", max(r["match_mm"] for r in rows))
