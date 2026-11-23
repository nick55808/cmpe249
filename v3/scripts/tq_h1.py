"""TeleopQ v3 step 5: H1 analysis. Does the TeleopQ score q predict each seed's Mimic yield better than smoothness S?
Inputs: out_v3/scores.csv, out_v3/yield_lane*.csv. Outputs: out_v3/h1_results.json, h1_table.md, q_vs_yield.png, merged.csv"""
import csv, glob, json, numpy as np
OUT = "/workspace/teleopq/out_v3"; rng = np.random.default_rng(0); B = 1000

scores = {r["seed_id"]: r for r in csv.DictReader(open(OUT + "/scores.csv"))}
yields = {}
for f in glob.glob(OUT + "/yield_lane*.csv"):
    for r in csv.DictReader(open(f)):
        if int(r["attempts"]) > 0: yields[r["seed_id"]] = r
ids = sorted(set(scores) & set(yields))
y = np.array([int(yields[i]["successes"]) / int(yields[i]["attempts"]) for i in ids])
col = lambda k: np.array([float(scores[i][k]) for i in ids])

def rank(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):  # average ties
        m = x == v; r[m] = r[m].mean()
    return r
def spearman(a, b):
    ra, rb = rank(a), rank(b); ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum()); return float((ra * rb).sum() / d) if d > 0 else float("nan")
def boot(f, *arrs):
    n = len(arrs[0]); out = []
    for _ in range(B):
        idx = rng.integers(0, n, n); out.append(f(*[a[idx] for a in arrs]))
    out = np.array(out); out = out[~np.isnan(out)]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
def partial_spearman(x, yv, *ctrl):
    R = lambda v: rank(v) - rank(v).mean()
    C = np.column_stack([R(c) for c in ctrl] + [np.ones(len(x))])
    res = lambda v: R(v) - C @ np.linalg.lstsq(C, R(v), rcond=None)[0]
    a, b = res(x), res(yv); return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))

scorers = {"TeleopQ q = z(V)-z(L)+z(S)": col("q"), "TeleopQ without V (z(S)-z(L))": col("q_noV"),
           "Smoothness S (SPARC)": col("S"), "Visibility V alone": col("V"), "Latency (-L) alone": -col("L"),
           "Action high-freq power (PSD, neg)": col("psd_hf"), "kNN trajectory typicality": col("knn"),
           "Episode length (shorter=better)": col("neg_len"), "Oracle (injected severity)": col("oracle")}
table = []
for name, s in scorers.items():
    table.append({"scorer": name, "rho": spearman(s, y), "ci95": boot(spearman, s, y)})
key = {"rho_q_minus_rho_S": spearman(col("q"), y) - spearman(col("S"), y),
       "ci95": boot(lambda q, s, yy: spearman(q, yy) - spearman(s, yy), col("q"), col("S"), y)}
key["H1_supported"] = key["ci95"][0] > 0
pc = {"partial_rho_V_yield_given_S_and_length": partial_spearman(col("V"), y, col("S"), col("length")),
      "partial_rho_q_yield_given_length": partial_spearman(col("q"), y, col("length")),
      "rho_length_yield": spearman(col("neg_len"), y)}
cond = {}
for i, yy in zip(ids, y): cond.setdefault(scores[i]["tag"], []).append(yy)
res = {"n_seeds": len(ids), "attempts_per_seed": "about 20 (stopped at >= 20)", "yield_mean": float(y.mean()),
       "scorers": table, "key_test_H1": key, "partial": pc,
       "yield_by_condition": {k: [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(cond.items())}}
json.dump(res, open(OUT + "/h1_results.json", "w"), indent=1)
with open(OUT + "/h1_table.md", "w") as fh:
    fh.write(f"Seeds: {len(ids)}; mean yield {y.mean():.3f}\n\n| Scorer | Spearman rho with yield | 95% bootstrap CI |\n|---|---|---|\n")
    for t in table: fh.write(f"| {t['scorer']} | {t['rho']:.3f} | [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}] |\n")
    fh.write(f"\n**Key test (H1):** rho(q) - rho(S) = {key['rho_q_minus_rho_S']:.3f}, 95% CI [{key['ci95'][0]:.3f}, {key['ci95'][1]:.3f}] -> H1 {'supported' if key['H1_supported'] else 'not supported'}\n")
    fh.write(f"\nPartial rho(V, yield | S, length) = {pc['partial_rho_V_yield_given_S_and_length']:.3f}; rho(length, yield) = {pc['rho_length_yield']:.3f}\n")
with open(OUT + "/merged.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["seed_id", "tag", "V", "L", "S", "q", "successes", "attempts", "yield"])
    for i, yy in zip(ids, y): w.writerow([i, scores[i]["tag"], scores[i]["V"], scores[i]["L"], scores[i]["S"], scores[i]["q"], yields[i]["successes"], yields[i]["attempts"], round(float(yy), 4)])
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    occ = [scores[i]["occlusion"] for i in ids]; cmap = {"none": "#2a7", "partial": "#e90", "heavy": "#c33"}
    for a, k, lab in ((ax[0], "q", "TeleopQ score q"), (ax[1], "S", "Smoothness S (SPARC)")):
        a.scatter(col(k), y, c=[cmap[o] for o in occ], s=22)
        a.set_xlabel(lab); a.set_ylabel("Per-seed Mimic yield"); a.set_title(f"rho = {spearman(col(k), y):.2f}")
    fig.suptitle("v3 H1: seed score vs Mimic yield (colour = occlusion level)"); fig.tight_layout(); fig.savefig(OUT + "/q_vs_yield.png", dpi=150)
    print("plot saved")
except Exception as e:
    print("plot skipped:", e)
print(open(OUT + "/h1_table.md").read())
