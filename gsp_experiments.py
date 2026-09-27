"""Baseline GSP experiments on the HEV multilayer dataset.

E1  graph-frequency content / smoothness vs degree-preserving random graphs
E2  denoising (graph, temporal, time-vertex Tikhonov, PCA) vs input SNR
E3  sparse-sensor reconstruction vs sampling ratio
E4  fault detection (AUC, delay) and localization (top-k)
E5  topology inference (PR curves, AUPRC)
Results: results/*.csv, results/*.tex; figures: made by make_figures.py
"""
import json, os, sys
import numpy as np
import networkx as nx
from scipy.fft import dct, idct
from sklearn.covariance import LedoitWolf, GraphicalLasso
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hevgsp.data import HEVDataset
from hevgsp.graph import signed_laplacian
from hevgsp.faults import FAULTS, FAULT_IDS

RES = "results"
os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(0)
D = HEVDataset("data")
nP = D.n_phys
A_all = D.g["A"]
W_all = D.g["W_signed"]
names = D.names


def load_split(split, field="X_clean", step=1):
    out = []
    for p in D.split(split):
        d = D.load(p, (field, "fault_id", "fault_level", "t"))
        out.append((d[field][::step].astype(np.float64), d))
    return out


def nmse_db(x, xh):
    return 10 * np.log10(np.sum((x - xh) ** 2) / np.sum(x ** 2))


# ---------------------------------------------------------------- graphs
A = A_all[:nP, :nP]
W = W_all[:nP, :nP]
Ls = signed_laplacian(W)
Lb = np.diag(A.sum(1)) - A
lam_s, U_s = np.linalg.eigh(Ls)


def rewire(A, W, k, seed):
    G = nx.from_numpy_array(A)
    nx.double_edge_swap(G, nswap=4 * G.number_of_edges(), max_tries=10 ** 6, seed=seed)
    Ar = nx.to_numpy_array(G, nodelist=range(len(A)))
    w = W[np.triu(A, 1) > 0]
    r = np.random.default_rng(seed)
    iu = np.argwhere(np.triu(Ar, 1) > 0)
    Wr = np.zeros_like(W)
    vals = r.permutation(w)[: len(iu)]
    Wr[iu[:, 0], iu[:, 1]] = vals
    return Ar, Wr + Wr.T


# ---------------------------------------------------------------- data
train = load_split("train", step=2)
test = load_split("test")
Ztr = [D.z(x)[:, :nP] for x, _ in train]
Zte = [D.z(x)[:, :nP] for x, _ in test]
Ztr_all = np.vstack(Ztr)
Zte_all = np.vstack(Zte)

# ================================================================ E1
def smooth_stats(L, Z):
    lmax = np.linalg.eigvalsh(L)[-1]
    num = np.einsum("ti,ij,tj->t", Z, L / lmax, Z)
    den = np.einsum("ti,ti->t", Z, Z) + 1e-12
    return float(np.mean(num / den))


def energy_curve(L, Z):
    lam, U = np.linalg.eigh(L)
    E = ((Z @ U) ** 2).mean(0)
    return lam / lam[-1], np.cumsum(E) / E.sum()


Zs = Zte_all[::5]
e1 = {"struct_signed": smooth_stats(Ls, Zs), "struct_binary": smooth_stats(Lb, Zs)}
x_s, c_s = energy_curve(Ls, Zs)
x_b, c_b = energy_curve(Lb, Zs)
rand_s, rand_curves = [], []
for k in range(50):
    Ar, Wr = rewire(A, W, k, k)
    Lr = signed_laplacian(Wr)
    rand_s.append(smooth_stats(Lr, Zs))
    rand_curves.append(energy_curve(Lr, Zs)[1])
rand_curves = np.array(rand_curves)
e1.update(rand_signed_mean=float(np.mean(rand_s)), rand_signed_std=float(np.std(rand_s)),
          z_score=float((np.mean(rand_s) - e1["struct_signed"]) / np.std(rand_s)))
# fraction of energy in lowest 25 % of graph frequencies
q = int(0.25 * nP)
e1.update(E_low25_struct=float(c_s[q - 1]), E_low25_rand=float(rand_curves[:, q - 1].mean()))
# multilayer (all nodes) smoothness, signed Laplacian on supra-graph
Lm = signed_laplacian(W_all)
Zm = np.vstack([D.z(x) for x, _ in test])[::5]
e1["multilayer_signed"] = smooth_stats(Lm, Zm)
rm = []
for k in range(20):
    Ar, Wr = rewire(A_all, W_all, k, 100 + k)
    rm.append(smooth_stats(signed_laplacian(Wr), Zm))
e1.update(multilayer_rand_mean=float(np.mean(rm)), multilayer_rand_std=float(np.std(rm)))
np.savez(os.path.join(RES, "e1_spectrum.npz"), x_s=x_s, c_s=c_s, x_b=x_b, c_b=c_b, rand=rand_curves)
json.dump(e1, open(os.path.join(RES, "e1_smoothness.json"), "w"), indent=2)
print("E1", e1)

# ================================================================ E2 denoising
SNRS = [-5, 0, 5, 10, 15, 20]


def tv_filter(Y, gamma, beta):
    """argmin ||X-Y||^2 + gamma tr(X Ls X^T) + beta ||D_t X||^2 (time x node)."""
    T = Y.shape[0]
    mu = 2 - 2 * np.cos(np.pi * np.arange(T) / T)
    C = dct(Y @ U_s, type=2, norm="ortho", axis=0)
    C /= 1 + gamma * lam_s[None, :] + beta * mu[:, None]
    return idct(C, type=2, norm="ortho", axis=0) @ U_s.T


# PCA subspace from training data
ev, V = np.linalg.eigh(np.cov(Ztr_all.T))
V = V[:, ::-1]


def pca_denoise(Y, k):
    P = V[:, :k]
    return Y @ P @ P.T


def denoise_eval(snr, params, Zs_list, seed):
    r = np.random.default_rng(seed)
    s = 10 ** (-snr / 20)
    num = {m: 0.0 for m in params}
    den = 0.0
    for Z in Zs_list:
        Y = Z + s * r.standard_normal(Z.shape)
        den += np.sum(Z ** 2)
        for m, p in params.items():
            if m == "noisy":
                Xh = Y
            elif m == "pca":
                Xh = pca_denoise(Y, p)
            else:
                Xh = tv_filter(Y, *p)
            num[m] += np.sum((Z - Xh) ** 2)
    return {m: 10 * np.log10(num[m] / den) for m in params}


grid_g = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
grid_b = [0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
grid_k = [3, 5, 8, 12, 16, 20, 25, 30, 35, 40]
val = Ztr[::3]
e2 = []
for snr in SNRS:
    best = {}
    for name, cand in [("graph", [(g, 0) for g in grid_g[1:]]), ("temporal", [(0, b) for b in grid_b[1:]]),
                       ("time_vertex", [(g, b) for g in grid_g for b in grid_b[1:]])]:
        scores = {c: denoise_eval(snr, {"m": c}, val, 7)["m"] for c in cand}
        best[name] = min(scores, key=scores.get)
    pk = {}
    for k in grid_k:
        pk[k] = denoise_eval(snr, {"pca": k}, val, 7)["pca"]
    best["pca"] = min(pk, key=pk.get)
    params = {"noisy": None, "temporal": best["temporal"], "graph": best["graph"],
              "time_vertex": best["time_vertex"], "pca": best["pca"]}
    r = denoise_eval(snr, params, Zte, 11)
    r.update(snr=snr, params={k: v for k, v in params.items()})
    e2.append(r)
    print("E2", snr, {k: round(v, 2) for k, v in r.items() if isinstance(v, float)}, best)
# realistic sensor noise: measured vs clean on test runs (physical layer)
meas = load_split("test", "X_meas")
Zm_list = [D.z(x)[:, :nP] for x, _ in meas]
meas_tr = [D.z(x)[:, :nP] for x, _ in load_split("train", "X_meas", step=2)][::3]


def tune_real(cands):
    sc = {}
    for c in cands:
        num = sum(np.sum((Zc - tv_filter(Ym, *c)) ** 2) for Zc, Ym in zip(val, meas_tr))
        sc[c] = num
    return min(sc, key=sc.get)


rp = {"graph": tune_real([(g, 0) for g in grid_g]), "temporal": tune_real([(0, b) for b in grid_b]),
      "time_vertex": tune_real([(g, b) for g in grid_g for b in grid_b])}
real = {"noisy": 0, "graph": 0, "temporal": 0, "time_vertex": 0}
den = 0
for Zc, Zm_ in zip(Zte, Zm_list):
    den += np.sum(Zc ** 2)
    real["noisy"] += np.sum((Zc - Zm_) ** 2)
    for m, p in rp.items():
        real[m] += np.sum((Zc - tv_filter(Zm_, *p)) ** 2)
real = {k: float(10 * np.log10(v / den)) for k, v in real.items()}
real["params"] = {k: list(v) for k, v in rp.items()}
json.dump({"synthetic": e2, "sensor_noise": real}, open(os.path.join(RES, "e2_denoising.json"), "w"),
          indent=2, default=str)
print("E2 real", real)

# ================================================================ E3 sampling
ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
Sig = np.cov(Ztr_all.T) + 1e-6 * np.eye(nP)
Zc3 = np.vstack(Zte)[::20]
Zm3 = np.vstack(Zm_list)[::20]


def recon(method, S, Yobs, gamma=0.3, K=None):
    Uo = np.setdiff1d(np.arange(nP), S)
    if method == "mean":
        Xh = np.zeros((Yobs.shape[0], nP))
        Xh[:, S] = Yobs
        return Xh
    if method == "tikhonov":
        M = np.zeros((len(S), nP)); M[np.arange(len(S)), S] = 1
        H = np.linalg.solve(M.T @ M + gamma * Ls + 1e-3 * np.eye(nP), M.T)
        return Yobs @ H.T
    if method == "bandlimited":
        UK = U_s[:, :K]
        return Yobs @ np.linalg.pinv(UK[S]).T @ UK.T
    if method == "lmmse":
        Xh = np.zeros((Yobs.shape[0], nP))
        Xh[:, S] = Yobs
        Xh[:, Uo] = Yobs @ np.linalg.solve(Sig[np.ix_(S, S)] + 0.01 * np.eye(len(S)), Sig[np.ix_(S, Uo)])
        return Xh


def greedy_sampling(M, K):
    """Greedy E-optimal design on the K lowest graph frequencies."""
    UK = U_s[:, :K]
    S = []
    for _ in range(M):
        best, bv = None, -np.inf
        for i in range(nP):
            if i in S:
                continue
            sv = np.linalg.svd(UK[S + [i]], compute_uv=False)
            v = sv[min(len(S), K - 1)] if len(S) + 1 <= K else sv[-1]
            if v > bv:
                best, bv = i, v
        S.append(best)
    return np.array(S)


Zc3v = np.vstack(Ztr)[::40]
Zm3v = np.vstack([D.z(x)[:, :nP] for x, _ in load_split("train", "X_meas", step=2)])[::40]


def tune_e3(Mn):
    best_g, best_k = None, None
    sg, sk = {}, {}
    for draw in range(8):
        S = np.sort(np.random.default_rng(900 + draw).choice(nP, Mn, replace=False))
        Uo = np.setdiff1d(np.arange(nP), S)
        for g in [0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
            sg[g] = sg.get(g, 0) + nmse_db(Zc3v[:, Uo], recon("tikhonov", S, Zm3v[:, S], gamma=g)[:, Uo])
        for K in range(1, Mn + 1):
            sk[K] = sk.get(K, 0) + nmse_db(Zc3v[:, Uo], recon("bandlimited", S, Zm3v[:, S], K=K)[:, Uo])
    return min(sg, key=sg.get), min(sk, key=sk.get)


e3 = []
for rr in ratios:
    Mn = max(2, int(round(rr * nP)))
    res = {m: [] for m in ["mean", "tikhonov", "bandlimited", "lmmse", "tikhonov_greedy"]}
    g_opt, K_opt = tune_e3(Mn)
    for draw in range(20):
        S = np.sort(rng.choice(nP, Mn, replace=False))
        Uo = np.setdiff1d(np.arange(nP), S)
        y = Zm3[:, S]
        for m in ["mean", "tikhonov", "lmmse"]:
            Xh = recon(m, S, y, gamma=g_opt)
            res[m].append(nmse_db(Zc3[:, Uo], Xh[:, Uo]))
        Xh = recon("bandlimited", S, y, K=K_opt)
        res["bandlimited"].append(nmse_db(Zc3[:, Uo], Xh[:, Uo]))
    Sg = np.sort(greedy_sampling(Mn, max(2, K_opt)))
    Uo = np.setdiff1d(np.arange(nP), Sg)
    res["tikhonov_greedy"].append(nmse_db(Zc3[:, Uo], recon("tikhonov", Sg, Zm3[:, Sg], gamma=g_opt)[:, Uo]))
    row = {"ratio": rr, "gamma": g_opt, "K": K_opt, **{m: float(np.mean(v)) for m, v in res.items()},
           **{m + "_std": float(np.std(v)) for m, v in res.items()}}
    e3.append(row)
    print("E3", {k: round(v, 2) for k, v in row.items()})
json.dump(e3, open(os.path.join(RES, "e3_sampling.json"), "w"), indent=2)

# ================================================================ E4 fault detection
Ltr = [D.z(d["X_meas"].astype(np.float64)) for d in
       [D.load(p, ("X_meas",)) for p in D.split("train")]]
Xn_tr = np.vstack(Ltr)
N = D.N
Lm_ = signed_laplacian(W_all)
Hlp = np.linalg.inv(np.eye(N) + 1.0 * Lm_)
# neighbour regression (graph-constrained linear predictor), ridge
nbr = [np.where(A_all[i] > 0)[0] for i in range(N)]
coef = []
for i in range(N):
    Xi = Xn_tr[:, nbr[i]]
    b = np.linalg.solve(Xi.T @ Xi + 10.0 * np.eye(len(nbr[i])), Xi.T @ Xn_tr[:, i])
    coef.append(b)
evp, Vp = np.linalg.eigh(np.cov(Xn_tr.T))
Vp = Vp[:, ::-1]
kp = int(np.searchsorted(np.cumsum(evp[::-1]) / evp.sum(), 0.95)) + 1
Pp = Vp[:, :kp]


def residuals(Z):
    R = {}
    R["zscore"] = Z
    R["pca_spe"] = Z - Z @ Pp @ Pp.T
    R["gsp_highpass"] = Z - Z @ Hlp.T
    R["gsp_nbr_reg"] = np.column_stack([Z[:, i] - Z[:, nbr[i]] @ coef[i] for i in range(N)])
    return R


# per-node residual scale from training
scale = {m: np.sqrt((r ** 2).mean(0)) + 1e-9 for m, r in residuals(Xn_tr).items()}
WIN = 50


def smooth(x):
    k = np.ones(WIN) / WIN
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="full")[: len(c)], 0, x)


def node_stats(Z):
    return {m: smooth((r / scale[m]) ** 2) for m, r in residuals(Z).items()}


METHODS = ["zscore", "pca_spe", "gsp_highpass", "gsp_nbr_reg"]


def agg(Snode, k=3):
    """Detection statistic: mean of the k largest per-node statistics."""
    return np.sort(Snode, axis=1)[:, -k:].mean(1)
neg = {m: [] for m in METHODS}
for p in D.split("test"):
    Z = D.z(D.load(p, ("X_meas",))["X_meas"].astype(np.float64))
    S = node_stats(Z)
    for m in METHODS:
        neg[m].append(agg(S[m])[WIN:])
neg = {m: np.concatenate(v) for m, v in neg.items()}
thr = {m: np.quantile(neg[m], 0.99) for m in METHODS}
rows, loc_rows = [], []
per_fault = {}
for p in D.split("fault"):
    d = D.load(p, ("X_meas", "fault_level", "t"))
    meta = d["meta"]["faults"][0]
    Z = D.z(d["X_meas"].astype(np.float64))
    S = node_stats(Z)
    lvl = d["fault_level"] / meta["severity"]
    pos = lvl >= 0.5
    pre = d["t"] < meta["t0"]
    pre[:WIN] = False
    comp = [names.index(n) for n in meta["nodes"]]
    EX = {"fault_sens_bias_yaw_s100_0.npz": ("yaw_rate", "a_lat", "steer_angle"),
          "fault_brake_drag_F_s100_0.npz": ("T_brake_F", "T_brake_R", "brake_pressure"),
          "fault_misfire_s100_0.npz": ("tq_eng", "tq_MG1", "T_cat"),
          "fault_edu_pump_s100_0.npz": ("T_cool_edu", "T_inv", "T_MG2")}
    if os.path.basename(p) in EX:
        sel = EX[os.path.basename(p)]
        np.savez(os.path.join(RES, "e4_example_" + meta["name"] + ".npz"), t=d["t"], t0=meta["t0"],
                 labels=np.array(sel), Z=Z[:, [names.index(n) for n in sel]],
                 **{"score_" + m: agg(S[m]) for m in METHODS}, **{"thr_" + m: thr[m] for m in METHODS})
    for m in METHODS:
        sc = agg(S[m])
        for key in ((meta["name"], m), (meta["name"], m, meta["severity"])):
            per_fault.setdefault(key, {"pos": [], "neg": []})
            per_fault[key]["pos"].append(sc[pos])
            per_fault[key]["neg"].append(sc[pre])
        al = np.where((sc > thr[m]) & (d["t"] >= meta["t0"]))[0]
        delay = float(d["t"][al[0]] - meta["t0"]) if len(al) else np.nan
        node_sc = S[m][pos].mean(0) / (S[m][pre].mean(0) + 1e-9) if pre.any() else S[m][pos].mean(0)
        order = np.argsort(-node_sc)
        rows.append(dict(run=meta["name"], severity=meta["severity"], method=m, delay_s=delay,
                         detected=bool(len(al))))
        loc_rows.append(dict(run=meta["name"], severity=meta["severity"], method=m,
                             top1=bool(order[0] in comp), top3=bool(np.isin(order[:3], comp).any()),
                             predicted=names[order[0]]))
auc = {}
for key, v in per_fault.items():
    m = key[1]
    pos_ = np.concatenate(v["pos"]); neg_ = np.concatenate([neg[m]] + v["neg"])
    y = np.r_[np.ones(len(pos_)), np.zeros(len(neg_))]
    auc[key] = roc_auc_score(y, np.r_[pos_, neg_])
with open(os.path.join(RES, "e4_detection_auc.csv"), "w") as fh:
    fh.write("fault,layer," + ",".join(METHODS) + "\n")
    for f in FAULTS:
        fh.write(f"{f},{FAULTS[f]['layer']}," + ",".join(f"{auc[(f, m)]:.4f}" for m in METHODS) + "\n")
with open(os.path.join(RES, "e4_detection_auc_by_severity.csv"), "w") as fh:
    fh.write("fault,severity," + ",".join(METHODS) + "\n")
    for f in FAULTS:
        for sv in (0.5, 1.0):
            fh.write(f"{f},{sv}," + ",".join(f"{auc[(f, m, sv)]:.4f}" for m in METHODS) + "\n")
with open(os.path.join(RES, "e4_runs.json"), "w") as fh:
    json.dump({"detection": rows, "localization": loc_rows, "threshold_far": 0.01,
               "pca_components": kp}, fh, indent=1, default=float)
for m in METHODS:
    a = np.mean([auc[(f, m)] for f in FAULTS])
    lr = [r for r in loc_rows if r["method"] == m]
    dr = [r for r in rows if r["method"] == m]
    print(f"E4 {m:13s} meanAUC {a:.3f} top1 {np.mean([r['top1'] for r in lr]):.2f} "
          f"top3 {np.mean([r['top3'] for r in lr]):.2f} detected {np.mean([r['detected'] for r in dr]):.2f} "
          f"median delay {np.nanmedian([r['delay_s'] for r in dr]):.1f}s")

# ================================================================ E5 topology inference
iu = np.triu_indices(nP, 1)
truth = A[iu] > 0
Zlev = np.vstack([D.z(x)[:, :nP] for x, _ in load_split("train", "X_meas", step=5)])
Zdif = np.vstack([np.diff(D.z(x)[:, :nP], axis=0) for x, _ in load_split("train", "X_meas", step=1)])[::5]


def kalofolias(Z, alpha=1.0, beta=0.5, iters=3000, lr=0.002):
    """min 2 w.z - alpha 1^T log(S w) + beta ||w||^2, w >= 0 (Kalofolias 2016), projected gradient."""
    Zn = Z / (np.linalg.norm(Z, axis=0) + 1e-12)
    G = Zn.T @ Zn
    dist = (np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G)[iu]
    dist = dist / dist.mean()
    m = len(dist)
    rows_ = np.concatenate([iu[0], iu[1]]); cols_ = np.concatenate([np.arange(m)] * 2)
    import scipy.sparse as sp
    S = sp.csr_matrix((np.ones(2 * m), (rows_, cols_)), shape=(nP, m))
    w = np.full(m, 0.1)
    for _ in range(iters):
        deg = S @ w + 1e-9
        g = 2 * dist - alpha * (S.T @ (1 / deg)) + 2 * beta * w
        w = np.maximum(w - lr * g, 0)
    return w


e5 = {}
curves = {}
for rep, Zr in [("levels", Zlev), ("differences", Zdif)]:
    C = np.corrcoef(Zr.T)
    P = np.linalg.inv(LedoitWolf().fit(Zr).covariance_)
    pc = -P / np.sqrt(np.outer(np.diag(P), np.diag(P)))
    from sklearn.covariance import graphical_lasso
    Clw = LedoitWolf().fit(Zr).covariance_
    Pg = None
    for a_ in (0.01, 0.02, 0.05, 0.1):
        try:
            Pg = graphical_lasso(Clw, alpha=a_, max_iter=300)[1]
            break
        except FloatingPointError:
            continue
    pg = -Pg / np.sqrt(np.outer(np.diag(Pg), np.diag(Pg)))
    wk = kalofolias(Zr)
    scores = {"correlation": np.abs(C[iu]), "partial_corr_LW": np.abs(pc[iu]),
              "graphical_lasso": np.abs(pg[iu]), "smoothness_Kalofolias": wk}
    for mth, s in scores.items():
        ap = average_precision_score(truth, s)
        au = roc_auc_score(truth, s)
        e5[f"{rep}/{mth}"] = dict(AUPRC=float(ap), AUROC=float(au))
        pr, rc, _ = precision_recall_curve(truth, s)
        curves[f"{rep}/{mth}"] = (rc, pr)
e5["chance_AUPRC"] = float(truth.mean())
# per-edge-type recall at the top-|E| cutoff for the best representation
best = max((k for k in e5 if "/" in k), key=lambda k: e5[k]["AUPRC"])
e5["best"] = best
json.dump(e5, open(os.path.join(RES, "e5_topology.json"), "w"), indent=2)
np.savez(os.path.join(RES, "e5_curves.npz"), **{k.replace("/", "__") + "__rc": v[0] for k, v in curves.items()},
         **{k.replace("/", "__") + "__pr": v[1] for k, v in curves.items()})
print("E5", json.dumps(e5, indent=1))
