"""Fleet (federated) learning and graph-based sensor self-healing on HEV-GSP.

F1  Federated graph residual detectors. Each of the 15 training nominal runs is
    one vehicle (one cycle x ambient condition). Each vehicle fits the graph
    neighbour-regression detector on its own data. We compare
      local    - a vehicle uses only its own model and normalisation,
      FedAvg   - vehicles share only regression coefficients and moment sums,
                 averaged with sample weights (one communication round),
      FedGram  - vehicles share per-node Gram sums (sufficient statistics);
                 equals centralised ridge exactly,
      central  - pooled raw data (upper reference).
    Evaluated on every fault run with the model of the vehicle whose operating
    condition matches (own-condition) and, for local, also a mismatched vehicle.
F2  Sensor self-healing. On sensor-fault runs, the detector localises the
    faulty node over the post-onset window; its measurement is then replaced
    by the graph neighbour prediction (virtual sensor). Error is measured
    against the clean ground truth.
Usage: python fleet_experiments.py --data data
"""
import argparse, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hevgsp.data import HEVDataset
from hevgsp.faults import FAULTS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="results")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
D = HEVDataset(a.data)
A = D.g["A"]
N = D.N
names = D.names
nbr = [np.where(A[i] > 0)[0] for i in range(N)]
LAM = 10.0
WIN = 50


def load(p, field="X_meas"):
    d = D.load(p, (field, "X_clean", "fault_level", "t"))
    return d


# ---------------------------------------------------------------- vehicles
veh = []
for p in D.split("train"):
    d = D.load(p, ("X_meas",))
    X = d["X_meas"].astype(np.float64)[::2]
    veh.append(dict(cond=(d["meta"]["cycle"], d["meta"]["T_amb"]), X=X, n=len(X),
                    s1=X.sum(0), s2=(X ** 2).sum(0)))
print("vehicles", len(veh))


def stats_from(vs):
    n = sum(v["n"] for v in vs)
    mu = sum(v["s1"] for v in vs) / n
    var = sum(v["s2"] for v in vs) / n - mu ** 2
    return mu, np.sqrt(np.maximum(var, 1e-12))


def z(X, mu, sd):
    return (X - mu) / np.where(sd > 1e-6, sd, 1.0)


def fit_ridge(Z):
    return [np.linalg.solve(Z[:, nbr[i]].T @ Z[:, nbr[i]] + LAM * np.eye(len(nbr[i])),
                            Z[:, nbr[i]].T @ Z[:, i]) for i in range(N)]


def residual(Z, coef):
    return np.column_stack([Z[:, i] - Z[:, nbr[i]] @ coef[i] for i in range(N)])


def model_local(v):
    mu, sd = stats_from([v])
    Z = z(v["X"], mu, sd)
    coef = fit_ridge(Z)
    sc = np.sqrt((residual(Z, coef) ** 2).mean(0)) + 1e-9
    return dict(mu=mu, sd=sd, coef=coef, sc=sc)


# global moments are aggregated federatively (sums only)
mu_g, sd_g = stats_from(veh)
# FedAvg: each vehicle fits on globally normalised local data, shares coefficients + residual moments
local_g = []
for v in veh:
    Z = z(v["X"], mu_g, sd_g)
    c = fit_ridge(Z)
    r2 = (residual(Z, c) ** 2).sum(0)
    local_g.append((c, r2, v["n"]))
ntot = sum(n for *_, n in local_g)
coef_avg = [sum(c[i] * n for c, _, n in local_g) / ntot for i in range(N)]
# residual scale needs a second round: vehicles report residual energy under the averaged model
r2_avg = sum((residual(z(v["X"], mu_g, sd_g), coef_avg) ** 2).sum(0) for v in veh)
M_fedavg = dict(mu=mu_g, sd=sd_g, coef=coef_avg, sc=np.sqrt(r2_avg / ntot) + 1e-9)
# FedGram: share per-node Gram blocks (sufficient statistics) -> exact centralised ridge
G = [np.zeros((len(nbr[i]), len(nbr[i]))) for i in range(N)]
b = [np.zeros(len(nbr[i])) for i in range(N)]
for v in veh:
    Z = z(v["X"], mu_g, sd_g)
    for i in range(N):
        G[i] += Z[:, nbr[i]].T @ Z[:, nbr[i]]
        b[i] += Z[:, nbr[i]].T @ Z[:, i]
coef_gram = [np.linalg.solve(G[i] + LAM * np.eye(len(nbr[i])), b[i]) for i in range(N)]
r2_g = sum((residual(z(v["X"], mu_g, sd_g), coef_gram) ** 2).sum(0) for v in veh)
M_gram = dict(mu=mu_g, sd=sd_g, coef=coef_gram, sc=np.sqrt(r2_g / ntot) + 1e-9)
# central: pooled raw data (same algebra as FedGram; computed independently as a check)
Xall = np.vstack([v["X"] for v in veh])
Zall = z(Xall, mu_g, sd_g)
coef_c = fit_ridge(Zall)
M_central = dict(mu=mu_g, sd=sd_g, coef=coef_c,
                 sc=np.sqrt((residual(Zall, coef_c) ** 2).mean(0)) + 1e-9)
print("FedGram vs central max coef diff", max(np.abs(coef_gram[i] - coef_c[i]).max() for i in range(N)))
M_local = [model_local(v) for v in veh]


def model_personal(v, lam_p):
    """Personalised: local normalisation, ridge coefficients shrunk toward the fleet (FedGram) model."""
    mu, sd = stats_from([v])
    Z = z(v["X"], mu, sd)
    coef = [np.linalg.solve(Z[:, nbr[i]].T @ Z[:, nbr[i]] + (LAM + lam_p * len(Z)) * np.eye(len(nbr[i])),
                            Z[:, nbr[i]].T @ Z[:, i] + lam_p * len(Z) * coef_gram[i]) for i in range(N)]
    sc = np.sqrt((residual(Z, coef) ** 2).mean(0)) + 1e-9
    return dict(mu=mu, sd=sd, coef=coef, sc=sc)


LAM_P = float(os.environ.get("LAM_P", "1.0"))
M_pers = [model_personal(v, LAM_P) for v in veh]


def smooth(x):
    k = np.ones(WIN) / WIN
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="full")[: len(c)], 0, x)


def node_stat(X, M):
    return smooth((residual(z(X, M["mu"], M["sd"]), M["coef"]) / M["sc"]) ** 2)


def agg(S, k=3):
    return np.sort(S, axis=1)[:, -k:].mean(1)


cond_idx = {v["cond"]: k for k, v in enumerate(veh)}
# negatives: nominal test runs scored by each model
test = [D.load(p, ("X_meas",)) for p in D.split("test")]


test_by_cond = {(d["meta"]["cycle"], d["meta"]["T_amb"]): d["X_meas"].astype(np.float64) for d in test}
rng = np.random.default_rng(0)
rows = []
per = {}
for p in D.split("fault"):
    d = D.load(p, ("X_meas", "fault_level", "t"))
    m = d["meta"]
    f = m["faults"][0]
    X = d["X_meas"].astype(np.float64)
    pos = d["fault_level"] / f["severity"] >= 0.5
    pre = d["t"] < f["t0"]
    pre[:WIN] = False
    own = cond_idx[(m["cycle"], m["T_amb"])]
    other = [k for k in range(len(veh)) if veh[k]["cond"][1] != m["T_amb"] and veh[k]["cond"][0] != m["cycle"]]
    oth = int(rng.choice(other))
    models = {"local_own": M_local[own], "local_mismatch": M_local[oth], "personal_own": M_pers[own],
              "personal_mismatch": M_pers[oth], "fedavg": M_fedavg,
              "fedgram": M_gram, "central": M_central}
    for k, M in models.items():
        sc = agg(node_stat(X, M))
        sn = agg(node_stat(test_by_cond[(m["cycle"], m["T_amb"])], M))[WIN:]
        per.setdefault((f["name"], k), {"pos": [], "neg": []})
        per[(f["name"], k)]["neg"].append(sn)
        per[(f["name"], k)]["pos"].append(sc[pos])
        per[(f["name"], k)]["neg"].append(sc[pre])
METH = ["local_own", "local_mismatch", "personal_own", "personal_mismatch", "fedavg", "fedgram", "central"]
auc = {}
for (fn, k), v in per.items():
    p_ = np.concatenate(v["pos"]); n_ = np.concatenate(v["neg"])
    auc[(fn, k)] = roc_auc_score(np.r_[np.ones(len(p_)), np.zeros(len(n_))], np.r_[p_, n_])
summary = {k: float(np.mean([auc[(fn, k)] for fn in FAULTS])) for k in METH}
by_layer = {k: {L: float(np.mean([auc[(fn, k)] for fn in FAULTS if FAULTS[fn]["layer"] == L]))
                for L in ("physical", "sensor", "ee")} for k in METH}
comm = {"fedavg_floats_per_vehicle": int(sum(len(x) for x in nbr) + 3 * N),
        "fedgram_floats_per_vehicle": int(sum(len(x) * (len(x) + 1) for x in nbr) + 2 * N),
        "raw_floats_per_vehicle": int(veh[0]["n"] * N)}
print("F1 mean AUC", {k: round(v, 3) for k, v in summary.items()})
print("F1 by layer", json.dumps(by_layer, indent=0))
print("F1 comm", comm)
with open(os.path.join(a.out, "f1_federated.json"), "w") as fh:
    json.dump(dict(mean_auc=summary, by_layer=by_layer, comm=comm, n_vehicles=len(veh),
                   per_fault={f"{fn}|{k}": v for (fn, k), v in auc.items()}), fh, indent=1)

# ---------------------------------------------------------------- F2 self-healing
M = M_central
lm = np.linalg.pinv(np.cov(Zall.T) + 1e-6 * np.eye(N))      # precision for LMMSE from all other nodes
heal = []
for p in D.split("fault"):
    d = D.load(p, ("X_meas", "X_clean", "fault_level", "t"))
    f = d["meta"]["faults"][0]
    if FAULTS[f["name"]]["layer"] != "sensor":
        continue
    X = d["X_meas"].astype(np.float64)
    Xc = d["X_clean"].astype(np.float64)
    Z = z(X, M["mu"], M["sd"])
    S = node_stat(X, M)
    post = (d["t"] >= f["t0"] + 60)                      # after a 60 s diagnosis window
    pre = (d["t"] < f["t0"]) & (d["t"] > 5)
    ratio = S[post].mean(0) / (S[pre].mean(0) + 1e-9)
    # isolation by hypothesis replacement: the node whose replacement by its graph
    # prediction most reduces the normalised residual energy of its closed neighbourhood
    Zp = Z[post]
    R0 = (residual(Zp, M["coef"]) / M["sc"]) ** 2
    Rpre = (residual(Z[pre], M["coef"]) / M["sc"]) ** 2
    gain = np.zeros(N)
    for c in range(N):
        Zh = Zp.copy()
        Zh[:, c] = Zp[:, nbr[c]] @ M["coef"][c]
        hood = np.r_[c, nbr[c]]
        Rh = np.column_stack([(Zh[:, i] - Zh[:, nbr[i]] @ M["coef"][i]) / M["sc"][i] for i in hood]) ** 2
        base = Rpre[:, hood].mean(0).sum()
        gain[c] = (R0[:, hood].mean(0).sum() - Rh.mean(0).sum()) / (base + 1e-9)
    j_repl = int(np.argmax(gain))                        # alternative isolation (reported only)
    j = j_ratio = int(np.argmax(ratio))                  # localised node
    true_j = names.index(f["primary"])
    # graph virtual sensor: neighbour regression (neighbours exclude the faulty node itself)
    zg = Z[:, nbr[j]] @ M["coef"][j]
    # data-driven virtual sensor: LMMSE of node j from all other nodes
    zl = -(Z @ lm[j] - lm[j, j] * Z[:, j]) / lm[j, j]
    tj = true_j
    to_phys = lambda zz, jj: zz * M["sd"][jj] + M["mu"][jj]
    err = lambda y: float(np.sqrt(np.mean((y[post] - Xc[post, tj]) ** 2)))
    raw = err(X[:, tj])
    healed_g = err(to_phys(zg, j)) if j == tj else raw
    healed_l = err(to_phys(zl, j)) if j == tj else raw
    heal.append(dict(run=d["meta"]["id"], fault=f["name"], severity=f["severity"], localised=names[j],
                     localised_by_replacement=names[j_repl], correct_by_replacement=bool(j_repl == tj),
                     correct=bool(j == tj), rmse_faulty=raw, rmse_graph=healed_g, rmse_lmmse=healed_l,
                     signal_std=float(Xc[:, tj].std())))
    print(f"F2 {d['meta']['id']:32s} repl-loc={names[j_repl]:10s} loc={names[j]:10s} ok={j == tj} raw={raw:.3f} graph={healed_g:.3f} lmmse={healed_l:.3f} (sd {Xc[:, tj].std():.3f})")
with open(os.path.join(a.out, "f2_self_healing.json"), "w") as fh:
    json.dump(heal, fh, indent=1)
