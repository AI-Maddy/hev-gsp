"""Generate the HEV multilayer graph-signal dataset.

Usage:  python generate_dataset.py [--out data] [--workers 2] [--duration 1200]
"""
import argparse, csv, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hevgsp.simulate import run_scenario
from hevgsp.faults import Fault, FAULTS, FAULT_IDS
from hevgsp import graph as G

CYCLES = ["urban", "suburban", "highway", "mixed", "hilly"]
AMBIENTS = [-5, 20, 35]


def scenario_list(seed=2026):
    rng = np.random.default_rng(seed)
    sc = []
    k = 0
    for rep, split in ((0, "train"), (1, "test")):
        for c in CYCLES:
            for T in AMBIENTS:
                sc.append(dict(id=f"nom_{split}_{c}_{T:+d}".replace("+", "p").replace("-", "m"),
                               cycle=c, T_amb=T, seed=1000 + k, faults=[], split=split))
                k += 1
    # fault runs: 15 faults x 2 severities x 2 operating conditions. Ambient and cycle are
    # rotated so that severity is not confounded with ambient temperature or cycle.
    for i, name in enumerate(FAULTS):
        for j, sev in enumerate((0.5, 1.0)):
            for k in range(2):
                T = AMBIENTS[(i + j + 2 * k) % 3]
                c = CYCLES[(2 * i + j + 3 * k) % len(CYCLES)]
                if name == "thermostat_stuck" and c in ("urban", "hilly"):
                    c = ("highway", "suburban")[k]            # needs a warm engine before onset
                t0 = float(np.round(rng.uniform(300, 500), 1))
                sc.append(dict(id=f"fault_{name}_s{int(sev * 100):03d}_{k}", cycle=c, T_amb=T,
                               seed=5000 + 100 * i + 10 * j + k, faults=[(name, t0, sev)], split="fault"))
    return sc


def _run(args):
    s, out, duration = args
    path = os.path.join(out, "runs", s["id"] + ".npz")
    if os.path.exists(path):
        return s["id"], 0.0
    t0 = time.time()
    r = run_scenario(s["cycle"], s["T_amb"], [Fault(*f) for f in s["faults"]], seed=s["seed"],
                     duration=duration)
    meta = r["meta"]; meta.update(id=s["id"], split=s["split"])
    np.savez_compressed(path, t=r["t"], X_clean=r["X_clean"], X_meas=r["X_meas"],
                        fault_id=r["fault_id"], fault_level=r["fault_level"],
                        v_ref=r["v_ref"].astype(np.float32),
                        **{"aux_" + k: v for k, v in r["aux"].items()}, meta=json.dumps(meta))
    return s["id"], time.time() - t0


def write_graph(out, runs):
    g = G.build()
    nm = g["names"]
    # nominal statistics (training nominal runs, clean signals)
    Xs = [np.load(os.path.join(out, "runs", s["id"] + ".npz"))["X_clean"] for s in runs if s["split"] == "train"]
    X = np.vstack(Xs).astype(np.float64)
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    C = np.corrcoef(Z.T)
    A = g["A"]
    W_signed = np.where(A > 0, C, 0.0)
    np.fill_diagonal(W_signed, 0)
    types = np.array(g["types"], dtype="U8")
    np.savez(os.path.join(out, "graph.npz"), names=np.array(nm), A=A, edge_type=types,
             layer=g["layer"], n_phys=g["n_phys"], W_abs=np.abs(W_signed), W_signed=W_signed,
             mu_nominal=mu, sd_nominal=sd, corr_nominal=C,
             edge_types=np.array(G.EDGE_TYPES))
    with open(os.path.join(out, "nodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "name", "unit", "layer", "component", "owner_or_attachment", "nominal_mean", "nominal_std"])
        for i, r in enumerate(G.node_table()):
            w.writerow([i, r["name"], r["unit"], r["layer"], r["component"], r["owner"], f"{mu[i]:.6g}", f"{sd[i]:.6g}"])
    with open(os.path.join(out, "edges.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "source", "target", "type", "layer", "corr_nominal"])
        for i, j, ty in g["edges"]:
            lay = "inter" if ty == "own" else ("physical" if g["layer"][i] == 0 else "ee")
            w.writerow([i, j, nm[i], nm[j], ty, lay, f"{C[i, j]:.4f}"])
    with open(os.path.join(out, "faults.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fault_id", "name", "layer", "component", "primary_node", "component_nodes", "ramp_s", "description"])
        w.writerow([0, "nominal", "", "", "", "", "", "no fault"])
        for n, d in FAULTS.items():
            w.writerow([FAULT_IDS[n], n, d["layer"], d["component"], d["primary"], ";".join(d["nodes"]), d["ramp"], d["desc"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--duration", type=float, default=1200.0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    sc = scenario_list()
    with open(os.path.join(a.out, "scenarios.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "cycle", "T_amb_C", "seed", "fault", "fault_id", "onset_s", "severity"])
        for s in sc:
            fn, t0, sev = s["faults"][0] if s["faults"] else ("", "", "")
            w.writerow([s["id"], s["split"], s["cycle"], s["T_amb"], s["seed"], fn,
                        FAULT_IDS.get(fn, 0), t0, sev])
    t0 = time.time()
    with Pool(a.workers) as p:
        for i, (rid, dt) in enumerate(p.imap_unordered(_run, [(s, a.out, a.duration) for s in sc])):
            print(f"[{i + 1}/{len(sc)}] {rid} {dt:.1f}s", flush=True)
    write_graph(a.out, sc)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
