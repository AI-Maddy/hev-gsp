"""Pack data/runs/*.npz into HDF5 files (readable from Python h5py and MATLAB h5read).

  hev_gsp_nominal.h5  - nominal train + test runs
  hev_gsp_faults.h5   - fault runs
Each file: /graph/{A, W_signed, W_abs, names, units, layer, edge_i, edge_j, edge_type,
mu_nominal, sd_nominal}, /runs/<id>/{t, X_clean, X_meas, fault_id, fault_level, v_ref, aux/*}
with run metadata as attributes.
"""
import csv, glob, json, os, sys
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hevgsp import graph as G

SRC, OUT = "data", "dist"
os.makedirs(OUT, exist_ok=True)
g = np.load(os.path.join(SRC, "graph.npz"), allow_pickle=True)
nodes = G.node_table()
edges = G.build()["edges"]
kw = dict(compression="gzip", compression_opts=6, shuffle=True)


def write(fname, runs):
    with h5py.File(os.path.join(OUT, fname), "w") as f:
        f.attrs["description"] = ("HEV multilayer graph-signal dataset (power-split HEV physics + AUTOSAR CP/AP "
                                  "zonal E/E layer). 10 Hz, 64 nodes. See README.")
        gg = f.create_group("graph")
        for k in ("A", "W_signed", "W_abs", "mu_nominal", "sd_nominal", "layer"):
            gg.create_dataset(k, data=g[k])
        gg.create_dataset("names", data=np.array([n["name"] for n in nodes], dtype="S"))
        gg.create_dataset("units", data=np.array([n["unit"] for n in nodes], dtype="S"))
        gg.create_dataset("component", data=np.array([n["component"] for n in nodes], dtype="S"))
        gg.create_dataset("edge_i", data=np.array([e[0] for e in edges]))
        gg.create_dataset("edge_j", data=np.array([e[1] for e in edges]))
        gg.create_dataset("edge_type", data=np.array([e[2] for e in edges], dtype="S"))
        gg.attrs["n_phys"] = int(g["n_phys"])
        rg = f.create_group("runs")
        for p in runs:
            d = np.load(p, allow_pickle=True)
            meta = json.loads(str(d["meta"]))
            r = rg.create_group(meta["id"])
            for k in ("t", "X_clean", "X_meas", "fault_level", "v_ref"):
                v = d[k]
                ch = (v.shape[0], 1) if v.ndim == 2 else None   # column chunks compress best
                r.create_dataset(k, data=v, chunks=ch, **kw)
            r.create_dataset("fault_id", data=d["fault_id"])
            a = r.create_group("aux")
            for k in d.files:
                if k.startswith("aux_"):
                    v = d[k]
                    a.create_dataset(k[4:], data=v, chunks=(v.shape[0], 1) if v.ndim == 2 else None, **kw)
            for k in ("cycle", "T_amb", "seed", "split", "duration", "dt"):
                r.attrs[k] = meta[k]
            r.attrs["faults_json"] = json.dumps(meta["faults"])
    print(fname, round(os.path.getsize(os.path.join(OUT, fname)) / 1e6, 1), "MB")


runs = sorted(glob.glob(os.path.join(SRC, "runs", "*.npz")))
PER_FILE = int(os.environ.get("RUNS_PER_FILE", "9"))       # keeps each file under ~25 MB
for tag, sel in (("nominal", [p for p in runs if os.path.basename(p).startswith("nom_")]),
                 ("faults", [p for p in runs if os.path.basename(p).startswith("fault_")])):
    parts = [sel[i:i + PER_FILE] for i in range(0, len(sel), PER_FILE)]
    for k, part in enumerate(parts, 1):
        write(f"hev_gsp_{tag}_part{k:02d}of{len(parts):02d}.h5", part)
