"""Dataset loader helpers."""
import glob, json, os
import numpy as np


class HEVDataset:
    def __init__(self, root="data"):
        self.root = root
        g = np.load(os.path.join(root, "graph.npz"), allow_pickle=True)
        self.g = {k: g[k] for k in g.files}
        self.names = list(self.g["names"])
        self.N = len(self.names)
        self.n_phys = int(self.g["n_phys"])
        self.mu, self.sd = self.g["mu_nominal"], self.g["sd_nominal"]
        self.runs = sorted(glob.glob(os.path.join(root, "runs", "*.npz")))

    def load(self, path, fields=("X_clean", "X_meas", "fault_id", "fault_level", "t")):
        d = np.load(path, allow_pickle=True)
        out = {k: d[k] for k in fields if k in d.files}
        out["meta"] = json.loads(str(d["meta"]))
        return out

    def split(self, split):
        return [p for p in self.runs if os.path.basename(p).startswith(
            {"train": "nom_train", "test": "nom_test", "fault": "fault"}[split])]

    def z(self, X):
        return (X - self.mu) / np.where(self.sd > 1e-9, self.sd, 1.0)
