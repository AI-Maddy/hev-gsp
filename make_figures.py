"""IEEE-ready figures (PDF vector + 300 dpi PNG) from results/ and data/."""
import json, os, sys
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hevgsp import graph as G
from hevgsp.faults import FAULTS

OUT = "figures"; RES = "results"
os.makedirs(OUT, exist_ok=True)
# categorical slots in fixed order (validated palette), plus ink tokens
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
MK = ["o", "s", "^", "D", "v", "P"]
LS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
COL1, COL2 = 3.5, 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "lines.markersize": 4, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 1 multilayer graph
def fig_graph():
    g = G.build()
    nm, A, T = g["names"], g["A"], g["types"]
    nP = g["n_phys"]
    Gp = nx.from_numpy_array(A[:nP, :nP])
    pos_p = nx.kamada_kawai_layout(Gp)
    P = np.array([pos_p[i] for i in range(nP)])
    P = (P - P.min(0)) / (P.max(0) - P.min(0))
    P[:, 0] = P[:, 0] * 1.0; P[:, 1] = 1.2 + P[:, 1] * 1.05
    # E/E layer: manual zonal layout
    ee = {"CANFD_F": (0.18, 0.42), "CANFD_R": (0.82, 0.42), "ETH_BB": (0.5, 0.42),
          "ZC_FRONT": (0.34, 0.42), "ZC_REAR": (0.66, 0.42), "CVC": (0.5, 0.70), "TCU": (0.5, 0.14)}
    fr = ["ECM", "PCU", "ESC", "EPS", "DCDC_ECU", "TMS"]
    rr = ["BMS", "SCU", "TPMS", "CCU"]
    for k, e in enumerate(fr):
        a = np.pi * (0.55 + 0.9 * k / (len(fr) - 1))
        ee[e] = (0.18 + 0.15 * np.cos(a), 0.42 + 0.30 * np.sin(a))
    for k, e in enumerate(rr):
        a = np.pi * (0.45 - 0.9 * k / (len(rr) - 1))
        ee[e] = (0.82 + 0.15 * np.cos(a), 0.42 + 0.30 * np.sin(a))
    pos = {i: tuple(P[i]) for i in range(nP)}
    for i, n in enumerate(nm[nP:], start=nP):
        pos[i] = ee[n]
    fig, ax = plt.subplots(figsize=(COL2, 4.3))
    ax.set_axis_off()
    for y0, y1, lab in [(1.13, 2.36, "Physical layer (47 sensor nodes)"), (0.0, 0.98, "E/E layer (AUTOSAR CP ECUs, zonal gateways, CAPI AP machines, buses)")]:
        ax.add_patch(plt.Rectangle((-0.04, y0), 1.08, y1 - y0, fc="#f6f6f4", ec=GRID, lw=0.6, zorder=0))
        ax.text(-0.03, y1 - 0.03, lab, ha="left", va="top", fontsize=7, color=INK2, style="italic")
    sty = {"own": dict(color=MUTED, lw=0.35, ls=":", alpha=0.6), "canfd": dict(color=INK2, lw=0.8),
           "eth": dict(color=INK, lw=1.3), "svc": dict(color=C[6], lw=0.8, ls="--")}
    for i, j, ty in g["edges"]:
        s = sty.get(ty, dict(color="#9aa7b8", lw=0.45))
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], zorder=1, solid_capstyle="round", **s)
    kinds = {n: k for n, _, k, _ in G.EE_NODES}
    comp = {n: c for n, _, c, _ in G.PHYS_NODES}
    for i, n in enumerate(nm):
        x, y = pos[i]
        if i < nP:
            ax.scatter(x, y, s=16, c=C[0], edgecolors="white", linewidths=0.6, zorder=3)
            ax.text(x, y + 0.025, n, fontsize=4.4, ha="center", va="bottom", color=INK, zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="#f6f6f4")])
        else:
            k = kinds[n]
            mk, col, sz = {"cp_ecu": ("o", C[1], 60), "cp_zonal_gw": ("s", C[1], 80),
                           "ap_machine": ("D", C[2], 80), "bus": ("h", INK2, 90)}[k]
            ax.scatter(x, y, s=sz, marker=mk, c=col, edgecolors="white", linewidths=0.8, zorder=3)
            dx, dy, ha, va = {"CVC": (0.035, 0, "left", "center"), "TCU": (0.035, 0, "left", "center"),
                              "CANFD_F": (0, -0.06, "center", "top"), "CANFD_R": (0, -0.06, "center", "top"),
                              "ETH_BB": (0.0, -0.06, "center", "top"), "ZC_FRONT": (0, 0.06, "center", "bottom"),
                              "ZC_REAR": (0, 0.06, "center", "bottom")}.get(
                n, (0, 0.055, "center", "bottom") if y > 0.42 else (0, -0.055, "center", "top"))
            ax.text(x + dx, y + dy, n.replace("_ECU", ""), fontsize=5.6, ha=ha, va=va,
                    color=INK, zorder=4, path_effects=[pe.withStroke(linewidth=2, foreground="#f6f6f4")])
    leg = [Line2D([], [], marker="o", ls="", mfc=C[0], mec="white", ms=5, label="sensor node"),
           Line2D([], [], marker="o", ls="", mfc=C[1], mec="white", ms=6, label="CP ECU"),
           Line2D([], [], marker="s", ls="", mfc=C[1], mec="white", ms=6, label="CP zonal gateway"),
           Line2D([], [], marker="D", ls="", mfc=C[2], mec="white", ms=6, label="AP machine (CAPI)"),
           Line2D([], [], marker="h", ls="", mfc=INK2, mec="white", ms=6, label="bus"),
           Line2D([], [], color="#9aa7b8", lw=0.8, label="physical coupling"),
           Line2D([], [], color=INK2, lw=0.8, label="CAN FD"),
           Line2D([], [], color=INK, lw=1.3, label="100BASE-T1"),
           Line2D([], [], color=C[6], lw=0.8, ls="--", label="ara::com service"),
           Line2D([], [], color=MUTED, lw=0.6, ls=":", label="sensor ownership")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.005), ncol=5, fontsize=6.5,
              handletextpad=0.4, columnspacing=1.2)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.02, 2.38)
    save(fig, "fig1_multilayer_graph")


# ------------------------------------------------------------------ Fig 2 spectrum
def fig_spectrum():
    d = np.load(os.path.join(RES, "e1_spectrum.npz"))
    e1 = json.load(open(os.path.join(RES, "e1_smoothness.json")))
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    x = np.arange(1, len(d["c_s"]) + 1) / len(d["c_s"])
    r = d["rand"]
    ax.fill_between(x, np.percentile(r, 2.5, 0), np.percentile(r, 97.5, 0), color=C[1], alpha=0.18, lw=0)
    ax.plot(x, r.mean(0), color=C[1], ls="--", label="degree-preserving random (95% band)")
    ax.plot(x, d["c_s"], color=C[0], label="structural graph, signed weights")
    ax.plot(x, d["c_b"], color=C[2], ls="-.", label="structural graph, binary")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("Graph-frequency index (fraction of $N$, ascending $\\lambda$)")
    ax.set_ylabel("Cumulative GFT energy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    ax.text(0.03, 0.93, f"TV$_{{signed}}$ = {e1['struct_signed']:.3f} vs {e1['rand_signed_mean']:.3f}$\\pm${e1['rand_signed_std']:.3f}",
            transform=ax.transAxes, fontsize=6.5, color=INK2)
    save(fig, "fig2_graph_spectrum")


# ------------------------------------------------------------------ Fig 3 denoising
def fig_denoise():
    e2 = json.load(open(os.path.join(RES, "e2_denoising.json")))["synthetic"]
    snr = [r["snr"] for r in e2]
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    ms = [("noisy", "no filtering"), ("temporal", "temporal Tikhonov"), ("graph", "graph Tikhonov"),
          ("pca", "PCA subspace (data-driven)"), ("time_vertex", "time-vertex Tikhonov")]
    for k, (m, lab) in enumerate(ms):
        ax.plot(snr, [r[m] for r in e2], color=C[k], ls=LS[k], marker=MK[k], label=lab)
    ax.set_xlabel("Input SNR (dB)"); ax.set_ylabel("NMSE (dB)")
    ax.legend(loc="upper right")
    save(fig, "fig3_denoising")


# ------------------------------------------------------------------ Fig 4 sampling
def fig_sampling():
    e3 = json.load(open(os.path.join(RES, "e3_sampling.json")))
    r = [x["ratio"] for x in e3]
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    ms = [("bandlimited", "bandlimited LS (random $\\mathcal{S}$)"),
          ("tikhonov", "graph Tikhonov (random $\\mathcal{S}$)"),
          ("tikhonov_greedy", "graph Tikhonov (greedy E-opt. $\\mathcal{S}$)"),
          ("lmmse", "LMMSE / kriging (data covariance)")]
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.text(0.8, 0.4, "mean imputation", ha="right", va="bottom", fontsize=6, color=INK2)
    for k, (m, lab) in enumerate(ms):
        y = np.array([x[m] for x in e3])
        ax.plot(r, y, color=C[k], ls=LS[k], marker=MK[k], label=lab)
        if m in ("tikhonov", "lmmse"):
            s_ = np.array([x[m + "_std"] for x in e3])
            ax.fill_between(r, y - s_, y + s_, color=C[k], alpha=0.14, lw=0)
    ax.set_xlabel("Sampling ratio $|\\mathcal{S}|/N$"); ax.set_ylabel("NMSE on unobserved nodes (dB)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.3)
    save(fig, "fig4_sampling")


# ------------------------------------------------------------------ Fig 5 detection AUC dot plot
def fig_auc():
    import csv
    rows = list(csv.DictReader(open(os.path.join(RES, "e4_detection_auc.csv"))))
    ms = [("zscore", "per-node $z$-score"), ("pca_spe", "PCA SPE"),
          ("gsp_highpass", "graph high-pass"), ("gsp_nbr_reg", "graph neighbour regression")]
    fig, ax = plt.subplots(figsize=(COL1, 3.6))
    y = np.arange(len(rows))[::-1]
    for k, (m, lab) in enumerate(ms):
        ax.scatter([float(r[m]) for r in rows], y + (k - 1.5) * 0.14, s=14, marker=MK[k], color=C[k],
                   edgecolors="white", linewidths=0.4, label=lab, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r["fault"].replace("_", " ") for r in rows], fontsize=6.5)
    lay = [r["layer"] for r in rows]
    for i, l in enumerate(lay):
        if i > 0 and l != lay[i - 1]:
            ax.axhline(y[i] + 0.5, color=MUTED, lw=0.6)
    ax.axvline(0.5, color=MUTED, lw=0.7, ls=":")
    ax.text(0.5, y[0] + 0.7, "chance", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xlabel("Detection AUC (4 runs per fault, pooled severities)")
    ax.set_xlim(0.33, 1.01)
    ax.set_ylim(y[-1] - 0.7, y[0] + 1.2)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.1), ncol=2, fontsize=6.5)
    save(fig, "fig5_fault_detection_auc")


# ------------------------------------------------------------------ Fig 6 fault example
def fig_example(name="sens_bias_yaw"):
    d = np.load(os.path.join(RES, f"e4_example_{name}.npz"))
    t, t0 = d["t"], float(d["t0"])
    fig, axs = plt.subplots(2, 1, figsize=(COL1, 3.0), sharex=True, gridspec_kw=dict(hspace=0.12))
    for k, lab in enumerate(d["labels"]):
        axs[0].plot(t, d["Z"][:, k], color=C[k], ls=LS[k], lw=0.8, label=str(lab))
    axs[0].set_ylabel("$z$-scored measurement")
    axs[0].legend(loc="upper left", ncol=3, fontsize=6.5)
    for k, (m, lab) in enumerate([("zscore", "per-node $z$-score"), ("pca_spe", "PCA SPE"),
                                  ("gsp_nbr_reg", "graph neighbour regression")]):
        sc = d["score_" + m] / d["thr_" + m]
        axs[1].plot(t, sc, color=C[k], ls=LS[k], lw=1.0, label=lab)
    axs[1].axhline(1, color=MUTED, lw=0.7, ls=":")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("score / threshold (1% FAR)"); axs[1].set_xlabel("Time (s)")
    axs[1].legend(loc="upper left", fontsize=6.3)
    for a in axs:
        a.axvline(t0, color=INK2, lw=0.7, ls="--")
    axs[1].text(t0 + 10, 0.03, "fault onset", fontsize=6, color=INK2)
    axs[1].set_ylim(0.02, None)
    save(fig, f"fig6_fault_example_{name}")


# ------------------------------------------------------------------ Fig 7 topology PR
def fig_topology():
    e5 = json.load(open(os.path.join(RES, "e5_topology.json")))
    d = np.load(os.path.join(RES, "e5_curves.npz"))
    fig, axs = plt.subplots(1, 2, figsize=(COL2 * 0.62, 2.2), sharey=True, gridspec_kw=dict(wspace=0.16))
    ms = [("correlation", "|correlation|"), ("partial_corr_LW", "|partial corr.| (Ledoit-Wolf)"),
          ("graphical_lasso", "graphical lasso"), ("smoothness_Kalofolias", "smoothness-based (Kalofolias)")]
    for ax, rep in zip(axs, ["levels", "differences"]):
        for k, (m, lab) in enumerate(ms):
            key = f"{rep}__{m}"
            ax.plot(d[key + "__rc"], d[key + "__pr"], color=C[k], ls=LS[k], lw=1.1,
                    label=lab)
        ax.axhline(e5["chance_AUPRC"], color=MUTED, lw=0.6, ls=":")
        ax.set_title("signal levels" if rep == "levels" else "first differences", color=INK)
        ax.set_xlabel("Recall"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        txt = "AUPRC\n" + "\n".join(f"{lab.split(' (')[0]}: {e5[rep + '/' + m]['AUPRC']:.2f}" for m, lab in ms) \
            + f"\nchance: {e5['chance_AUPRC']:.2f}"
        ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=INK,
                bbox=dict(fc="white", ec=GRID, lw=0.5, pad=2))
    axs[0].set_ylabel("Precision")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=6.5)
    save(fig, "fig7_topology_pr")


if __name__ == "__main__":
    fig_graph(); fig_spectrum(); fig_denoise(); fig_sampling(); fig_auc(); fig_topology()
    for n in ("sens_bias_yaw", "brake_drag_F", "misfire", "edu_pump"):
        fig_example(n)
    print("figures written")
