"""Benchmarks reproduced from the literature: three ODEs and the heat equation.

    python scripts/literature.py          # train, then make figures + table
    python scripts/literature.py plot     # figures + table from saved results

ODEs (Kyriienko et al., Hunout et al., Panichi et al.): four qubit circuits x
three seeds, input rescaled to [-pi/2, pi/2], condition weight 10.
Heat equation (Panichi et al.): two-input MS circuit, three seeds.

-> results/literature_{odes,heat}.{csv,npz}
   draft_paper/figures/fig_literature_odes_qubit.pdf, fig_literature_ode_errors_qubit.pdf, fig_draft_heat.pdf
"""

import sys

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from qoqpinn.circuits import ARCHITECTURES
from qoqpinn.plotstyle import ARCH_COLOR, ARCH_LABEL, ARCH_STYLE, INK, RESULTS, plt, save
from qoqpinn.problems import LITERATURE

SEEDS = (0, 1, 2)
W_COND = 10.0   # condition-loss weight, as the boundary scaling of Berger et al.


def run():
    from qoqpinn.pde import solve_pde
    from qoqpinn.solver import solve
    rows, store = [], {}
    for arch in ARCHITECTURES:                 # arch outermost: one compile per circuit
        for name, p in LITERATURE.items():
            for seed in SEEDS:
                r = solve(p, arch=arch, seed=seed, rescale=True, condition_weight=W_COND)
                rows.append(dict(problem=name, arch=arch, seed=seed, rel_l2=r["rel_l2"],
                                 final_loss=r["loss"][-1]))
                store[f"{name}__{arch}__{seed}__u"] = r["u"]
                store[f"{name}__t"] = r["t"]
                print(f"{name:16s} {arch:8s} s{seed}  rel-L2 {r['rel_l2']:.2e}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "literature_odes.csv", index=False)
    np.savez(RESULTS / "literature_odes.npz", **store)

    rows, store = [], {}
    for seed in SEEDS:
        r = solve_pde("heat", seed=seed, condition_weight=W_COND)
        rows.append(dict(problem="heat", seed=seed, rel_l2=r["rel_l2"], final_loss=r["loss"][-1]))
        for k in ("u", "ref", "x", "t", "loss"):
            store[f"heat__{seed}__{k}"] = r[k]
        print(f"heat     s{seed}  rel-L2 {r['rel_l2']:.2e}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "literature_heat.csv", index=False)
    np.savez(RESULTS / "literature_heat.npz", **store)


def best(df, name, arch):
    sub = df[(df.problem == name) & (df.arch == arch)]
    return int(sub.loc[sub.final_loss.idxmin(), "seed"])


def ode_figures(od, st):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8))
    for ax, (name, p) in zip(axes, LITERATURE.items()):
        t = st[f"{name}__t"]
        ax.plot(t, p["exact"](t), color=INK, lw=2.2, alpha=0.35, label="exact")
        for arch in ARCHITECTURES:
            ax.plot(t, st[f"{name}__{arch}__{best(od, name, arch)}__u"], color=ARCH_COLOR[arch],
                    ls=ARCH_STYLE[arch], label=ARCH_LABEL[arch])
        ax.set_title(p["label"], fontsize=9)
        ax.set_xlabel("x" if name != "hunout_spring" else "t")
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_literature_odes_qubit")

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2))
    for j, (name, prob) in enumerate(LITERATURE.items()):
        t = st[f"{name}__t"]
        for a, arch in enumerate(ARCHITECTURES):
            e = np.abs(st[f"{name}__{arch}__{best(od, name, arch)}__u"] - prob["exact"](t))
            axes[0, j].semilogy(t, np.maximum(e, 1e-12), color=ARCH_COLOR[arch], ls=ARCH_STYLE[arch],
                                label=ARCH_LABEL[arch])
            v = od[(od.problem == name) & (od.arch == arch)].rel_l2
            axes[1, j].scatter(np.full(len(v), a), v, color=ARCH_COLOR[arch], s=14)
            axes[1, j].hlines(v.median(), a - 0.3, a + 0.3, color=ARCH_COLOR[arch])
        axes[0, j].set_title(prob["label"], fontsize=8)
        axes[0, j].set_xlabel("x" if name != "hunout_spring" else "t")
        axes[1, j].set_yscale("log")
        axes[1, j].set_xticks(range(len(ARCHITECTURES)), [ARCH_LABEL[a] for a in ARCHITECTURES],
                              rotation=30, fontsize=7)
    axes[0, 0].set_ylabel("$|u_\\theta-u|$")
    axes[1, 0].set_ylabel("relative $L^2$ error")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    save(fig, "fig_literature_ode_errors_qubit")


def heat_figure(hd, sp):
    s = int(hd.sort_values("final_loss").seed.iloc[0])
    x, t, u, ref = (sp[f"heat__{s}__{k}"] for k in ("x", "t", "u", "ref"))
    fig = plt.figure(figsize=(9.6, 5.6))
    gs = GridSpec(2, 3, figure=fig)
    ext = (x.min(), x.max(), t.min(), t.max())
    vmax = np.abs(ref).max()
    for j, (data, title) in enumerate(((ref, "reference"), (u, "quantum circuit"))):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(data, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(f"heat equation: {title}", fontsize=8)
        ax.set_xlabel("x"); ax.set_ylabel("t"); ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(np.abs(u - ref), origin="lower", extent=ext, aspect="auto", cmap="Blues")
    ax.set_title("|error|", fontsize=8); ax.set_xlabel("x"); ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax = fig.add_subplot(gs[1, :2])
    rows = np.linspace(0, t.shape[0] - 1, 4).round().astype(int)
    for c, r in zip(plt.cm.viridis(np.linspace(0, 0.85, 4)), rows):
        ax.plot(x[r], ref[r], color=c, lw=1.8, label=f"t = {t[r, 0]:.2f}")
        ax.plot(x[r], u[r], color=c, lw=1.2, ls="--")
    ax.set_xlabel("x"); ax.set_ylabel("T"); ax.legend(fontsize=7)
    ax.set_title("slices: reference (solid), circuit (dashed)", fontsize=8)
    ax = fig.add_subplot(gs[1, 2])
    for seed, c in zip(SEEDS, ARCH_COLOR.values()):
        ax.semilogy(sp[f"heat__{seed}__loss"], color=c, lw=1, label=f"seed {seed}")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(fontsize=7)
    ax.set_title("training loss, all seeds", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_draft_heat")


if __name__ == "__main__":
    if sys.argv[1:] != ["plot"]:
        run()
    od = pd.read_csv(RESULTS / "literature_odes.csv")
    hd = pd.read_csv(RESULTS / "literature_heat.csv")
    ode_figures(od, np.load(RESULTS / "literature_odes.npz"))
    heat_figure(hd, np.load(RESULTS / "literature_heat.npz"))
