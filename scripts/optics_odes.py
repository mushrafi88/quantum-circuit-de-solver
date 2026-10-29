"""Seven quantum-optics ODEs x four qubit circuits x three seeds.

    python scripts/optics_odes.py          # train, then make figures + table
    python scripts/optics_odes.py plot     # figures + table from saved results

-> results/optics_odes.{csv,npz}
   draft_paper/figures/fig_optics_odes_qubit.pdf, fig_loss_curves_qubit.pdf, fig_pointwise_error_qubit.pdf
   draft_paper/tables/accuracy_qubit.tex
"""

import sys

import numpy as np
import pandas as pd

from qoqpinn.circuits import ARCHITECTURES, init_weights
from qoqpinn.plotstyle import ARCH_COLOR, ARCH_LABEL, ARCH_STYLE, INK, RESULTS, ROOT, plt, save
from qoqpinn.problems import PROBLEMS

SEEDS = (0, 1, 2)
N_QUBITS, N_LAYERS, ITERS = 4, 3, 2000
TABLES = ROOT / "draft_paper" / "tables"
TABLES.mkdir(exist_ok=True)


def run():
    from qoqpinn.solver import solve
    rows, store = [], {}
    for arch in ARCHITECTURES:                 # arch outermost: one compile per circuit
        for name in PROBLEMS:
            for seed in SEEDS:
                r = solve(name, arch=arch, n_qubits=N_QUBITS, n_layers=N_LAYERS,
                          iterations=ITERS, seed=seed)
                rows.append(dict(problem=name, arch=arch, seed=seed,
                                 rel_l2=r["rel_l2"], final_loss=r["loss"][-1]))
                key = f"{name}__{arch}__{seed}"
                store[key + "__u"], store[key + "__w"], store[key + "__loss"] = r["u"], r["weights"], r["loss"]
                store[name + "__t"] = r["t"]
                print(f"{name:22s} {arch:8s} s{seed}  rel-L2 {r['rel_l2']:.2e}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "optics_odes.csv", index=False)
    np.savez(RESULTS / "optics_odes.npz", **store)


def best_seed(df, name, arch):
    sub = df[(df.problem == name) & (df.arch == arch)]
    return int(sub.loc[sub.final_loss.idxmin(), "seed"])


def _tex_sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def accuracy_table(df):
    med = df.pivot_table(index="problem", columns="arch", values="rel_l2",
                         aggfunc="median").reindex(index=list(PROBLEMS), columns=list(ARCHITECTURES))
    lines = []
    for name in PROBLEMS:
        vals = med.loc[name]
        cells = [f"{{\\boldmath {_tex_sci(v)}}}" if a == vals.idxmin() else _tex_sci(v)
                 for a, v in vals.items()]
        lines.append(f"{PROBLEMS[name]['label']} & " + " & ".join(cells) + r" \\")
    params = [str(init_weights(a, N_QUBITS).size) for a in ARCHITECTURES]
    (TABLES / "accuracy_qubit.tex").write_text(
        f"\\begin{{tabular}}{{l{'r' * len(ARCHITECTURES)}}}\n\\toprule\n"
        f"system & {' & '.join(ARCH_LABEL[a] for a in ARCHITECTURES)} \\\\\n\\midrule\n"
        + "\n".join(lines) + f"\n\\midrule\ntrainable parameters & {' & '.join(params)} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n")
    print("  table  -> draft_paper/tables/accuracy_qubit.tex")


def figures(df, store):
    names = list(PROBLEMS)
    # fits
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.8))
    for ax, name in zip(axes.flat, names):
        t = store[name + "__t"]
        ax.plot(t, PROBLEMS[name]["exact"](t), color=INK, lw=2.2, alpha=0.35, label="exact")
        for arch in ARCHITECTURES:
            ax.plot(t, store[f"{name}__{arch}__{best_seed(df, name, arch)}__u"],
                    color=ARCH_COLOR[arch], ls=ARCH_STYLE[arch], label=ARCH_LABEL[arch])
        ax.set_title(PROBLEMS[name]["label"], fontsize=9)
        ax.set_xlabel("t")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles, labels, loc="center", fontsize=9, title="lowest-loss seed")
    fig.tight_layout()
    save(fig, "fig_optics_odes_qubit")

    # training loss and pointwise error of the same seeds
    for kind in ("loss", "error"):
        fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.6))
        for ax, name in zip(axes.flat, names):
            t = store[name + "__t"]
            for arch in ARCHITECTURES:
                key = f"{name}__{arch}__{best_seed(df, name, arch)}"
                if kind == "loss":
                    y = store[key + "__loss"]
                    ax.plot(np.arange(1, len(y) + 1), y, color=ARCH_COLOR[arch],
                            ls=ARCH_STYLE[arch], lw=1.1, label=ARCH_LABEL[arch])
                else:
                    err = np.abs(store[key + "__u"] - PROBLEMS[name]["exact"](t))
                    ax.plot(t, np.maximum(err, 1e-12), color=ARCH_COLOR[arch],
                            ls=ARCH_STYLE[arch], lw=1.1, label=ARCH_LABEL[arch])
            ax.set_yscale("log")
            ax.set_title(PROBLEMS[name]["label"], fontsize=9)
            ax.set_xlabel("Adam step" if kind == "loss" else "t")
        axes[0, 0].set_ylabel("training loss" if kind == "loss" else r"$|u-u_{\rm exact}|$")
        axes[1, 0].set_ylabel(axes[0, 0].get_ylabel())
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes.flat[-1].axis("off")
        axes.flat[-1].legend(handles, labels, loc="center", fontsize=8, title="lowest-loss seed")
        fig.tight_layout()
        save(fig, "fig_loss_curves_qubit" if kind == "loss" else "fig_pointwise_error_qubit")


if __name__ == "__main__":
    if sys.argv[1:] != ["plot"]:
        run()
    df = pd.read_csv(RESULTS / "optics_odes.csv")
    accuracy_table(df)
    figures(df, np.load(RESULTS / "optics_odes.npz"))
