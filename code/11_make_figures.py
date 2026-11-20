"""Draw the four figures in the manuscript.

Figure 1  fig1_mechanism.png        what coarsening does to the number when risk is constant
Figure 2  fig2_calibration.png      rejection rates at equal risk, and power
Figure 3  fig3_fifty_x.png          the same failure in the policy statistic
Figure 4  fig4_reference_set.png    where the London answer turns over

Part of the reproducibility package for "Crime concentration after spatial publication:
aligning random baselines with observation processes". Reads outputs/ and writes figures/.
Style choices are for print: line style and marker carry series identity, so the figures survive
greyscale reproduction, and the type is sized for the journal's text width.

"""
from pathlib import Path
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("SPATIAL_OUTPUT", REPO / "outputs"))
FIG = REPO / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 7, "legend.fontsize": 6.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

OP5 = [("identity", "Identity"), ("match", "Matching"), ("60", "60 m"),
       ("160", "160 m"), ("250", "250 m")]
OP6 = OP5 + [("vertex", "Vertex")]


def read(name):
    with open(OUT / name, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def figure_one():
    """What coarsening does to the number when nothing about risk changes.

    One panel, deliberately. The point is the crossing: at a 250 m grid a city with no place
    effect produces a more concentrated count distribution than real Chicago does at exact
    coordinates.
    """
    rows = [r for r in read("calibration_summary.csv") if float(r["sigma"]) == 0]
    idx = {m: i for i, (m, _) in enumerate(OP5)}
    n = len(OP5)
    gini = [0.0] * n
    for r in rows:
        if r["method"] == "operator":
            gini[idx[r["mode"]]] = float(r["mean_gini"])
    observed = 0.7961058813745547        # Chicago observed Gini, Table 1

    fig, ax = plt.subplots(figsize=(5.5, 3.1), layout="constrained")
    x = list(range(n))
    ax.axhline(observed, color="0.45", lw=0.9, ls="--")
    ax.annotate("observed Chicago 0.796", (0.02, observed), textcoords="offset points",
                xytext=(0, 4), fontsize=6.6, color="0.35", ha="left")
    ax.plot(x, gini, marker="o", ms=4.2, lw=1.2, color="0.12")
    ax.annotate("0.519", (0, gini[0]), textcoords="offset points",
                xytext=(7, 5), fontsize=6.8, color="0.25")
    ax.annotate("0.882", (n - 1, gini[-1]), textcoords="offset points",
                xytext=(-4, 6), ha="right", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in OP5])
    ax.set_ylim(0.45, 0.93)
    ax.set_ylabel("Gini")
    ax.set_xlabel("Observation operator")
    p = FIG / "fig1_mechanism.png"
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", p.name)


def figure_two():
    """Equal-risk size against the operator: the cliff, and the reason for it."""
    rows = [r for r in read("calibration_summary.csv") if float(r["sigma"]) == 0]
    idx = {m: i for i, (m, _) in enumerate(OP5)}
    n = len(OP5)
    length, oper, gini = [0.0] * n, [0.0] * n, [0.0] * n
    lo, hi = [0.0] * n, [0.0] * n
    for r in rows:
        i = idx[r["mode"]]
        if r["method"] == "length":
            length[i] = float(r["rejection"])
        else:
            oper[i] = float(r["rejection"])
            lo[i], hi[i] = float(r["low"]), float(r["high"])
            gini[i] = float(r["mean_gini"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.6), layout="constrained")
    x = list(range(n))

    ax.axhline(0.05, color="0.55", lw=0.8, ls=":", zorder=1)
    ax.text(-0.42, 0.075, "nominal 0.05", fontsize=6.8, color="0.35", ha="left")
    ax.plot(x, length, marker="s", ms=4.2, lw=1.1, color="0.12", label="Direct length null")
    ax.errorbar(x, oper, yerr=[[o - a for o, a in zip(oper, lo)],
                               [b - o for o, b in zip(oper, hi)]],
                marker="o", ms=4.2, lw=1.1, ls="--", color="0.55", capsize=1.8,
                label="Operator-matched null")
    for i, v in enumerate(length):
        if v > 0.5:
            ax.annotate("1.00", (i, v), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in OP5])
    ax.set_xlim(-0.45, n - 0.3)
    ax.set_ylim(-0.04, 1.12)
    ax.set_ylabel("Rejection rate at equal risk")
    ax.set_title("(a) Size of the test", loc="left")
    ax.legend(loc="center right", frameon=False, handlelength=2.2)

    ax2.plot(x, gini, marker="^", ms=4.2, lw=1.1, color="0.25")
    ax2.set_xticks(x)
    ax2.set_xticklabels([lab for _, lab in OP5])
    ax2.set_xlim(-0.45, n - 0.3)
    ax2.set_ylim(0.45, 0.95)
    ax2.set_ylabel("Mean Gini of equal-risk data")
    ax2.set_title("(b) Why it happens", loc="left")
    ax2.annotate("%.3f" % gini[0], (0, gini[0]), textcoords="offset points",
                 xytext=(0, 7), ha="center", fontsize=6.8)
    ax2.annotate("%.3f" % gini[-1], (n - 1, gini[-1]), textcoords="offset points",
                 xytext=(-6, 4), ha="right", fontsize=6.8)

    p = FIG / "fig2_size_by_operator.png"
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", p.name)


def figure_three():
    """50-X under equal risk, and the size each baseline delivers."""
    rows = read("fifty_x_calibration.csv")
    idx = {m: i for i, (m, _) in enumerate(OP6)}
    n = len(OP6)
    net = {}
    for network in ("Chicago", "London"):
        eq, matched, direct = [0.0] * n, [0.0] * n, [0.0] * n
        for r in rows:
            if r["network"] != network:
                continue
            i = idx[r["mode"]]
            if r["baseline"] == "length":
                direct[i] = float(r["eq_size"])
            else:
                eq[i] = float(r["null_mean"])
                matched[i] = float(r["eq_size"])
        net[network] = (eq, matched, direct)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.7), layout="constrained")
    x = list(range(n))
    styles = {"Chicago": dict(marker="o", ls="-", color="0.12"),
              "London": dict(marker="s", ls="--", color="0.55")}

    for network in ("Chicago", "London"):
        ax.plot(x, net[network][0], ms=4.2, lw=1.1, label=network, **styles[network])
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in OP6], rotation=20, ha="right")
    ax.set_ylim(2.5, 21.5)
    ax.set_ylabel("Equal-risk 50-X (%)")
    ax.set_title("(a) Concentration under equal risk", loc="left")
    ax.legend(loc="lower left", frameon=False, handlelength=2.2)

    ax2.axhline(0.05, color="0.55", lw=0.8, ls=":", zorder=1)
    for network in ("Chicago", "London"):
        ax2.plot(x, net[network][1], ms=4.2, lw=1.1,
                 label="%s, matched" % network, **styles[network])
    for network, mk in (("Chicago", "^"), ("London", "v")):
        ax2.plot(x, net[network][2], ms=4.0, lw=1.0, ls=":", marker=mk,
                 color=styles[network]["color"], label="%s, direct" % network)
    ax2.set_xticks(x)
    ax2.set_xticklabels([lab for _, lab in OP6], rotation=20, ha="right")
    ax2.set_ylim(-0.05, 1.18)
    ax2.set_ylabel("Rejection rate at equal risk")
    ax2.set_title("(b) Size of the 50-X test", loc="left")
    ax2.legend(loc="center right", frameon=False, ncol=1,
               handlelength=2.0, fontsize=6.6)

    p = FIG / "fig3_fifty_x.png"
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", p.name)


def figure_four():
    """Where the London answer turns over, and why it turns over there."""
    rows = read("reference_set_sweep.csv")
    size = [int(r["reference_size"]) for r in rows]
    pval = [float(r["p"]) for r in rows]
    null = [float(r["null_mean"]) for r in rows]
    observed = 0.89363                      # London observed Gini, Table 9

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.7), layout="constrained")
    xs = [s / 1000 for s in size]

    ax.step(xs, pval, where="post", lw=1.1, color="0.12", marker="o", ms=3.8)
    ax.set_xscale("log")
    ax.set_xlim(44, 260)
    ax.set_ylim(-0.05, 1.12)
    ax.axhline(0.05, color="0.55", lw=0.8, ls=":")
    ax.text(46, 0.09, "nominal 0.05", fontsize=6.6, color="0.35", ha="left")
    ax.axvline(52.533, color="0.7", lw=0.8, ls="--")
    ax.annotate("turnover\n52,533", (52.533, 0.62), fontsize=6.6, color="0.35",
                ha="left", va="center", xytext=(4, 0), textcoords="offset points")
    ax.set_xticks([50, 75, 100, 150, 200])
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: "%d" % v))
    ax.xaxis.set_minor_locator(LogLocator(subs="all", numticks=30))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Reference-set size (thousands)")
    ax.set_ylabel("p-value")
    ax.annotate("p = 0.001", (112, 0.001), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=6.6, color="0.35")
    ax.set_title("(a) The answer turns over once", loc="left")

    ax2.plot(xs, null, marker="^", ms=3.8, lw=1.1, color="0.3",
             label="Null mean Gini")
    ax2.axhline(observed, color="0.12", lw=1.0, ls="--", label="Observed 0.894")
    ax2.set_xscale("log")
    ax2.set_xlim(44, 260)
    ax2.set_ylim(0.58, 0.94)
    ax2.set_xticks([50, 75, 100, 150, 200])
    ax2.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: "%d" % v))
    ax2.xaxis.set_minor_locator(LogLocator(subs="all", numticks=30))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel("Reference-set size (thousands)")
    ax2.set_ylabel("Gini")
    ax2.set_title("(b) Why it turns over there", loc="left")
    ax2.legend(loc="lower left", frameon=False, handlelength=2.2)

    p = FIG / "fig4_reference_set.png"
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", p.name)


if __name__ == "__main__":
    figure_one()
    figure_two()
    figure_three()
    figure_four()
    for f in sorted(FIG.glob("*.png")):
        print("  %-34s %6.1f KB" % (f.name, f.stat().st_size / 1024))
