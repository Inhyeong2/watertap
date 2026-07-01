"""Overlay RBAT vs IPR Optimized-LCOW (and CAPEX-ratio) opt-vs-conserv curves on a SINGLE shared
axis, so the two potable-reuse schemes are directly comparable across system capacity and RO stage
count.

Reads the per-stage opt-vs-conserv CSVs written by dpr_analysis_v3 (run_rbat_opt_vs_conserv /
run_ipr_opt_vs_conserv):
  output/optimal_vs_conserv_n<N>/optimal_vs_conserv_CA.csv       (RBAT, TERTIARY)
  output/optimal_vs_conserv_n<N>/optimal_vs_conserv_IPR_CA.csv   (IPR,  SECONDARY)

CAVEAT (annotated on the figure): RBAT is costed on TERTIARY effluent (feed TDS 0.5 kg/m3) and IPR
on SECONDARY effluent (feed TDS 1.0 kg/m3), i.e. different source waters -- this is a reuse-SCHEME
comparison, not an identical-feed comparison.

Output:
  output/rbat_vs_ipr_CA_LCOW.png
  output/rbat_vs_ipr_CA_capexratio.png

Run:
  python -m watertap.flowsheets.potable_reuse.plot_rbat_vs_ipr
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "output")

# Stages where BOTH trains have opt-vs-conserv data get overlaid.
STAGES = [1, 2, 3]
STATE = "CA"

RBAT_COLOR, IPR_COLOR = "darkgreen", "steelblue"
STAGE_LS = {1: ":", 2: "-", 3: "--"}   # dotted = 1-stage, solid = 2-stage, dashed = 3-stage
# Single-stage low-capacity LCOW spikes to ~10 $/m3 (fixed unit costs dominate at 1 MGD), an ~10x
# span vs the 3-stage high-capacity end (~0.9). A log y-axis keeps every curve legible.
LOG_Y = True


def _load(train, n):
    """Return the Optimized-case dataframe (sorted by flow) for one train+stage, or None."""
    fname = "optimal_vs_conserv_CA.csv" if train == "RBAT" else "optimal_vs_conserv_IPR_CA.csv"
    path = os.path.join(OUT, f"optimal_vs_conserv_n{n}", fname)
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d = d[d["case"] == "Optimized"].dropna(subset=["flow_MGD"]).sort_values("flow_MGD")
    return d


def _plot(ycol, ylab, out_png):
    fig, ax = plt.subplots(figsize=(9, 6))
    for n in STAGES:
        for train, color in (("RBAT", RBAT_COLOR), ("IPR", IPR_COLOR)):
            d = _load(train, n)
            if d is None or d.dropna(subset=[ycol]).empty:
                continue
            dd = d.dropna(subset=[ycol])
            sysrec = d["recovery"].mean() * 100
            ax.plot(dd["flow_MGD"], dd[ycol], marker="o", color=color, ls=STAGE_LS[n],
                    label=f"{train} {n}-stage (Optimized, sys recovery ~{sysrec:.0f}%)")
    if LOG_Y and ycol == "LCOW":
        ax.set_yscale("log")
    ax.set_xlabel("System capacity (MGD)")
    ax.set_ylabel(ylab)
    ax.set_title(f"RBAT vs IPR ({STATE}): Optimized {ycol} vs system capacity\n"
                 "RBAT = TERTIARY feed (TDS 0.5) ; IPR = SECONDARY feed (TDS 1.0) -- scheme comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("saved:", out_png)


def main():
    _plot("LCOW", "LCOW ($/m$^3$)", os.path.join(OUT, f"rbat_vs_ipr_{STATE}_LCOW.png"))
    _plot("capex_ratio", "CAPEX ratio [ann.CAPEX/(ann.CAPEX+OPEX)]",
          os.path.join(OUT, f"rbat_vs_ipr_{STATE}_capexratio.png"))


if __name__ == "__main__":
    main()
