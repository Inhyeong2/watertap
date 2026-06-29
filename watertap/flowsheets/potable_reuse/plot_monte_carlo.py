"""
Visualize a DPR Monte Carlo result CSV (from DPR_sweep_v2.run_monte_carlo).

Default plot: system capacity (feed flow, MGD) on x, LCOW ($/m3) on y, as a scatter
of the random samples. Points are colored by brine disposal cost so the third swept
dimension is visible too. Non-converged samples (LCOW = NaN) are dropped and counted.

Usage:
    python plot_monte_carlo.py                      # default CSV in output/monte_carlo
    python plot_monte_carlo.py path/to/results.csv  # explicit CSV
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, don't open a window
import matplotlib.pyplot as plt

MGD = 0.0438126  # m3/s per MGD

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(
    _MODULE_DIR, "output", "monte_carlo", "mc_v2_CA_RBAT_n1000.csv"
)


def plot_flow_vs_lcow(
    csv_path=None, output_png=None, show=False, label=None, ylim=None, xlim=None,
    color_by="brine_disposal_cost", point_color="tab:blue",
    curves=None,
):
    """Scatter system capacity (MGD) vs LCOW from a Monte Carlo CSV. ``label`` (e.g.
    "CA / RBAT") is added to the title to distinguish per-state plots. ``ylim`` (lo, hi)
    fixes the LCOW axis so several plots share the same scale for direct comparison.

    ``color_by`` names a CSV column used to color the points (with a colorbar). It
    defaults to ``brine_disposal_cost`` (the RBAT third swept dimension). If that column
    is absent -- e.g. a CBAT run, which has no brine handle -- the points are drawn in a
    single color instead. Pass ``color_by=None`` to force a single-color scatter.

    Returns (output_png, n_total, n_failed)."""
    if csv_path is None:
        csv_path = DEFAULT_CSV
    df = pd.read_csv(csv_path)

    # parameter_sweep prefixes the first header column with "# "; normalize names.
    df.columns = [c.lstrip("# ").strip() for c in df.columns]

    n_total = len(df)
    converged = df.dropna(subset=["LCOW"]).copy()
    n_failed = n_total - len(converged)

    converged["flow_MGD"] = converged["feed_flow"] / MGD

    fig, ax = plt.subplots(figsize=(8, 6))
    if color_by is not None and color_by in converged.columns:
        sc = ax.scatter(
            converged["flow_MGD"],
            converged["LCOW"],
            c=converged[color_by],
            cmap="viridis",
            s=18,
            alpha=0.75,
            edgecolors="none",
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(f"{color_by} ($/m$^3$)")
    else:
        # No color dimension (e.g. CBAT has no brine_disposal_cost) -> single color.
        ax.scatter(
            converged["flow_MGD"],
            converged["LCOW"],
            color=point_color,
            s=18,
            alpha=0.55,
            edgecolors="none",
            label="Monte Carlo samples",
        )

    # Optional overlay curves. ``curves`` is a list of dicts:
    #   {"x":..., "y":..., "color":..., "label":..., "ls":"-"}
    # e.g. worst-case (cost-max inputs) and best-case (cost-min inputs) optimal LCOW.
    if curves:
        for c in curves:
            ax.plot(c["x"], c["y"], color=c.get("color", "black"), lw=2.2,
                    ls=c.get("ls", "-"), marker="o", ms=3, label=c.get("label"), zorder=5)
        ax.legend()

    ax.set_xlabel("System capacity (MGD)")
    ax.set_ylabel("LCOW ($/m$^3$)")
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    title = "DPR Monte Carlo: LCOW vs system capacity"
    if label:
        title += f" [{label}]"
    ax.set_title(
        f"{title}\n"
        f"{len(converged)} converged / {n_total} samples"
        + (f" ({n_failed} not converged, dropped)" if n_failed else "")
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_png is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        output_png = os.path.join(os.path.dirname(csv_path), base + "_flow_vs_LCOW.png")
    fig.savefig(output_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return output_png, n_total, n_failed


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    png, n_total, n_failed = plot_flow_vs_lcow(csv)
    print(f"samples: {n_total}, not converged: {n_failed}")
    print(f"saved: {png}")
