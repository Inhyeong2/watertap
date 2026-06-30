"""Driver: multi-stage (n=2 and n=3) RBAT Monte Carlo + envelopes + optimal-vs-conservative.

For each stage count it produces, per state (CA/CO/FL):
  1. a 1000-sample Monte Carlo scatter (feed_flow 1-100 MGD, TOC 7-15 mg/L, TDS 500-1000
     mg/L, brine 0.05-0.66 $/m3) -- CSV + flow-vs-LCOW PNG;
  2. worst/best optimal-LCOW envelope curves overlaid on that scatter (same plot);
  3. optimal-vs-conservative LCOW and CAPEX-ratio figures vs system capacity.

Outputs are stage-tagged so different stage counts are never overwritten:
  monte_carlo/mc_v2_<STATE>_RBAT_v3_n<N>_n<NSAMP>.csv/.png
  optimal_vs_conserv_n<N>/optimal_vs_conserv_<STATE>.csv/.png

It can also run the IPR train (UF->RO->UV/AOP, CA / SECONDARY): a Monte Carlo scatter per stage
count with outlier repair (no OVC/envelope -- those are RBAT/CBAT-specific):
  monte_carlo/mc_v2_CA_IPR_v3_n<N>_n<NSAMP>.csv/.png

Usage:
  python -m watertap.flowsheets.potable_reuse.run_mc_analysis_v3 smoke      # fast RBAT check
  python -m watertap.flowsheets.potable_reuse.run_mc_analysis_v3 full       # RBAT real run
  python -m watertap.flowsheets.potable_reuse.run_mc_analysis_v3 ipr-smoke  # fast IPR check
  python -m watertap.flowsheets.potable_reuse.run_mc_analysis_v3 ipr        # IPR real run (CA)
"""
import os
import sys
import logging

# Quiet the per-block IDAES initialization INFO logs: this driver runs thousands of solves,
# and the init chatter would bury the progress prints (and bloat the log file).
for _name in ("idaes", "idaes.init", "pyomo"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from pyomo.environ import value
import watertap.flowsheets.potable_reuse.dpr_analysis_v3 as ana
import watertap.flowsheets.potable_reuse.DPR_sweep_v3 as swp

_HERE = os.path.dirname(os.path.abspath(__file__))

# IPR (indirect potable reuse) is only regulation-supported for CA and treats SECONDARY effluent.
IPR_STATE = "CA"
IPR_ENV_FLOWS = [1, 5, 10, 20, 35, 50, 65, 80, 100]


def _configure_stage(n):
    """Point both modules (and their import-time-derived constants) at stage count n."""
    swp.N_STAGES = n
    ana.N_STAGES = n
    swp.TAG = f"v3_n{n}"
    ana.OVC_DIR = os.path.join(_HERE, "output", f"optimal_vs_conserv_n{n}")


def run(stages, states, num_samples):
    for n in stages:
        _configure_stage(n)
        print(f"\n################  N_STAGES = {n}  ################", flush=True)

        # 1. Monte Carlo scatter (CSV + PNG) per state, same random samples across states.
        print(f"[{n}-stage] Monte Carlo ({num_samples} samples) for {states} ...", flush=True)
        swp.run_monte_carlo_multistate(
            states=states, treatment_train="RBAT",
            num_samples=num_samples, seed=0, verbose=False,
        )

        # 2. Optimal vs Conservative (LCOW + CAPEX ratio figures).
        print(f"[{n}-stage] Optimal vs Conservative ...", flush=True)
        ana.run_rbat_opt_vs_conserv(states=states)

        # 3. Worst/best envelope curves overlaid on the MC scatter (RBAT only).
        print(f"[{n}-stage] Worst/best envelope overlay ...", flush=True)
        ana.ENV_JOBS = [
            (s, "RBAT",
             f"mc_v2_{s}_RBAT_v3_n{n}_n{num_samples}.csv",
             f"{s} / RBAT ({n}-stage)", ana.RBAT_PT, ana.RBAT_CURVE)
            for s in states
        ]
        ana.run_mc_envelopes()

    print("\n>>> ALL DONE", flush=True)


def _ipr_worst_lcow_at(flow):
    """Worst-case (TOC 15 mg/L, SECONDARY TDS, brine 0.66) optimized IPR LCOW at one flow, via a
    fresh build with a small flow-perturbation multistart. Returns None if no start converges."""
    for eps in (0.0, 1e-3, -1e-3, 5e-3, -5e-3, 1e-2, -1e-2):
        try:
            ok, m = ana._solve_train_at(IPR_STATE, flow * (1 + eps), 0.015, None, 0.66,
                                        train="IPR", effluent_type="SECONDARY")
            if ok:
                return value(m.fs.LCOW)
        except Exception:
            pass
    return None


def run_ipr(stages, num_samples):
    """IPR (UF->RO->UV/AOP, CA / SECONDARY) Monte Carlo per stage count, with outlier repair.

    For each stage count: a Monte Carlo scatter (CSV + flow-vs-LCOW PNG, using IPR_MC_CONFIG --
    TOC 10-15 mg/L, TDS pinned at the SECONDARY 1000 mg/L, feed_flow 1-100 MGD, brine 0.05-0.66),
    then a repair pass that re-solves any sample sitting above the IPR worst-case envelope (a
    warm-start local-optimum artifact) and re-plots. OVC / best-worst envelope overlays are RBAT
    /CBAT-specific and are not produced for IPR here.

    Output (stage-tagged): monte_carlo/mc_v2_CA_IPR_v3_n<N>_n<NSAMP>.csv / .png
    """
    from watertap.flowsheets.potable_reuse.plot_monte_carlo import plot_flow_vs_lcow
    for n in stages:
        _configure_stage(n)
        print(f"\n################  IPR  N_STAGES = {n}  ################", flush=True)

        print(f"[IPR {n}-stage] Monte Carlo ({num_samples} samples), CA / SECONDARY ...", flush=True)
        res = swp.run_monte_carlo_multistate(
            states=(IPR_STATE,), treatment_train="IPR", effluent_type="SECONDARY",
            num_samples=num_samples, seed=0, plot=False, verbose=False,  # final plot done below
        )
        csv = res[IPR_STATE][0]

        print(f"[IPR {n}-stage] Outlier repair (vs worst-case envelope) ...", flush=True)
        wf, wl = [], []
        for f in IPR_ENV_FLOWS:
            v = _ipr_worst_lcow_at(f)
            if v is not None:
                wf.append(f)
                wl.append(v)
        nfix = ana.repair_mc_outliers(IPR_STATE, csv, wf, wl, tol=0.01,
                                      train="IPR", effluent_type="SECONDARY", verbose=False)
        print(f"[IPR {n}-stage] repaired {nfix} outlier(s); wrote {os.path.basename(csv)}",
              flush=True)

    # Final figures: single-color (sky-blue) scatter + worst/best envelope, shared y-axis.
    replot_ipr_envelopes(stages, num_samples)
    print("\n>>> IPR DONE", flush=True)


def replot_ipr_envelopes(stages, num_samples):
    """Re-plot the existing IPR Monte Carlo CSVs to match the RBAT envelope figures: a single-color
    (sky-blue) scatter with worst/best optimal-LCOW envelope curves overlaid, and a y-axis shared
    across every stage count so the figures are directly comparable. Reads the CSVs (already
    outlier-repaired); it does NOT re-run the Monte Carlo."""
    import numpy as np
    import pandas as pd
    from watertap.flowsheets.potable_reuse.plot_monte_carlo import plot_flow_vs_lcow

    files = {n: os.path.join(ana.MC_DIR,
                             f"mc_v2_{IPR_STATE}_IPR_v3_n{n}_n{num_samples}.csv") for n in stages}

    # Common LCOW axis across all stage counts.
    ylo = yhi = None
    for f in files.values():
        d = pd.read_csv(f)
        d.columns = [c.lstrip("# ").strip() for c in d.columns]
        d = d.dropna(subset=["LCOW"])
        ylo = d["LCOW"].min() if ylo is None else min(ylo, d["LCOW"].min())
        yhi = d["LCOW"].max() if yhi is None else max(yhi, d["LCOW"].max())
    ylim = (max(0, ylo - 0.03 * (yhi - ylo)), yhi + 0.03 * (yhi - ylo))

    fx = np.array(ana.ENV_FLOWS, dtype=float)
    for n in stages:
        ana.N_STAGES = n  # envelope solves build at this stage count
        print(f"[IPR {n}-stage] worst/best envelope ...", flush=True)
        yw = ana._env_curve(IPR_STATE, "IPR", "worst")
        yb = ana._env_curve(IPR_STATE, "IPR", "best")
        curves = [
            {"x": fx[~np.isnan(yw)], "y": yw[~np.isnan(yw)], "color": ana.IPR_CURVE,
             "label": "worst-case optimal (max TOC, brine)", "ls": "-"},
            {"x": fx[~np.isnan(yb)], "y": yb[~np.isnan(yb)], "color": ana.IPR_CURVE,
             "label": "best-case optimal (min TOC, brine)", "ls": "--"},
        ]
        png, _, _ = plot_flow_vs_lcow(
            files[n], label=f"{IPR_STATE} / IPR ({n}-stage)",
            color_by=None, point_color=ana.IPR_PT, ylim=ylim, curves=curves,
            title_prefix="IPR")
        pd.DataFrame({"flow_MGD": ana.ENV_FLOWS, "worst_LCOW": yw, "best_LCOW": yb}).to_csv(
            files[n].replace(".csv", "_envelope.csv"), index=False)
        print(f"[IPR {n}-stage] -> {os.path.basename(png)}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "smoke":
        # Fast end-to-end validation: 1 stage, 1 state, tiny sample/flow grids.
        ana.FLOWS_MGD = [10, 50]
        ana.RBAT_CASES = [("Optimized", None), ("Conserv rec=0.40", 0.40)]
        ana.ENV_FLOWS = [5, 50]
        run(stages=[2], states=("CO",), num_samples=12)
    elif mode == "ipr-smoke":
        run_ipr(stages=[2], num_samples=12)        # fast IPR end-to-end check
    elif mode == "ipr":
        run_ipr(stages=[2, 3], num_samples=1000)   # IPR full run (CA / SECONDARY)
    elif mode == "ipr-replot":
        # Re-draw IPR figures (sky-blue scatter + envelope, shared axis) from existing CSVs.
        replot_ipr_envelopes(stages=[2, 3], num_samples=1000)
    else:
        run(stages=[2, 3], states=("CA", "CO", "FL"), num_samples=1000)
