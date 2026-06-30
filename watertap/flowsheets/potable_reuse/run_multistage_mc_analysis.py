"""Driver: multi-stage (n=2 and n=3) RBAT Monte Carlo + envelopes + optimal-vs-conservative.

For each stage count it produces, per state (CA/CO/FL):
  1. a 1000-sample Monte Carlo scatter (feed_flow 1-100 MGD, TOC 7-15 mg/L, TDS 500-1000
     mg/L, brine 0.05-0.66 $/m3) -- CSV + flow-vs-LCOW PNG;
  2. worst/best optimal-LCOW envelope curves overlaid on that scatter (same plot);
  3. optimal-vs-conservative LCOW and CAPEX-ratio figures vs system capacity.

Outputs are stage-tagged so single-stage results are never overwritten:
  monte_carlo/mc_v2_<STATE>_RBAT_ROspiral_add_const_n<N>_n<NSAMP>.csv/.png
  optimal_vs_conserv_n<N>/optimal_vs_conserv_<STATE>.csv/.png

Usage:
  python -m watertap.flowsheets.potable_reuse.run_multistage_mc_analysis smoke   # fast check
  python -m watertap.flowsheets.potable_reuse.run_multistage_mc_analysis full    # real run
"""
import os
import sys
import logging

# Quiet the per-block IDAES initialization INFO logs: this driver runs thousands of solves,
# and the init chatter would bury the progress prints (and bloat the log file).
for _name in ("idaes", "idaes.init", "pyomo"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import watertap.flowsheets.potable_reuse.dpr_spiral_analysis_multistage as ana
import watertap.flowsheets.potable_reuse.DPR_sweep_v2_spiral_multistage as swp

_HERE = os.path.dirname(os.path.abspath(__file__))


def _configure_stage(n):
    """Point both modules (and their import-time-derived constants) at stage count n."""
    swp.N_STAGES = n
    ana.N_STAGES = n
    swp.TAG = f"ROspiral_add_const_n{n}"
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
             f"mc_v2_{s}_RBAT_ROspiral_add_const_n{n}_n{num_samples}.csv",
             f"{s} / RBAT ({n}-stage)", ana.RBAT_PT, ana.RBAT_CURVE)
            for s in states
        ]
        ana.run_mc_envelopes()

    print("\n>>> ALL DONE", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "smoke":
        # Fast end-to-end validation: 1 stage, 1 state, tiny sample/flow grids.
        ana.FLOWS_MGD = [10, 50]
        ana.RBAT_CASES = [("Optimized", None), ("Conserv rec=0.40", 0.40)]
        ana.ENV_FLOWS = [5, 50]
        run(stages=[2], states=("CO",), num_samples=12)
    else:
        run(stages=[2, 3], states=("CA", "CO", "FL"), num_samples=1000)
