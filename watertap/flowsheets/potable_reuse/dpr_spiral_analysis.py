"""Consolidated analysis/figure drivers for the spiral (constrained-RO) DPR flowsheet.

Combines what used to be three separate scripts:
  1. Optimized vs Conservative LCOW vs system capacity -- RBAT  (recovery-stepped)
  2. Optimized vs Conservative LCOW vs system capacity -- CBAT  (GAC conservative)
  3. Monte Carlo worst/best optimal-LCOW envelope curves + uniform restyle of MC figures

All three import the spiral flowsheet model (DPR_flowsheet_v2_spiral) and the shared plot
helper (plot_monte_carlo). The Monte Carlo *sampling* itself lives in DPR_sweep_v2_spiral.py;
this module only post-processes / overlays envelope curves on those results.

Run from repo root:
  python -m watertap.flowsheets.potable_reuse.dpr_spiral_analysis rbat        # opt vs conserv (RBAT)
  python -m watertap.flowsheets.potable_reuse.dpr_spiral_analysis cbat        # opt vs conserv (CBAT)
  python -m watertap.flowsheets.potable_reuse.dpr_spiral_analysis envelopes   # MC worst/best + restyle
  python -m watertap.flowsheets.potable_reuse.dpr_spiral_analysis all         # all of the above
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyomo.environ import value, check_optimal_termination
from pyomo.util.calc_var_value import calculate_variable_from_constraint
import watertap.flowsheets.potable_reuse.DPR_flowsheet_v2_spiral as dpr
from watertap.flowsheets.potable_reuse.plot_monte_carlo import plot_flow_vs_lcow

MGD = 0.0438126
_HERE = os.path.dirname(os.path.abspath(__file__))
OVC_DIR = os.path.join(_HERE, "output", "optimal_vs_conserv")   # opt-vs-conserv outputs
MC_DIR = os.path.join(_HERE, "output", "monte_carlo")           # Monte Carlo outputs
FLOWS_MGD = [1, 5, 10, 20, 35, 50, 65, 80, 100]
PERTURB = (0.0, 1e-4, 5e-4, 1e-3)   # knife-edge rescue: nudge feed flow slightly


# ===========================================================================
# Part 1 -- Optimized vs Conservative, RBAT (recovery-stepped)
# ===========================================================================
# (label, recovery)  recovery=None -> Optimized (free). rec=0.45 ~ optimized recovery,
# isolating the upstream-only penalty.
RBAT_CASES = [("Optimized", None), ("Conserv rec=0.45", 0.45), ("Conserv rec=0.40", 0.40),
              ("Conserv rec=0.35", 0.35), ("Conserv rec=0.30", 0.30)]
RBAT_STATES = ["CA", "CO", "FL"]


def _rbat_point(flow_mgd, recovery, state, tds=0.5):
    LRV_req, det, st, tt, eff, sol = dpr.DPR_initial_setting(
        state=state, treatment_train="RBAT", effluent_type="TERTIARY")
    m = dpr.build_RO(solute_list=sol, state=st, effluent_type=eff, treatment_train=tt,
                     LRVO3_req=LRV_req["ozone_crypto_lrv_required"],
                     LRVClvirus_req=LRV_req["cl2_virus_lrv_required"],
                     LRVClgiardia_req=LRV_req["cl2_giardia_lrv_required"])
    dpr.set_operating_conditions(m, solute_list=sol, treatment_train=tt)
    m.fs.feed.flow_vol[0].fix(flow_mgd * MGD)
    m.fs.feed.conc_mass_comp[0, "tds"].fix(tds)
    dpr.solve(m.fs.feed)
    dpr.scale_system(m, treatment_train=tt)
    dpr.initialize_system(m, treatment_train=tt)
    dpr.solve(m)
    dpr.add_costing(m, treatment_train=tt)
    dpr.initialize_costing(m, treatment_train=tt)
    dpr.optimize_operation(m, state=st, effluent_type=eff, treatment_train=tt)
    RO = m.fs.RO_main.RO
    nonRO = m.fs.non_RO
    res = dpr.solve(m)  # optimized solution (also warm start for the conservative re-solve)
    if recovery is not None:
        # Conservative: over-dose ozone (drop O3toTOC_ratio_constraint, force contact_time 10
        # + O3toTOC 1) + BAF EBCT 30 + UV H2O2 10 + RO recovery fixed.
        nonRO.Ozone.O3toTOC_ratio_constraint.deactivate()
        nonRO.Ozone.contact_time[0].fix(10)
        nonRO.Ozone.O3toTOC[0].fix(1.0)
        nonRO.BAF.EBCT[0].fix(30)
        nonRO.UV_AOP.hydrogen_peroxide_dose[0].fix(10)
        RO.recovery_vol_phase[0, "Liq"].fix(recovery)
        res = dpr.solve(m)
    Ja = value(RO.mixed_permeate[0].flow_vol_phase["Liq"]) / value(RO.area) * 3.6e6
    return check_optimal_termination(res), value(m.fs.LCOW), value(RO.recovery_vol_phase[0, "Liq"]), Ja


def _rbat_rescue(flow_mgd, recovery, state):
    last = None
    for eps in PERTURB:
        try:
            ok, lcow, rec, flux = _rbat_point(flow_mgd * (1 + eps), recovery, state)
            if ok:
                return lcow, rec, flux
            last = (lcow, rec, flux)
        except Exception:
            last = None
    return (float("nan"), float("nan"), float("nan")) if last is None else (float("nan"), last[1], last[2])


def run_rbat_opt_vs_conserv(states=RBAT_STATES):
    os.makedirs(OVC_DIR, exist_ok=True)
    for state in states:
        print(f"\n========== RBAT opt-vs-conserv: {state} ==========")
        rows = []
        csv = os.path.join(OVC_DIR, f"optimal_vs_conserv_{state}.csv")
        for label, rec in RBAT_CASES:
            for f in FLOWS_MGD:
                lcow, r, flux = _rbat_rescue(f, rec, state)
                rows.append({"state": state, "case": label, "recovery_target": rec,
                             "flow_MGD": f, "LCOW": lcow, "recovery": r, "flux_LMH": flux})
                print(f"[{state}][{label:18s}] {f:4d} MGD -> LCOW={lcow:.4f} recov={r:.3f} flux={flux:.1f}")
                pd.DataFrame(rows).to_csv(csv, index=False)
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, _ in RBAT_CASES:
            d = df[df["case"] == label].dropna(subset=["LCOW"]).sort_values("flow_MGD")
            ax.plot(d["flow_MGD"], d["LCOW"], marker="o", label=label)
        ax.set_xlabel("System capacity (MGD)"); ax.set_ylabel("LCOW ($/m$^3$)")
        ax.set_title(f"DPR RBAT ({state}): Optimized vs Conservative (recovery-stepped)\n"
                     "LCOW vs system capacity")
        ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
        png = os.path.join(OVC_DIR, f"optimal_vs_conserv_{state}.png")
        fig.savefig(png, dpi=150); plt.close(fig)
        print("saved:", csv, "|", png)


# ===========================================================================
# Part 2 -- Optimized vs Conservative, CBAT (GAC conservative)
# ===========================================================================
# CA omitted: CBAT cannot meet CA's regulatory requirement (needs RO/RBAT). CBAT = CO/FL.
CBAT_CASES = [("Optimized", False), ("Conservative", True)]
CBAT_STATES = ["CO", "FL"]
GAC_REMOVAL_CONSERV = 0.80   # conservative GAC TOC removal (-> effluent TOC ~1.4 mg/L < 2)
GAC_EBCT_CONSERV = 20        # min (max of the 10-20 range)


def _cbat_point(flow_mgd, conservative, state, tds=0.5):
    LRV_req, det, st, tt, eff, sol = dpr.DPR_initial_setting(
        state=state, treatment_train="CBAT", effluent_type="TERTIARY")
    m = dpr.build_nonRO(solute_list=sol, state=st, effluent_type=eff, treatment_train=tt,
                        LRVO3_req=LRV_req["ozone_crypto_lrv_required"],
                        LRVClvirus_req=LRV_req["cl2_virus_lrv_required"],
                        LRVClgiardia_req=LRV_req["cl2_giardia_lrv_required"])
    dpr.set_operating_conditions(m, solute_list=sol, treatment_train=tt)
    m.fs.feed.flow_vol[0].fix(flow_mgd * MGD)
    m.fs.feed.conc_mass_comp[0, "tds"].fix(tds)
    dpr.solve(m.fs.feed)
    dpr.scale_system(m, treatment_train=tt)
    dpr.initialize_system(m, treatment_train=tt)
    dpr.solve(m)
    dpr.add_costing(m, treatment_train=tt)
    dpr.initialize_costing(m, treatment_train=tt)
    dpr.optimize_operation(m, state=st, effluent_type=eff, treatment_train=tt)
    nz = m.fs.non_RO
    res = dpr.solve(m)
    if conservative:
        # ozone over-dose
        nz.Ozone.O3toTOC_ratio_constraint.deactivate()
        nz.Ozone.contact_time[0].fix(10)
        nz.Ozone.O3toTOC[0].fix(1.0)
        nz.BAF.EBCT[0].fix(30)
        nz.UV_AOP.hydrogen_peroxide_dose[0].fix(10)
        # GAC: drop BAF/GAC split optimization, fix conservative removal + max EBCT, and fix
        # required_BV (computed from its cubic constraint) so the 0-DOF solve converges.
        nz.total_toc_removal.deactivate()
        nz.GAC.EBCT[0].fix(GAC_EBCT_CONSERV)
        nz.GAC.removal_frac_mass_comp[0, "toc"].fix(GAC_REMOVAL_CONSERV)
        calculate_variable_from_constraint(nz.GAC.required_BV[0], nz.required_BV_constraint[0])
        nz.required_BV_constraint.deactivate()
        nz.GAC.required_BV[0].fix()
        res = dpr.solve(m)
    eff_toc = value(m.fs.treated_nonRO.properties[0].conc_mass_comp["toc"]) * 1000
    return (check_optimal_termination(res), value(m.fs.LCOW),
            value(nz.GAC.EBCT[0]), value(nz.GAC.removal_frac_mass_comp[0, "toc"]), eff_toc)


def _cbat_rescue(flow_mgd, conservative, state):
    last = None
    for eps in PERTURB:
        try:
            ok, lcow, gac, grem, etoc = _cbat_point(flow_mgd * (1 + eps), conservative, state)
            if ok:
                return lcow, gac, grem, etoc
            last = (lcow, gac, grem, etoc)
        except Exception:
            last = None
    return (float("nan"),) * 4 if last is None else (float("nan"), last[1], last[2], last[3])


def run_cbat_opt_vs_conserv(states=CBAT_STATES):
    os.makedirs(OVC_DIR, exist_ok=True)
    for state in states:
        print(f"\n========== CBAT opt-vs-conserv: {state} ==========")
        rows = []
        csv = os.path.join(OVC_DIR, f"optimal_vs_conserv_CBAT_{state}.csv")
        for label, cons in CBAT_CASES:
            for f in FLOWS_MGD:
                lcow, gac, grem, etoc = _cbat_rescue(f, cons, state)
                rows.append({"state": state, "case": label, "flow_MGD": f, "LCOW": lcow,
                             "gac_EBCT": gac, "gac_removal": grem, "eff_TOC_mgL": etoc})
                print(f"[{state}][{label:12s}] {f:4d} MGD -> LCOW={lcow:.4f} "
                      f"gacEBCT={gac:.1f} gacRem={grem:.3f} effTOC={etoc:.2f}")
                pd.DataFrame(rows).to_csv(csv, index=False)
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, _ in CBAT_CASES:
            d = df[df["case"] == label].dropna(subset=["LCOW"]).sort_values("flow_MGD")
            ax.plot(d["flow_MGD"], d["LCOW"], marker="o", label=label)
        ax.set_xlabel("System capacity (MGD)"); ax.set_ylabel("LCOW ($/m$^3$)")
        ax.set_title(f"DPR CBAT ({state}): Optimized vs Conservative\nLCOW vs system capacity")
        ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
        png = os.path.join(OVC_DIR, f"optimal_vs_conserv_CBAT_{state}.png")
        fig.savefig(png, dpi=150); plt.close(fig)
        print("saved:", csv, "|", png)


# ===========================================================================
# Part 3 -- Monte Carlo worst/best optimal-LCOW envelopes + uniform restyle
# ===========================================================================
ENV_FLOWS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# MC input-range corners (worst -> max LCOW, best -> min LCOW)
ENV_INPUTS = {
    "worst": {"toc": 0.015, "tds": 1.0, "brine": 0.66},
    "best":  {"toc": 0.007, "tds": 0.5, "brine": 0.05},
}
RBAT_PT, RBAT_CURVE = "yellowgreen", "darkgreen"
CBAT_PT, CBAT_CURVE = "mediumpurple", "indigo"
# (state, train, MC csv filename, label, point color, curve color)
ENV_JOBS = [
    ("CA", "RBAT", "mc_v2_CA_RBAT_ROspiral_add_const_n1000.csv", "CA / RBAT (spiral)", RBAT_PT, RBAT_CURVE),
    ("CO", "RBAT", "mc_v2_CO_RBAT_ROspiral_add_const_n1000.csv", "CO / RBAT (spiral)", RBAT_PT, RBAT_CURVE),
    ("FL", "RBAT", "mc_v2_FL_RBAT_ROspiral_add_const_n1000.csv", "FL / RBAT (spiral)", RBAT_PT, RBAT_CURVE),
    ("CO", "CBAT", "mc_v2_CO_CBAT_n1000.csv", "CO / CBAT", CBAT_PT, CBAT_CURVE),
    ("FL", "CBAT", "mc_v2_FL_CBAT_n1000.csv", "FL / CBAT", CBAT_PT, CBAT_CURVE),
]


def _env_optimal_lcow(state, train, flow_mgd, mode):
    inp = ENV_INPUTS[mode]
    LRV, det, st, tt, eff, sol = dpr.DPR_initial_setting(
        state=state, treatment_train=train, effluent_type="TERTIARY")
    build = dpr.build_RO if train == "RBAT" else dpr.build_nonRO
    m = build(solute_list=sol, state=st, effluent_type=eff, treatment_train=tt,
              LRVO3_req=LRV["ozone_crypto_lrv_required"],
              LRVClvirus_req=LRV["cl2_virus_lrv_required"],
              LRVClgiardia_req=LRV["cl2_giardia_lrv_required"])
    dpr.set_operating_conditions(m, solute_list=sol, treatment_train=tt)
    m.fs.feed.flow_vol[0].fix(flow_mgd * MGD)
    m.fs.feed.conc_mass_comp[0, "toc"].fix(inp["toc"])
    if train == "RBAT":
        m.fs.feed.conc_mass_comp[0, "tds"].fix(inp["tds"])
    dpr.solve(m.fs.feed)
    dpr.scale_system(m, treatment_train=tt)
    dpr.initialize_system(m, treatment_train=tt)
    dpr.solve(m)
    dpr.add_costing(m, treatment_train=tt)
    dpr.initialize_costing(m, treatment_train=tt)
    if train == "RBAT":
        bd = m.fs.zo_costing.brine_disposal_cost
        bd.fix(inp["brine"]) if bd.is_variable_type() else bd.set_value(inp["brine"])
    dpr.optimize_operation(m, state=st, effluent_type=eff, treatment_train=tt)
    res = dpr.solve(m)
    return check_optimal_termination(res), value(m.fs.LCOW)


def _env_curve(state, train, mode):
    ys = []
    for f in ENV_FLOWS:
        v = float("nan")
        for eps in PERTURB:
            try:
                ok, lc = _env_optimal_lcow(state, train, f * (1 + eps), mode)
                if ok:
                    v = lc
                    break
            except Exception:
                pass
        ys.append(v)
        print(f"  [{train} {state} {mode}] {f:4d} MGD -> {v:.4f}")
    return np.array(ys, dtype=float)


def run_mc_envelopes():
    """Compute worst/best envelope curves and re-plot the MC figures (uniform single color +
    common axes + overlay curves). Reads existing MC CSVs in output/monte_carlo/."""
    # common axes across all MC csvs
    xlo = xhi = ylo = yhi = None
    for _, _, fname, *_ in ENV_JOBS:
        d = pd.read_csv(os.path.join(MC_DIR, fname))
        d.columns = [c.lstrip("# ").strip() for c in d.columns]
        d = d.dropna(subset=["LCOW"])
        fx = d["feed_flow"] / MGD
        xlo, xhi = (fx.min() if xlo is None else min(xlo, fx.min())), (fx.max() if xhi is None else max(xhi, fx.max()))
        ylo, yhi = (d["LCOW"].min() if ylo is None else min(ylo, d["LCOW"].min())), (d["LCOW"].max() if yhi is None else max(yhi, d["LCOW"].max()))
    xlim = (max(0, xlo - 0.03 * (xhi - xlo)), xhi + 0.03 * (xhi - xlo))
    ylim = (max(0, ylo - 0.03 * (yhi - ylo)), yhi + 0.03 * (yhi - ylo))

    fx = np.array(ENV_FLOWS, dtype=float)
    combined = []
    for state, train, fname, lbl, pt, cc in ENV_JOBS:
        print(f"=== MC envelope: {train} {state} ===")
        yw = _env_curve(state, train, "worst")
        yb = _env_curve(state, train, "best")
        wlbl = "worst-case optimal (max TOC/TDS/brine)" if train == "RBAT" else "worst-case optimal (max TOC)"
        blbl = "best-case optimal (min TOC/TDS/brine)" if train == "RBAT" else "best-case optimal (min TOC)"
        curves = [
            {"x": fx[~np.isnan(yw)], "y": yw[~np.isnan(yw)], "color": cc, "label": wlbl, "ls": "-"},
            {"x": fx[~np.isnan(yb)], "y": yb[~np.isnan(yb)], "color": cc, "label": blbl, "ls": "--"},
        ]
        png, n, nf = plot_flow_vs_lcow(
            os.path.join(MC_DIR, fname), label=lbl, color_by=None, point_color=pt,
            xlim=xlim, ylim=ylim, curves=curves)
        pd.DataFrame({"flow_MGD": ENV_FLOWS, "worst_LCOW": yw, "best_LCOW": yb}).to_csv(
            os.path.join(MC_DIR, fname.replace(".csv", "_envelope.csv")), index=False)
        for i, f in enumerate(ENV_FLOWS):
            combined.append({"train": train, "state": state, "flow_MGD": f,
                             "worst_LCOW": yw[i], "best_LCOW": yb[i]})
        print(f"  -> {os.path.basename(png)}")
    pd.DataFrame(combined).to_csv(os.path.join(MC_DIR, "mc_envelopes_all.csv"), index=False)
    print("saved:", os.path.join(MC_DIR, "mc_envelopes_all.csv"))


# ===========================================================================
# Part 4 -- Combined RBAT vs CBAT, Optimized vs Conservative (per state)
# ===========================================================================
def run_rbat_cbat_combined(states=("CO", "FL"), rbat_cons_rec=0.40):
    """One figure per state overlaying RBAT and CBAT, each Optimized (solid) and
    Conservative (dashed). RBAT light green, CBAT light purple. Reads the existing
    optimal-vs-conservative CSVs (Part 1 / Part 2) -- no re-solving.
    RBAT's conservative case uses recovery = rbat_cons_rec (default 0.40)."""
    rbat_cons_case = f"Conserv rec={rbat_cons_rec:.2f}"

    def _series(df, case):
        d = df[df["case"] == case].dropna(subset=["LCOW"]).sort_values("flow_MGD")
        return d["flow_MGD"], d["LCOW"]

    for state in states:
        rb = pd.read_csv(os.path.join(OVC_DIR, f"optimal_vs_conserv_{state}.csv"))
        cb = pd.read_csv(os.path.join(OVC_DIR, f"optimal_vs_conserv_CBAT_{state}.csv"))
        fig, ax = plt.subplots(figsize=(8, 6))
        x, y = _series(rb, "Optimized")
        ax.plot(x, y, color=RBAT_PT, ls="-", marker="o", ms=4, label="RBAT Optimized")
        x, y = _series(rb, rbat_cons_case)
        ax.plot(x, y, color=RBAT_PT, ls="--", marker="o", ms=4,
                label=f"RBAT Conservative (rec={rbat_cons_rec:.2f})")
        x, y = _series(cb, "Optimized")
        ax.plot(x, y, color=CBAT_PT, ls="-", marker="s", ms=4, label="CBAT Optimized")
        x, y = _series(cb, "Conservative")
        ax.plot(x, y, color=CBAT_PT, ls="--", marker="s", ms=4, label="CBAT Conservative")
        ax.set_xlabel("System capacity (MGD)"); ax.set_ylabel("LCOW ($/m$^3$)")
        ax.set_title(f"DPR {state}: RBAT vs CBAT -- Optimized vs Conservative\n"
                     "LCOW vs system capacity")
        ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
        png = os.path.join(OVC_DIR, f"optimal_vs_conserv_RBATvsCBAT_{state}.png")
        fig.savefig(png, dpi=150); plt.close(fig)
        print("saved:", png)


# ===========================================================================
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("rbat", "all"):
        run_rbat_opt_vs_conserv()
    if cmd in ("cbat", "all"):
        run_cbat_opt_vs_conserv()
    if cmd in ("envelopes", "env", "all"):
        run_mc_envelopes()
    if cmd in ("combined", "all"):
        run_rbat_cbat_combined()
