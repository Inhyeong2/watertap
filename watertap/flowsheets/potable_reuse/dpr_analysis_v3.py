"""Consolidated analysis/figure drivers for the spiral (constrained-RO) DPR flowsheet.

Combines what used to be three separate scripts:
  1. Optimized vs Conservative LCOW vs system capacity -- RBAT  (recovery-stepped)
  2. Optimized vs Conservative LCOW vs system capacity -- CBAT  (GAC conservative)
  3. Monte Carlo worst/best optimal-LCOW envelope curves + uniform restyle of MC figures

All three import the spiral flowsheet model (DPR_flowsheet_v2_spiral) and the shared plot
helper (plot_monte_carlo). The Monte Carlo *sampling* itself lives in DPR_sweep_v2_spiral.py;
this module only post-processes / overlays envelope curves on those results.

Run from repo root:
  python -m watertap.flowsheets.potable_reuse.dpr_analysis_v3 rbat        # opt vs conserv (RBAT)
  python -m watertap.flowsheets.potable_reuse.dpr_analysis_v3 cbat        # opt vs conserv (CBAT)
  python -m watertap.flowsheets.potable_reuse.dpr_analysis_v3 envelopes   # MC worst/best + restyle
  python -m watertap.flowsheets.potable_reuse.dpr_analysis_v3 all         # all of the above
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyomo.environ import value, check_optimal_termination, units as pyunits
from pyomo.util.calc_var_value import calculate_variable_from_constraint
import watertap.flowsheets.potable_reuse.DPR_flowsheet_v3 as dpr
from watertap.flowsheets.potable_reuse.plot_monte_carlo import plot_flow_vs_lcow

MGD = 0.0438126

# Number of RO stages for the multi-stage RBAT flowsheet (single feed pump + retentate
# cascade + permeate mixer). N_STAGES=1 reproduces the single-stage spiral results exactly.
N_STAGES = 3


# --- Multi-stage RO accessors -------------------------------------------------------------
# In the multi-stage flowsheet m.fs.RO_main.RO is INDEXED by stage, so the single-stage
# attributes (recovery_vol_phase / area / mixed_permeate) no longer exist on a scalar RO.
# These helpers expose the system-level equivalents used throughout this analysis.
def ro_sys_recovery(m):
    """System volumetric recovery across the whole RO array (= total permeate / RO feed)."""
    return value(m.fs.RO_main.sys_recovery_vol)


def ro_total_area(m):
    """Total membrane area summed over all stages [m2]."""
    return sum(value(m.fs.RO_main.RO[i].area) for i in m.fs.RO_main.stages)


def ro_permeate_flow(m):
    """Blended permeate volumetric flow (P_mixer outlet) [m3/s]."""
    return value(m.fs.RO_main.P_mixer.mixed_state[0].flow_vol_phase["Liq"])


def fix_conservative_recovery(m, target):
    """Conservative (non-optimized) design: fix EACH stage's recovery to ``target``.

    This is the direct multi-stage generalization of the single-stage conservative case
    (where the one stage's recovery == the system recovery): every membrane stage runs at the
    same prescribed per-stage recovery, and the resulting system recovery is
    1 - (1 - target)^n (higher with more stages). Splitting a low SYSTEM target equally across
    stages instead would force an unrealistically low per-stage recovery that conflicts with
    the velocity/rejection/flux constraints, so per-stage fixing is used."""
    for i in m.fs.RO_main.stages:
        rv = m.fs.RO_main.RO[i].recovery_vol_phase[0, "Liq"]
        # Clear the (tapered) optimization bounds first: a fixed value outside its bounds is
        # flagged as trivially infeasible by the presolve. A fixed var is a constant, so bounds
        # are moot once fixed.
        rv.setlb(None)
        rv.setub(None)
        rv.fix(target)


def _capex_ratio(m, train):
    """annualized_capex / (annualized_capex + opex).
    annualized_capex = total_capital_cost * CRF (same annualization as LCOW; zo CRF == ro CRF).
    opex = total_operating_cost (+ brine_disposal_cost for RBAT -- in this model brine disposal
    sits in total_externalities, not total_operating_cost, so it must be added explicitly)."""
    yr = pyunits.USD_2020 / pyunits.year
    ann_capex = value(pyunits.convert(
        m.fs.total_capital_cost * m.fs.zo_costing.capital_recovery_factor, to_units=yr))
    opex = value(pyunits.convert(m.fs.total_operating_cost, to_units=yr))
    if train in ("RBAT", "IPR"):
        # RBAT and IPR both build m.fs.brine_disposal_cost as an externality (RO array + ERD +
        # brine product), so brine disposal must be added to opex for both.
        opex += value(pyunits.convert(m.fs.brine_disposal_cost, to_units=yr))
    return ann_capex / (ann_capex + opex)
_HERE = os.path.dirname(os.path.abspath(__file__))
OVC_DIR = os.path.join(_HERE, "output", f"optimal_vs_conserv_n{N_STAGES}")   # opt-vs-conserv outputs (stage-specific, so single-stage results are not overwritten)
MC_DIR = os.path.join(_HERE, "output", "monte_carlo")           # Monte Carlo outputs
FLOWS_MGD = [1, 5, 10, 20, 35, 50, 65, 80, 100]
PERTURB = (0.0, 1e-4, 5e-4, 1e-3)   # knife-edge rescue: nudge feed flow slightly


# ===========================================================================
# Part 1 -- Optimized vs Conservative, RBAT (recovery-stepped)
# ===========================================================================
# (label, recovery)  recovery=None -> Optimized (free). For the MULTI-STAGE array `recovery`
# is the fixed PER-STAGE recovery, so the system recovery is 1-(1-rec)^n_stages. The values
# were lowered from the single-stage set (0.45-0.30): at n_stages>=2 a per-stage 0.45 gives a
# system recovery ~= the optimized one (and at n=3 a uniform 0.45 is infeasible -- the
# concentrated back stage cannot sustain it), so it showed almost no Optimized-vs-Conservative
# gap. The 0.40-0.20 range spans system recoveries clearly BELOW optimal for both n=2 and n=3
# (n=3: 78/73/66/58/49%; n=2: 64/58/51/44/36%) and all converge (per-stage 0.15 does not).
RBAT_CASES = [("Optimized", None), ("Conserv rec=0.40", 0.40), ("Conserv rec=0.35", 0.35),
              ("Conserv rec=0.30", 0.30), ("Conserv rec=0.25", 0.25), ("Conserv rec=0.20", 0.20)]
RBAT_STATES = ["CA", "CO", "FL"]


def _rbat_point(flow_mgd, recovery, state, tds=0.5):
    LRV_req, det, st, tt, eff, sol = dpr.DPR_initial_setting(
        state=state, treatment_train="RBAT", effluent_type="TERTIARY")
    m = dpr.build_RO(solute_list=sol, state=st, effluent_type=eff, treatment_train=tt,
                     LRVO3_req=LRV_req["ozone_crypto_lrv_required"],
                     LRVClvirus_req=LRV_req["cl2_virus_lrv_required"],
                     LRVClgiardia_req=LRV_req["cl2_giardia_lrv_required"],
                     n_stages=N_STAGES)
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
    nonRO = m.fs.non_RO
    res = dpr.solve(m)  # optimized solution (also warm start for the conservative re-solve)
    if recovery is not None:
        # Conservative: over-dose ozone (drop O3toTOC_ratio_constraint, force contact_time 10
        # + O3toTOC 1) + BAF EBCT 30 + UV H2O2 10 + RO recovery fixed. For the multi-stage
        # array, ``recovery`` is the target SYSTEM recovery, split equally across stages.
        nonRO.Ozone.O3toTOC_ratio_constraint.deactivate()
        nonRO.Ozone.contact_time[0].fix(10)
        nonRO.Ozone.O3toTOC[0].fix(1.0)
        nonRO.BAF.EBCT[0].fix(30)
        nonRO.UV_AOP.hydrogen_peroxide_dose[0].fix(10)
        fix_conservative_recovery(m, recovery)
        res = dpr.solve(m)
    # System average flux [LMH] = total blended permeate / total membrane area.
    Ja = ro_permeate_flow(m) / ro_total_area(m) * 3.6e6
    cr = _capex_ratio(m, "RBAT")
    return check_optimal_termination(res), value(m.fs.LCOW), ro_sys_recovery(m), Ja, cr


def _rbat_rescue(flow_mgd, recovery, state):
    last = None
    for eps in PERTURB:
        try:
            ok, lcow, rec, flux, cr = _rbat_point(flow_mgd * (1 + eps), recovery, state)
            if ok:
                return lcow, rec, flux, cr
            last = (lcow, rec, flux, cr)
        except Exception:
            last = None
    return (float("nan"),) * 4 if last is None else (float("nan"), last[1], last[2], last[3])


def run_rbat_opt_vs_conserv(states=RBAT_STATES):
    os.makedirs(OVC_DIR, exist_ok=True)
    for state in states:
        print(f"\n========== RBAT opt-vs-conserv: {state} ==========")
        rows = []
        csv = os.path.join(OVC_DIR, f"optimal_vs_conserv_{state}.csv")
        for label, rec in RBAT_CASES:
            for f in FLOWS_MGD:
                lcow, r, flux, cr = _rbat_rescue(f, rec, state)
                rows.append({"state": state, "case": label, "recovery_target": rec,
                             "flow_MGD": f, "LCOW": lcow, "recovery": r, "flux_LMH": flux,
                             "capex_ratio": cr})
                print(f"[{state}][{label:18s}] {f:4d} MGD -> LCOW={lcow:.4f} recov={r:.3f} "
                      f"flux={flux:.1f} capexR={cr:.3f}")
                pd.DataFrame(rows).to_csv(csv, index=False)
        df = pd.DataFrame(rows)
        for ycol, ylab, tag in [("LCOW", "LCOW ($/m$^3$)", ""),
                                ("capex_ratio", "CAPEX ratio [ann.CAPEX/(ann.CAPEX+OPEX)]", "_capexratio")]:
            fig, ax = plt.subplots(figsize=(8.5, 6))
            for label, rec in RBAT_CASES:
                d = df[df["case"] == label].dropna(subset=[ycol]).sort_values("flow_MGD")
                if d.empty:
                    # A conservative per-stage recovery that does not converge for this
                    # n_stages (e.g. uniform 0.45/stage is infeasible for a 3-stage cascade --
                    # the concentrated back stage cannot sustain it) is simply omitted.
                    continue
                # Headline the SYSTEM recovery (= 1-(1-r)^n, read from the achieved values),
                # with the fixed per-stage recovery noted in parentheses.
                sysrec = df[df["case"] == label]["recovery"].mean() * 100
                leg = (f"Optimized (sys recovery ~{sysrec:.0f}%)" if rec is None
                       else f"Conservative: sys recovery {sysrec:.0f}% (per-stage {rec*100:.0f}%)")
                ax.plot(d["flow_MGD"], d[ycol], marker="o", label=leg)
            ax.set_xlabel("System capacity (MGD)"); ax.set_ylabel(ylab)
            ax.set_title(f"DPR RBAT ({state}, {N_STAGES}-stage): Optimized vs Conservative\n"
                         f"{ycol} vs system capacity  (label = SYSTEM recovery; per-stage in parens)")
            ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
            png = os.path.join(OVC_DIR, f"optimal_vs_conserv_{state}{tag}.png")
            fig.savefig(png, dpi=150); plt.close(fig)
            print("saved:", png)
        print("saved:", csv)


# ===========================================================================
# Part 1b -- Optimized vs Conservative, IPR (recovery-stepped)
# ===========================================================================
# IPR (UF->RO->UV/AOP, CA / SECONDARY). Same recovery-stepped conservative idea as RBAT: for each
# fixed per-stage RO recovery the SYSTEM recovery is 1-(1-rec)^n. The ONLY conservative knobs IPR
# has are the RO per-stage recovery and the UV/AOP H2O2 dose -- there is no ozone/BAF/final-Cl to
# over-dose (those units are absent from the IPR train). The recovery set matches RBAT so the two
# opt-vs-conserv figures are directly comparable. CA only (IPR is CA-regulation-supported).
IPR_OVC_CASES = [("Optimized", None), ("Conserv rec=0.40", 0.40), ("Conserv rec=0.35", 0.35),
                 ("Conserv rec=0.30", 0.30), ("Conserv rec=0.25", 0.25), ("Conserv rec=0.20", 0.20)]
IPR_OVC_STATES = ["CA"]


def _ipr_point(flow_mgd, recovery, state):
    LRV, det, st, tt, eff, sol = dpr.DPR_initial_setting(
        state=state, treatment_train="IPR", effluent_type="SECONDARY")
    m = dpr.build_RO(solute_list=sol, state=st, effluent_type=eff, treatment_train=tt,
                     n_stages=N_STAGES)
    dpr.set_operating_conditions(m, solute_list=sol, treatment_train=tt)
    m.fs.feed.flow_vol[0].fix(flow_mgd * MGD)
    dpr.solve(m.fs.feed)
    dpr.scale_system(m, treatment_train=tt)
    dpr.initialize_system(m, treatment_train=tt)
    dpr.solve(m)
    dpr.add_costing(m, treatment_train=tt)
    dpr.initialize_costing(m, treatment_train=tt)
    dpr.optimize_operation(m, state=st, effluent_type=eff, treatment_train=tt)
    nonRO = m.fs.non_RO
    res = dpr.solve(m)  # optimized solution (also warm start for the conservative re-solve)
    if recovery is not None:
        # Conservative: over-dose UV/AOP H2O2 (max of its 2-10 mg/L range) + fix RO per-stage
        # recovery. IPR has no ozone/BAF/Cl2 to over-dose.
        nonRO.UV_AOP.hydrogen_peroxide_dose[0].fix(10)
        fix_conservative_recovery(m, recovery)
        res = dpr.solve(m)
    # System average flux [LMH] = total blended permeate / total membrane area.
    Ja = ro_permeate_flow(m) / ro_total_area(m) * 3.6e6
    cr = _capex_ratio(m, "IPR")
    return check_optimal_termination(res), value(m.fs.LCOW), ro_sys_recovery(m), Ja, cr


def _ipr_rescue(flow_mgd, recovery, state):
    last = None
    for eps in PERTURB:
        try:
            ok, lcow, rec, flux, cr = _ipr_point(flow_mgd * (1 + eps), recovery, state)
            if ok:
                return lcow, rec, flux, cr
            last = (lcow, rec, flux, cr)
        except Exception:
            last = None
    return (float("nan"),) * 4 if last is None else (float("nan"), last[1], last[2], last[3])


def run_ipr_opt_vs_conserv(states=IPR_OVC_STATES):
    os.makedirs(OVC_DIR, exist_ok=True)
    for state in states:
        print(f"\n========== IPR opt-vs-conserv: {state} ==========")
        rows = []
        csv = os.path.join(OVC_DIR, f"optimal_vs_conserv_IPR_{state}.csv")
        for label, rec in IPR_OVC_CASES:
            for f in FLOWS_MGD:
                lcow, r, flux, cr = _ipr_rescue(f, rec, state)
                rows.append({"state": state, "case": label, "recovery_target": rec,
                             "flow_MGD": f, "LCOW": lcow, "recovery": r, "flux_LMH": flux,
                             "capex_ratio": cr})
                print(f"[{state}][{label:18s}] {f:4d} MGD -> LCOW={lcow:.4f} recov={r:.3f} "
                      f"flux={flux:.1f} capexR={cr:.3f}")
                pd.DataFrame(rows).to_csv(csv, index=False)
        df = pd.DataFrame(rows)
        for ycol, ylab, tag in [("LCOW", "LCOW ($/m$^3$)", ""),
                                ("capex_ratio", "CAPEX ratio [ann.CAPEX/(ann.CAPEX+OPEX)]", "_capexratio")]:
            fig, ax = plt.subplots(figsize=(8.5, 6))
            for label, rec in IPR_OVC_CASES:
                d = df[df["case"] == label].dropna(subset=[ycol]).sort_values("flow_MGD")
                if d.empty:
                    # A conservative per-stage recovery that does not converge for this n_stages
                    # (e.g. a high uniform recovery is infeasible for the concentrated back stage
                    # of a 3-stage cascade) is simply omitted.
                    continue
                sysrec = df[df["case"] == label]["recovery"].mean() * 100
                leg = (f"Optimized (sys recovery ~{sysrec:.0f}%)" if rec is None
                       else f"Conservative: sys recovery {sysrec:.0f}% (per-stage {rec*100:.0f}%)")
                ax.plot(d["flow_MGD"], d[ycol], marker="o", label=leg)
            ax.set_xlabel("System capacity (MGD)"); ax.set_ylabel(ylab)
            ax.set_title(f"IPR ({state}, {N_STAGES}-stage): Optimized vs Conservative\n"
                         f"{ycol} vs system capacity  (label = SYSTEM recovery; per-stage in parens)")
            ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
            png = os.path.join(OVC_DIR, f"optimal_vs_conserv_IPR_{state}{tag}.png")
            fig.savefig(png, dpi=150); plt.close(fig)
            print("saved:", png)
        print("saved:", csv)


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
    cr = _capex_ratio(m, "CBAT")
    return (check_optimal_termination(res), value(m.fs.LCOW),
            value(nz.GAC.EBCT[0]), value(nz.GAC.removal_frac_mass_comp[0, "toc"]), eff_toc, cr)


def _cbat_rescue(flow_mgd, conservative, state):
    last = None
    for eps in PERTURB:
        try:
            ok, lcow, gac, grem, etoc, cr = _cbat_point(flow_mgd * (1 + eps), conservative, state)
            if ok:
                return lcow, gac, grem, etoc, cr
            last = (lcow, gac, grem, etoc, cr)
        except Exception:
            last = None
    return (float("nan"),) * 5 if last is None else (float("nan"), last[1], last[2], last[3], last[4])


def run_cbat_opt_vs_conserv(states=CBAT_STATES):
    os.makedirs(OVC_DIR, exist_ok=True)
    for state in states:
        print(f"\n========== CBAT opt-vs-conserv: {state} ==========")
        rows = []
        csv = os.path.join(OVC_DIR, f"optimal_vs_conserv_CBAT_{state}.csv")
        for label, cons in CBAT_CASES:
            for f in FLOWS_MGD:
                lcow, gac, grem, etoc, cr = _cbat_rescue(f, cons, state)
                rows.append({"state": state, "case": label, "flow_MGD": f, "LCOW": lcow,
                             "gac_EBCT": gac, "gac_removal": grem, "eff_TOC_mgL": etoc,
                             "capex_ratio": cr})
                print(f"[{state}][{label:12s}] {f:4d} MGD -> LCOW={lcow:.4f} "
                      f"gacEBCT={gac:.1f} gacRem={grem:.3f} effTOC={etoc:.2f} capexR={cr:.3f}")
                pd.DataFrame(rows).to_csv(csv, index=False)
        df = pd.DataFrame(rows)
        for ycol, ylab, tag in [("LCOW", "LCOW ($/m$^3$)", ""),
                                ("capex_ratio", "CAPEX ratio [ann.CAPEX/(ann.CAPEX+OPEX)]", "_capexratio")]:
            fig, ax = plt.subplots(figsize=(8, 6))
            for label, _ in CBAT_CASES:
                d = df[df["case"] == label].dropna(subset=[ycol]).sort_values("flow_MGD")
                ax.plot(d["flow_MGD"], d[ycol], marker="o", label=label)
            ax.set_xlabel("System capacity (MGD)"); ax.set_ylabel(ylab)
            ax.set_title(f"DPR CBAT ({state}): Optimized vs Conservative\n{ycol} vs system capacity")
            ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
            png = os.path.join(OVC_DIR, f"optimal_vs_conserv_CBAT_{state}{tag}.png")
            fig.savefig(png, dpi=150); plt.close(fig)
            print("saved:", png)
        print("saved:", csv)


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
IPR_PT, IPR_CURVE = "skyblue", "steelblue"
# IPR envelope corners: TOC swept 10-15 mg/L; TDS pinned at the SECONDARY default ("tds": None ->
# leave at the profile value, not fixed here); brine over the MC range. IPR has no TDS/ozone/Cl2
# knobs (UF->RO->UV/AOP), so the worst/best corner is (TOC, brine) only.
ENV_INPUTS_IPR = {
    "worst": {"toc": 0.015, "tds": None, "brine": 0.66},
    "best":  {"toc": 0.010, "tds": None, "brine": 0.05},
}
# (state, train, MC csv filename, label, point color, curve color)
ENV_JOBS = [
    ("CA", "RBAT", "mc_v2_CA_RBAT_ROspiral_add_const_n1000.csv", "CA / RBAT (spiral)", RBAT_PT, RBAT_CURVE),
    ("CO", "RBAT", "mc_v2_CO_RBAT_ROspiral_add_const_n1000.csv", "CO / RBAT (spiral)", RBAT_PT, RBAT_CURVE),
    ("FL", "RBAT", "mc_v2_FL_RBAT_ROspiral_add_const_n1000.csv", "FL / RBAT (spiral)", RBAT_PT, RBAT_CURVE),
    ("CO", "CBAT", "mc_v2_CO_CBAT_n1000.csv", "CO / CBAT", CBAT_PT, CBAT_CURVE),
    ("FL", "CBAT", "mc_v2_FL_CBAT_n1000.csv", "FL / CBAT", CBAT_PT, CBAT_CURVE),
]


def _env_optimal_lcow(state, train, flow_mgd, mode):
    # IPR (UF->RO->UV/AOP, SECONDARY) reuses the general fresh-build solver; tds=None leaves feed TDS
    # at the SECONDARY default (IPR does not sweep TDS).
    if train == "IPR":
        inp = ENV_INPUTS_IPR[mode]
        ok, m = _solve_train_at(state, flow_mgd, inp["toc"], inp["tds"], inp["brine"],
                                train="IPR", effluent_type="SECONDARY")
        return ok, value(m.fs.LCOW)
    inp = ENV_INPUTS[mode]
    LRV, det, st, tt, eff, sol = dpr.DPR_initial_setting(
        state=state, treatment_train=train, effluent_type="TERTIARY")
    build = dpr.build_RO if train == "RBAT" else dpr.build_nonRO
    build_kw = dict(solute_list=sol, state=st, effluent_type=eff, treatment_train=tt,
                    LRVO3_req=LRV["ozone_crypto_lrv_required"],
                    LRVClvirus_req=LRV["cl2_virus_lrv_required"],
                    LRVClgiardia_req=LRV["cl2_giardia_lrv_required"])
    if train == "RBAT":
        build_kw["n_stages"] = N_STAGES
    m = build(**build_kw)
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


# Envelope curves are theoretically monotone-decreasing in capacity (economy of scale), and a
# best-case LCOW must lie below the worst-case. The LCOW optimization is non-convex, though, so
# an isolated point can converge to a WORSE local optimum (Ipopt terminates "optimal" but in
# the wrong basin), showing up as an unphysical upward "bump". These control the automatic
# detect-and-repair pass in _env_curve.
ENV_SMOOTH_TOL = 0.05  # flag an interior point sitting >5% above the interpolation of its neighbors
ENV_MULTISTART_PERTURB = (0.0, 1e-4, -1e-4, 5e-4, -5e-4, 1e-3, -1e-3,
                          3e-3, -3e-3, 5e-3, -5e-3, 1e-2, -1e-2)


def _env_lcow_multistart(state, train, flow, mode):
    """Multi-start optimal LCOW: solve from several nearby starting points (tiny flow
    perturbations seed different initializations) and keep the MINIMUM converged LCOW. This
    escapes the worse local-optimum basin a single fresh-build solve can fall into."""
    best = float("inf")
    for eps in ENV_MULTISTART_PERTURB:
        try:
            ok, lc = _env_optimal_lcow(state, train, flow * (1 + eps), mode)
            if ok and lc < best:
                best = lc
        except Exception:
            pass
    return best if best < float("inf") else float("nan")


def _env_curve(state, train, mode):
    # 1. First pass: cheap first-converged LCOW at each flow.
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
    ys = np.array(ys, dtype=float)

    # 2. Automatic repair pass: on a monotone-decreasing curve every interior point should sit
    # between its neighbors, so any point well ABOVE the linear interpolation of its neighbors is
    # a non-convex local-optimum artifact (e.g. the 15 MGD best-case / 40 MGD worst-case bumps).
    # Recompute those with a multi-start and keep the lower LCOW. Repeat a couple of passes since
    # fixing one point can unmask an adjacent one.
    fx = np.array(ENV_FLOWS, dtype=float)
    for _pass in range(2):
        changed = False
        for i in range(1, len(ys) - 1):
            c, l, r = ys[i], ys[i - 1], ys[i + 1]
            if np.isnan(c) or np.isnan(l) or np.isnan(r):
                continue
            ref = l + (r - l) * (fx[i] - fx[i - 1]) / (fx[i + 1] - fx[i - 1])
            if c > ref * (1 + ENV_SMOOTH_TOL):
                new = _env_lcow_multistart(state, train, fx[i], mode)
                if not np.isnan(new) and new < c - 1e-9:
                    print(f"  [repair {train} {state} {mode}] {int(fx[i]):4d} MGD: "
                          f"{c:.4f} -> {new:.4f} (was >{int(ENV_SMOOTH_TOL*100)}% above neighbors)")
                    ys[i] = new
                    changed = True
        if not changed:
            break
    return ys


def _solve_train_at(state, flow_mgd, toc, tds, brine, train="RBAT", effluent_type="TERTIARY"):
    """Build the multi-stage RO flowsheet (RBAT or IPR) at EXPLICIT feed inputs, optimize LCOW, and
    return (converged, model). A fresh build gives point-local scaling, so it escapes the warm-start
    local optimum a swept sample can get stuck in. ``tds=None`` leaves feed TDS at the effluent
    profile default (used for IPR, where TDS is pinned at the SECONDARY value and not swept)."""
    LRV, det, st, tt, eff, sol = dpr.DPR_initial_setting(
        state=state, treatment_train=train, effluent_type=effluent_type)
    build_kw = dict(solute_list=sol, state=st, effluent_type=eff, treatment_train=tt,
                    n_stages=N_STAGES)
    if tt != "IPR":  # IPR carries no ozone/Cl2 LRV requirements
        build_kw.update(LRVO3_req=LRV["ozone_crypto_lrv_required"],
                        LRVClvirus_req=LRV["cl2_virus_lrv_required"],
                        LRVClgiardia_req=LRV["cl2_giardia_lrv_required"])
    m = dpr.build_RO(**build_kw)
    dpr.set_operating_conditions(m, solute_list=sol, treatment_train=tt)
    m.fs.feed.flow_vol[0].fix(flow_mgd * MGD)
    m.fs.feed.conc_mass_comp[0, "toc"].fix(toc)
    if tds is not None:
        m.fs.feed.conc_mass_comp[0, "tds"].fix(tds)
    dpr.solve(m.fs.feed)
    dpr.scale_system(m, treatment_train=tt)
    dpr.initialize_system(m, treatment_train=tt)
    dpr.solve(m)
    dpr.add_costing(m, treatment_train=tt)
    dpr.initialize_costing(m, treatment_train=tt)
    bd = m.fs.zo_costing.brine_disposal_cost
    bd.fix(brine) if bd.is_variable_type() else bd.set_value(brine)
    dpr.optimize_operation(m, state=st, effluent_type=eff, treatment_train=tt)
    res = dpr.solve(m)
    return check_optimal_termination(res), m


def _solve_rbat_at(state, flow_mgd, toc, tds, brine):
    """Back-compat RBAT wrapper for _solve_train_at (TERTIARY effluent)."""
    return _solve_train_at(state, flow_mgd, toc, tds, brine,
                           train="RBAT", effluent_type="TERTIARY")


def repair_mc_outliers(state, mc_csv, worst_flows, worst_lcow, tol=0.01,
                       train="RBAT", effluent_type="TERTIARY", verbose=True):
    """Re-solve Monte Carlo samples whose LCOW lies ABOVE the worst-case envelope.

    The worst-case envelope (all feed inputs at their range maxima) is the theoretical upper
    bound of the optimized-LCOW cloud, so any sample above it is a warm-start local-optimum
    artifact of the sweep -- physically impossible, not a real value. Each such sample is
    re-solved from a fresh build with a multi-start over tiny flow perturbations; the lower-LCOW
    solution and its full output row replace the inflated one. Patches the CSV in place and
    returns the number of points repaired.

    ``train`` / ``effluent_type`` select the flowsheet (RBAT/TERTIARY or IPR/SECONDARY). For IPR
    the Monte Carlo does not sweep feed TDS, so the CSV has no ``feed_tds`` column; TDS is then left
    at the effluent profile default in the re-solve (``tds=None``)."""
    from watertap.flowsheets.potable_reuse.DPR_sweep_v3 import (
        set_up_sensitivity,
    )
    raw = pd.read_csv(mc_csv)
    orig_cols = list(raw.columns)
    raw.columns = [c.lstrip("# ").strip() for c in raw.columns]
    has_tds = "feed_tds" in raw.columns
    we = pd.Series(np.interp(raw["feed_flow"] / MGD, worst_flows, worst_lcow), index=raw.index)
    idxs = raw.index[raw["LCOW"].notna() & (raw["LCOW"] > we * (1 + tol))]
    n_fixed = 0
    for idx in idxs:
        flow = raw.at[idx, "feed_flow"] / MGD
        toc, brine = raw.at[idx, "feed_toc"], raw.at[idx, "brine_disposal_cost"]
        tds = raw.at[idx, "feed_tds"] if has_tds else None  # IPR: TDS pinned at profile default
        old = raw.at[idx, "LCOW"]
        bound = we[idx]
        best_lcow, best_m = old, None
        for eps in ENV_MULTISTART_PERTURB:
            try:
                ok, m = _solve_train_at(state, flow * (1 + eps), toc, tds, brine,
                                        train=train, effluent_type=effluent_type)
                lc = value(m.fs.LCOW)
                if ok and lc < best_lcow:
                    best_lcow, best_m = lc, m
            except Exception:
                pass
            # Early stop: once at/below the worst-case envelope the point is no longer an
            # artifact, so no need to keep multi-starting (the usual case is the eps=0 fresh
            # build already lands here).
            if best_lcow <= bound:
                break
        if best_m is not None and best_lcow < old - 1e-9:
            for k, v in set_up_sensitivity(best_m, treatment_train=train).items():
                if k in raw.columns:
                    try:
                        raw.at[idx, k] = value(v)
                    except Exception:
                        pass
            n_fixed += 1
            if verbose:
                print(f"  [mc-repair {state}] flow={flow:6.1f} MGD: LCOW {old:.3f} -> {best_lcow:.3f}")
    if n_fixed:
        raw.columns = orig_cols
        raw.to_csv(mc_csv, index=False)
    return n_fixed


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
        # Auto-repair MC samples sitting above the worst-case envelope (warm-start local-optimum
        # artifacts). RBAT only -- the worst envelope is the RBAT (TOC/TDS/brine) corner. This
        # patches the MC CSV in place, so the scatter plotted below no longer shows impossible
        # above-envelope points.
        if train == "RBAT":
            finite = ~np.isnan(yw)
            n_rep = repair_mc_outliers(state, os.path.join(MC_DIR, fname),
                                       fx[finite], yw[finite])
            if n_rep:
                print(f"  [mc-repair {state}] re-solved {n_rep} above-envelope sample(s)")
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

    def _series(df, case, ycol):
        d = df[df["case"] == case].dropna(subset=[ycol]).sort_values("flow_MGD")
        return d["flow_MGD"], d[ycol]

    for state in states:
        rb = pd.read_csv(os.path.join(OVC_DIR, f"optimal_vs_conserv_{state}.csv"))
        cb = pd.read_csv(os.path.join(OVC_DIR, f"optimal_vs_conserv_CBAT_{state}.csv"))
        for ycol, ylab, tag in [
            ("LCOW", "LCOW ($/m$^3$)", ""),
            ("capex_ratio", "CAPEX ratio [ann.CAPEX/(ann.CAPEX+OPEX)]", "_capexratio"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 6))
            x, y = _series(rb, "Optimized", ycol)
            ax.plot(x, y, color=RBAT_PT, ls="-", marker="o", ms=4, label="RBAT Optimized")
            x, y = _series(rb, rbat_cons_case, ycol)
            ax.plot(x, y, color=RBAT_PT, ls="--", marker="o", ms=4,
                    label=f"RBAT Conservative (rec={rbat_cons_rec:.2f})")
            x, y = _series(cb, "Optimized", ycol)
            ax.plot(x, y, color=CBAT_PT, ls="-", marker="s", ms=4, label="CBAT Optimized")
            x, y = _series(cb, "Conservative", ycol)
            ax.plot(x, y, color=CBAT_PT, ls="--", marker="s", ms=4, label="CBAT Conservative")
            ax.set_xlabel("System capacity (MGD)"); ax.set_ylabel(ylab)
            ax.set_title(f"DPR {state}: RBAT vs CBAT -- Optimized vs Conservative\n"
                         f"{ycol} vs system capacity")
            ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
            png = os.path.join(OVC_DIR, f"optimal_vs_conserv_RBATvsCBAT_{state}{tag}.png")
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
