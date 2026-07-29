"""
Multi-variable warm-start parameter sweep for DPR_flowsheet_v2.

Improvements over DPR_sweep_v1 (which only swept feed flow):

  1. Declarative, name-based sweep targets. Sweeps are specified as
     ``{handle_name: (min, max, nx)}`` and resolved through
     ``DPR_flowsheet_v2.get_sweep_handles`` -- so a multi-variable sweep never has to
     hard-code model paths like ``m.fs.RO_main.RO.A_comp[0, "H2O"]``. Any number of
     handles can be combined; parameter_sweep takes the Cartesian product, and the
     reused model warm-starts each sample from the previous one.

  2. Degree-of-freedom safety checks (in ``build_sweep_params``). Sweeping an input
     (already fixed) leaves DOF unchanged, but sweeping a *decision* variable fixes
     it -- removing a DOF and turning the run into a parametric study. Before the
     sweep runs this: predicts the post-sweep DOF and raises a clear error if it
     would go negative (over-constrained) instead of failing later as opaque NaNs;
     widens a target's bounds to cover the sweep range so a fixed end-of-range value
     cannot conflict with an ``optimize_operation`` bound; and warns which decision
     variables are being fixed (the result is no longer cost-optimal in those).

Like v1, each sample is solved with the plain ``DPR_flowsheet_v2.solve`` (warm start
+ the solver's own autoscaling). Per-sample rescaling was tried and dropped -- it did
not help and made the flow sweep worse (see DPR_flowsheet_v2 docstring).

Example (2-D sweep over feed flow x feed TDS)::

    from watertap.flowsheets.potable_reuse.DPR_sweep_v2 import run_sweep, MGD
    run_sweep(
        state="CA", treatment_train="RBAT",
        sweep_config={
            "feed_flow": (1 * MGD, 100 * MGD, 21),
            "feed_tds":  (0.5, 2.0, 5),
        },
    )
"""
import os

import numpy as np
from pyomo.environ import value, check_optimal_termination
from parameter_sweep import (
    LinearSample,
    UniformSample,
    PredeterminedRandomSample,
    parameter_sweep,
)
from idaes.core.util.model_statistics import degrees_of_freedom
# Spiral RO + added RO design constraints (flux<=20 LMH, v_exit>=0.1, length 6-8).
# This sweep is identical to DPR_sweep_v2 except it builds the spiral flowsheet, so the
# RBAT LCOW reflects the spiral/constrained RO. Output files carry a distinct tag so they
# do not overwrite the flat-RO Monte Carlo results.
import watertap.flowsheets.potable_reuse.DPR_flowsheet_v4 as dpr

# Number of RO stages for the multi-stage RBAT flowsheet (single feed pump + retentate
# cascade + permeate mixer). N_STAGES=1 reproduces the single-stage RO sweep exactly.
N_STAGES = 3

TAG = f"v4_n{N_STAGES}"  # stage-specific tag so n-stage MC results do not overwrite each other / older results

MGD = 0.0438126  # m3/s per MGD

# Sweep results are written here (next to this module, not the working directory),
# so output lands in the same place regardless of where the sweep is launched from.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _resolve_output_path(output_filename, sweep_config, state, treatment_train):
    """Resolve the CSV path. A bare filename goes into OUTPUT_DIR; a path with its
    own directory is respected as-is. None -> auto-named file in OUTPUT_DIR."""
    if output_filename is None:
        tag = "_".join(sweep_config.keys())
        output_filename = f"sweep_v2_{state}_{treatment_train}_{tag}.csv"
    # If the caller gave just a filename (no directory), place it in OUTPUT_DIR.
    if os.path.dirname(output_filename) == "":
        output_filename = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    return output_filename


def set_up_sensitivity(m, treatment_train="RBAT"):
    outputs = {
        "LCOW": m.fs.LCOW,
        "LCOT": m.fs.LCOT,
        "Total CAPEX": m.fs.total_capital_cost,
        "Total OPEX": m.fs.total_operating_cost,
        "SEC": m.fs.specific_energy_intensity,
    }
    if treatment_train == "RBAT":
        # System recovery (whole array) and the single shared feed-pump pressure.
        outputs["RO system recovery"] = m.fs.RO_main.sys_recovery_vol
        outputs["RO pressure (Pa)"] = (
            m.fs.RO_main.pump.control_volume.properties_out[0].pressure
        )
        # Per-stage recovery and area (RO is indexed by stage). Total area is the sum.
        for i in m.fs.RO_main.stages:
            outputs[f"RO recovery s{i}"] = m.fs.RO_main.RO[i].recovery_vol_phase[0, "Liq"]
            outputs[f"RO area s{i} (m2)"] = m.fs.RO_main.RO[i].area
    return outputs


def _resolve_targets(m, targets, treatment_train="RBAT", verbose=True):
    """Shared front-end for the grid sweep and the Monte Carlo sampler.

    ``targets`` is ``{handle_name: (min, max)}``. For each swept target this:
      * resolves the Pyomo object via ``get_sweep_handles`` and finds its group
        (inputs/decisions/costing);
      * if it is a *decision* (currently unfixed) Var, notes that fixing it for the
        run turns it into a parametric study (no longer cost-optimal in that var) and
        that it removes one degree of freedom;
      * widens the variable's bounds to cover the [min, max] range, so fixing it to an
        end-of-range value cannot conflict with a bound left over from
        ``optimize_operation`` (e.g. RO_recovery's [0.30, 0.90]).

    Returns ``(vars_by_name, n_decision_fixed, decision_names)``. (Costing Params have
    no DOF impact and no bounds, so they are passed through untouched.)
    """
    handles = dpr.get_sweep_handles(m, treatment_train)
    name_to_group = {n: g for g, d in handles.items() for n in d}

    vars_by_name = {}
    n_decision_fixed = 0          # currently-unfixed Vars the run will fix -> -1 DOF each
    decision_names = []
    for name, (lo, hi) in targets.items():
        var = dpr.resolve_sweep_handle(handles, name)
        group = name_to_group.get(name, "?")

        if hasattr(var, "is_variable_type") and var.is_variable_type():
            if not var.fixed:
                n_decision_fixed += 1
            # Widen bounds so the fixed sample value never sits outside them.
            lb, ub = var.lb, var.ub
            if lb is not None and lo < lb:
                var.setlb(lo)
                if verbose:
                    print(f"[sweep] widened lower bound of '{name}': {lb} -> {lo}")
            if ub is not None and hi > ub:
                var.setub(hi)
                if verbose:
                    print(f"[sweep] widened upper bound of '{name}': {ub} -> {hi}")

        if group == "decisions":
            decision_names.append(name)

        vars_by_name[name] = var

    return vars_by_name, n_decision_fixed, decision_names


def _check_sweep_dof(m, n_decision_fixed, decision_names, verbose=True):
    """Predict the post-sweep DOF and raise a clear error if fixing the decision
    targets would over-constrain the model (DOF < 0) instead of letting it fail later
    as opaque NaNs. Shared by the grid sweep and the Monte Carlo sampler."""
    dof_before = degrees_of_freedom(m)
    dof_after = dof_before - n_decision_fixed
    if dof_after < 0:
        raise ValueError(
            f"Run is over-constrained: fixing {n_decision_fixed} decision "
            f"variable(s) {decision_names} would drop degrees of freedom from "
            f"{dof_before} to {dof_after}. Remove a decision target, "
            f"or free a compensating variable, so DOF stays >= 0."
        )

    if verbose:
        if decision_names:
            print(
                f"[sweep] decision variables fixed for this run (parametric, "
                f"NOT cost-optimal in these): {decision_names}"
            )
        kind = "pure evaluation (optimizer has no freedom left)" if dof_after == 0 \
            else "parametric optimization"
        print(
            f"[sweep] DOF: {dof_before} -> {dof_after} after fixing "
            f"{n_decision_fixed} decision var(s); mode: {kind}"
        )

    return dof_before, dof_after


def build_sweep_params(m, sweep_config, treatment_train="RBAT", verbose=True):
    """Turn a {handle_name: (min, max, nx)} config into parameter_sweep LinearSamples,
    with degree-of-freedom safety checks. See ``_resolve_targets`` /
    ``_check_sweep_dof`` for the per-target bound widening and DOF prediction.
    """
    targets = {name: (lo, hi) for name, (lo, hi, nx) in sweep_config.items()}
    vars_by_name, n_decision_fixed, decision_names = _resolve_targets(
        m, targets, treatment_train=treatment_train, verbose=verbose
    )
    _check_sweep_dof(m, n_decision_fixed, decision_names, verbose=verbose)

    return {
        name: LinearSample(vars_by_name[name], lo, hi, nx)
        for name, (lo, hi, nx) in sweep_config.items()
    }


def run_sweep(
    state="CA",
    treatment_train="RBAT",
    sweep_config=None,
    output_filename=None,
    interpolate_nan_outputs=True,
    verbose=True,
):
    """Run a (multi-variable) sweep over DPR_flowsheet_v2.

    Parameters
    ----------
    sweep_config : dict
        ``{handle_name: (min, max, nx)}``. ``handle_name`` is any key returned by
        ``DPR_flowsheet_v2.get_sweep_handles``. Defaults to the v1 behavior
        (feed flow, 1-100 MGD, 21 points) when None.
    output_filename : str, optional
        Where to write the CSV. None -> auto-named file in ``OUTPUT_DIR``
        (``<this module>/output/``). A bare filename is placed in ``OUTPUT_DIR``;
        a path that includes a directory is used as given.
    """
    if sweep_config is None:
        sweep_config = {"feed_flow": (1 * MGD, 100 * MGD, 21)}

    # Build once, quietly; the model is reused (and warm-started) across all samples.
    m = dpr.main(state=state, treatment_train=treatment_train, verbose=False, n_stages=N_STAGES)[0]

    sweep_params = build_sweep_params(
        m, sweep_config, treatment_train=treatment_train, verbose=verbose
    )
    outputs = set_up_sensitivity(m, treatment_train=treatment_train)

    output_filename = _resolve_output_path(
        output_filename, sweep_config, state, treatment_train
    )

    global_results = parameter_sweep(
        m,
        sweep_params,
        outputs,
        csv_results_file_name=output_filename,
        optimize_function=dpr.solve,
        optimize_kwargs={"fail_flag": False},
        # Backstop for any residual non-converged (off-design) points.
        interpolate_nan_outputs=interpolate_nan_outputs,
    )
    return global_results, output_filename


# --------------------------------------------------------------------------- #
# Monte Carlo (random uniform) sampling                                         #
# --------------------------------------------------------------------------- #

# Default Monte Carlo ranges for the CA case study.
#   system capacity (feed_flow) 1-100 MGD -> 0.0438126-4.38126 m3/s
# Feed concentrations are in the model's native kg/m3 (1 kg/m3 = 1000 mg/L), so the
# mg/L ranges below are converted:
#   TOC  7-15 mg/L  -> 0.007-0.015 kg/m3
#   TDS  500-1000 mg/L -> 0.5-1.0  kg/m3
# brine_disposal_cost is already in $/m3.
#
# ORDER MATTERS: feed_flow is listed FIRST on purpose. parameter_sweep sorts random
# samples by the first swept variable (see _create_global_combo_array), so putting the
# most solver-sensitive variable (system capacity, a 100x range) first makes the
# warm-start path monotonic in that dimension -- the same smooth path the 1-100 MGD
# grid sweep used to converge 21/21. The narrow-range water-quality / cost variables
# still jump randomly between samples but barely perturb the warm start.
DEFAULT_MC_CONFIG = {
    "feed_flow": (1 * MGD, 100 * MGD),
    "feed_toc": (0.007, 0.015),
    "feed_tds": (0.5, 1.0),
    "brine_disposal_cost": (0.05, 0.66),
}

# Monte Carlo ranges for the CBAT (non-RO) train. CBAT has no RO/brine, so the RBAT-only
# brine_disposal_cost handle is dropped (it does not exist in get_sweep_handles for CBAT).
# feed_tds is deliberately NOT swept here so it stays pinned at its TERTIARY default of
# 0.5 kg/m3 = 500 mg/L. Only system capacity and feed TOC vary:
#   system capacity (feed_flow) 1-100 MGD
#   TOC  7-15 mg/L  -> 0.007-0.015 kg/m3
CBAT_MC_CONFIG = {
    "feed_flow": (1 * MGD, 100 * MGD),
    "feed_toc": (0.007, 0.015),
}

# Monte Carlo ranges for the IPR (UF->RO->UV/AOP) train, which treats SECONDARY effluent.
# Relative to the RBAT/tertiary default: secondary effluent carries more organics, so TOC is
# swept 10-15 mg/L (0.010-0.015 kg/m3) instead of 7-15; and feed_tds is NOT swept -- it stays
# pinned at the SECONDARY default of 1.0 kg/m3 = 1000 mg/L (secondary effluent salinity). System
# capacity and brine disposal cost vary as before.
#   system capacity (feed_flow) 1-100 MGD
#   TOC  10-15 mg/L -> 0.010-0.015 kg/m3
#   brine_disposal_cost 0.05-0.66 $/m3
IPR_MC_CONFIG = {
    "feed_flow": (1 * MGD, 100 * MGD),
    "feed_toc": (0.010, 0.015),
    "brine_disposal_cost": (0.05, 0.66),
}


def build_mc_params(
    m, mc_config, num_samples, treatment_train="RBAT", verbose=True, predetermined=None
):
    """Turn a {handle_name: (min, max)} config into parameter_sweep samples for a Monte
    Carlo run. parameter_sweep stacks the per-handle draws column-wise (one row per
    sample), not as a Cartesian grid. Shares the bound widening and DOF safety checks
    with the grid sweep (see ``_resolve_targets`` / ``_check_sweep_dof``).

    ``predetermined`` (optional): a ``{handle_name: array}`` of pre-drawn sample values.
    When given, each handle uses those exact values (via ``PredeterminedRandomSample``)
    instead of drawing fresh ones -- this is how the multi-state run feeds the SAME
    random inputs to every state so only the outputs (LCOW, ...) differ. When None,
    each handle is drawn independently ``num_samples`` times from a uniform distribution
    (numpy.random.uniform).
    """
    vars_by_name, n_decision_fixed, decision_names = _resolve_targets(
        m, mc_config, treatment_train=treatment_train, verbose=verbose
    )
    _check_sweep_dof(m, n_decision_fixed, decision_names, verbose=verbose)

    if predetermined is not None:
        return {
            name: PredeterminedRandomSample(vars_by_name[name], predetermined[name])
            for name in mc_config
        }
    return {
        name: UniformSample(vars_by_name[name], lo, hi, num_samples)
        for name, (lo, hi) in mc_config.items()
    }


# Multi-start set for the rescue: each tiny relative feed-flow shift seeds a slightly
# different initialization, so trying several of them recovers a point that errors / hits a
# "locally infeasible" termination at one start but converges at another (a numerical
# knife-edge, not true infeasibility). The first converging start is taken. Both signs are
# tried, up to 1e-2 relative, so the recovered outputs differ from the true point by far less
# than any meaningful resolution; they are recorded against the original (unperturbed) CSV
# inputs. 0.0 (the exact value) is tried first.
_RESCUE_FLOW_PERTURBATIONS = (0.0, 1e-4, -1e-4, 5e-4, -5e-4, 1e-3, -1e-3,
                              3e-3, -3e-3, 5e-3, -5e-3, 1e-2, -1e-2)


def _solve_single_point(
    state, treatment_train, point_values, name_to_group, effluent_type="TERTIARY"
):
    """Build a FRESH model at one input point and solve it the way ``main`` does, so
    its scaling is computed at that point instead of being warm-started (with a stale,
    build-time scaling) from a distant sample. This is what the rescue pass uses to
    recover the high-capacity / high-salinity samples the warm-started sweep drops.

    The exact sampled value is tried first; if it hits a solver error / non-convergence
    (a numerical knife-edge), the feed flow is nudged by a tiny relative amount and the
    build is retried (see ``_RESCUE_FLOW_PERTURBATIONS``).

    ``point_values`` maps ``get_sweep_handles`` names to the sampled values;
    ``name_to_group`` says which group each name is in, so values are injected at the
    same stage parameter_sweep would: inputs BEFORE scaling, costing AFTER costing is
    built, decisions AFTER ``optimize_operation`` unfixes them. Returns
    ``(m, converged)``; ``m`` is the last attempt's model (None if the first build
    raised before constructing one).
    """
    m = None
    for eps in _RESCUE_FLOW_PERTURBATIONS:
        # Nudge only feed flow (the dominant, solver-sensitive dimension) off the
        # knife-edge; eps == 0.0 is the exact value.
        pv = point_values
        if eps and "feed_flow" in point_values:
            pv = dict(point_values)
            pv["feed_flow"] = point_values["feed_flow"] * (1 + eps)

        try:
            LRV_req, _details, state, treatment_train, effluent_type, solute_list = (
                dpr.DPR_initial_setting(
                    state=state,
                    treatment_train=treatment_train,
                    effluent_type=effluent_type,
                )
            )
            build = dpr.build_nonRO if treatment_train == "CBAT" else dpr.build_RO
            build_kw = dict(
                solute_list=solute_list,
                state=state,
                effluent_type=effluent_type,
                treatment_train=treatment_train,
            )
            if treatment_train != "IPR":  # IPR carries no ozone/Cl2 LRV requirements
                build_kw.update(
                    LRVO3_req=LRV_req["ozone_crypto_lrv_required"],
                    LRVClvirus_req=LRV_req["cl2_virus_lrv_required"],
                    LRVClgiardia_req=LRV_req["cl2_giardia_lrv_required"],
                )
            if treatment_train in ("RBAT", "IPR"):  # RO trains take a stage count
                build_kw["n_stages"] = N_STAGES
            m = build(**build_kw)
            dpr.set_operating_conditions(
                m, solute_list=solute_list, treatment_train=treatment_train, state=state
            )

            # Inputs: apply BEFORE scaling so scale_system reads the right magnitudes.
            handles = dpr.get_sweep_handles(m, treatment_train)  # costing empty here
            feed_touched = False
            for name, val in pv.items():
                if name_to_group.get(name) == "inputs":
                    dpr.resolve_sweep_handle(handles, name).fix(val)
                    feed_touched = feed_touched or name.startswith("feed_")
            if feed_touched:
                dpr.solve(m.fs.feed)  # refresh feed flow_mass_comp from new flow/conc

            dpr.scale_system(m, treatment_train=treatment_train)
            dpr.initialize_system(m, treatment_train=treatment_train)
            dpr.solve(m)

            dpr.add_costing(m, treatment_train=treatment_train)
            dpr.initialize_costing(m, treatment_train=treatment_train)

            # Costing values exist only now.
            handles = dpr.get_sweep_handles(m, treatment_train)
            for name, val in pv.items():
                if name_to_group.get(name) == "costing":
                    obj = dpr.resolve_sweep_handle(handles, name)
                    obj.fix(val) if obj.is_variable_type() else obj.set_value(val)

            dpr.optimize_operation(
                m,
                state=state,
                effluent_type=effluent_type,
                treatment_train=treatment_train,
            )

            # Decisions: optimize_operation just unfixed them; re-fix any swept ones to
            # the sample value (no-op for the default MC, which sweeps no decisions).
            for name, val in pv.items():
                if name_to_group.get(name) == "decisions":
                    dpr.resolve_sweep_handle(handles, name).fix(val)

            converged = check_optimal_termination(dpr.solve(m))
        except Exception:  # a solver error / build blow-up -> try the next nudge
            converged = False

        if converged:
            # Multi-start: return the FIRST perturbation that converges (the wider both-sign
            # set above gives several restarts, so a point that errors / hits a "locally
            # infeasible" termination at one start is recovered from another). Any worse local
            # optimum a rescued point lands in is cleaned up afterwards by repair_mc_outliers
            # (which re-solves above-envelope points), keeping this pass fast.
            return m, True
    return m, False


def _rescue_failed_in_csv(
    csv_path,
    state,
    treatment_train,
    mc_config,
    name_to_group,
    effluent_type="TERTIARY",
    verbose=True,
):
    """Re-solve every non-converged (LCOW NaN) row in ``csv_path`` by rebuilding a
    fresh model at that row's sampled inputs (so scaling is computed at the point) and
    write the recovered outputs back in place. Returns ``(n_failed, n_rescued)``."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    # parameter_sweep prefixes the first header with "# "; strip for lookup, restore
    # on write so the CSV format is unchanged.
    clean = {c: c.lstrip("# ").strip() for c in df.columns}
    df.rename(columns=clean, inplace=True)

    failed = list(df.index[df["LCOW"].isna()])
    input_names = list(mc_config.keys())
    n_rescued = 0
    for n, i in enumerate(failed, 1):
        point = {name: float(df.at[i, name]) for name in input_names}
        try:
            m, ok = _solve_single_point(
                state, treatment_train, point, name_to_group, effluent_type
            )
        except Exception as exc:  # a build/init blow-up shouldn't abort the rescue
            ok = False
            if verbose:
                print(f"[rescue] {n}/{len(failed)} row {i}: error {exc!r}")
        if ok:
            for k, obj in set_up_sensitivity(
                m, treatment_train=treatment_train
            ).items():
                if k in df.columns:
                    df.at[i, k] = value(obj)
            n_rescued += 1
        if verbose:
            flow_mgd = point.get("feed_flow", float("nan")) / MGD
            print(
                f"[rescue] {n}/{len(failed)} row {i} (flow={flow_mgd:.1f} MGD): "
                f"{'recovered' if ok else 'still failed'}"
            )

    df.rename(columns={v: k for k, v in clean.items()}, inplace=True)
    df.to_csv(csv_path, index=False)
    return len(failed), n_rescued


def run_monte_carlo(
    state="CA",
    treatment_train="RBAT",
    effluent_type="TERTIARY",
    mc_config=None,
    num_samples=1000,
    output_filename=None,
    seed=None,
    interpolate_nan_outputs=False,
    rescue=True,
    plot=True,
    verbose=True,
    predetermined=None,
):
    """Run a Monte Carlo (random uniform) sampling over DPR_flowsheet_v2.

    Unlike ``run_sweep`` (which sweeps a deterministic Cartesian grid), this draws
    ``num_samples`` independent uniform random samples for each target and solves the
    warm-started model at each one, writing inputs + outputs to a CSV. By default it
    then runs a rescue pass and a plot (see below).

    Parameters
    ----------
    mc_config : dict, optional
        ``{handle_name: (min, max)}``. ``handle_name`` is any key returned by
        ``DPR_flowsheet_v2.get_sweep_handles``. Defaults to ``DEFAULT_MC_CONFIG``
        (CA case study: system capacity 1-100 MGD, feed TOC 7-15 mg/L, feed TDS
        500-1000 mg/L, brine disposal cost 0.05-0.66 $/m3).
    num_samples : int
        Number of random samples to draw (default 1000).
    seed : int, optional
        Seed for ``numpy.random`` so a run is reproducible. None -> non-deterministic.
    interpolate_nan_outputs : bool
        Default False: non-converged samples are kept as NaN in the CSV (so it is
        clear which input combinations failed) rather than interpolated -- grid-based
        interpolation is not meaningful for scattered random points.
    rescue : bool
        Default True: after the sweep, re-solve every non-converged sample by
        rebuilding a fresh model at its inputs (scaling computed at the point) and
        write the recovered outputs back into the CSV. The warm-started sweep tends to
        drop the numerically hardest corner (high capacity + high salinity), where the
        build-time scaling has gone stale; those points are not infeasible, they just
        need a point-local scaling, which this provides.
    plot : bool
        Default True: after the (rescued) CSV is written, save a system-capacity vs
        LCOW scatter PNG next to it via ``plot_monte_carlo.plot_flow_vs_lcow``.
    output_filename : str, optional
        Where to write the CSV. None -> auto-named file in ``OUTPUT_DIR``. A bare
        filename is placed in ``OUTPUT_DIR``; a path with a directory is used as given.
    predetermined : dict, optional
        ``{handle_name: array}`` of pre-drawn sample values to use instead of drawing
        fresh ones. Used by ``run_monte_carlo_multistate`` to feed every state the same
        random inputs (so only the outputs differ). When given, ``seed`` / ``num_samples``
        are not used for drawing.

    Returns
    -------
    (global_results, output_filename, png_path)
        ``png_path`` is None when ``plot=False``.
    """
    if mc_config is None:
        mc_config = dict(IPR_MC_CONFIG if treatment_train == "IPR" else DEFAULT_MC_CONFIG)
    if seed is not None:
        np.random.seed(seed)

    # Build once, quietly; the model is reused (and warm-started) across all samples.
    m = dpr.main(state=state, treatment_train=treatment_train, effluent_type=effluent_type,
                 verbose=False, n_stages=N_STAGES)[0]

    sweep_params = build_mc_params(
        m,
        mc_config,
        num_samples,
        treatment_train=treatment_train,
        verbose=verbose,
        predetermined=predetermined,
    )
    outputs = set_up_sensitivity(m, treatment_train=treatment_train)

    # Group of each handle (for the rescue pass to inject values at the right stage).
    handles = dpr.get_sweep_handles(m, treatment_train=treatment_train)
    name_to_group = {n: g for g, d in handles.items() for n in d}

    if output_filename is None:
        tag = "_".join(mc_config.keys())
        output_filename = f"mc_v2_{state}_{treatment_train}_n{num_samples}_{tag}.csv"
    if os.path.dirname(output_filename) == "":
        output_filename = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    global_results = parameter_sweep(
        m,
        sweep_params,
        outputs,
        csv_results_file_name=output_filename,
        optimize_function=dpr.solve,
        optimize_kwargs={"fail_flag": False},
        interpolate_nan_outputs=interpolate_nan_outputs,
    )

    if rescue:
        n_failed, n_rescued = _rescue_failed_in_csv(
            output_filename,
            state,
            treatment_train,
            mc_config,
            name_to_group,
            effluent_type=effluent_type,
            verbose=verbose,
        )
        if verbose:
            print(
                f"[rescue] recovered {n_rescued}/{n_failed} non-converged sample(s) "
                f"via fresh-build re-solve"
            )

    png_path = None
    if plot:
        from watertap.flowsheets.potable_reuse.plot_monte_carlo import (
            plot_flow_vs_lcow,
        )

        png_path, n_total, n_fail_after = plot_flow_vs_lcow(
            output_filename, label=f"{state} / {treatment_train}"
        )
        if verbose:
            print(
                f"[plot] saved {png_path} "
                f"({n_total - n_fail_after}/{n_total} points plotted)"
            )

    return global_results, output_filename, png_path


def run_monte_carlo_multistate(
    states=("CA", "CO", "FL"),
    treatment_train="RBAT",
    effluent_type="TERTIARY",
    mc_config=None,
    num_samples=1000,
    seed=0,
    output_dir=None,
    rescue=True,
    plot=True,
    verbose=True,
):
    """Run the Monte Carlo for several states on the SAME random input samples.

    The sample values (feed flow, TOC, TDS, brine disposal cost) are drawn ONCE and
    reused for every state, so the per-state CSVs share identical input columns row for
    row -- only the outputs (LCOW, CAPEX, ...) differ, because the state changes only
    the regulatory LRV requirements (disinfection dosing), not the influent. Produces
    one CSV and one PNG per state; the PNGs share a common LCOW y-axis (the global
    min/max across all states) so they can be compared side by side.

    Parameters
    ----------
    states : iterable of str
        States to run (default CA, CO, FL). Each must be supported by
        ``DPR_flowsheet_v2.DPR_initial_setting`` (CA / CO / FL / AZ).
    output_dir : str, optional
        Directory for the per-state CSV/PNG. Default ``OUTPUT_DIR/monte_carlo``.

    Returns
    -------
    dict
        ``{state: (csv_path, png_path)}``.
    """
    if mc_config is None:
        mc_config = dict(IPR_MC_CONFIG if treatment_train == "IPR" else DEFAULT_MC_CONFIG)
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "monte_carlo")
    os.makedirs(output_dir, exist_ok=True)

    # Draw the sample arrays ONCE; every state reuses these exact values.
    if seed is not None:
        np.random.seed(seed)
    predetermined = {
        name: np.random.uniform(lo, hi, num_samples)
        for name, (lo, hi) in mc_config.items()
    }

    # Run + rescue every state first (plot deferred: the shared y-axis needs the LCOW
    # range across all states, which we only know once they have all run).
    csv_by_state = {}
    for state in states:
        if verbose:
            print(f"\n=== Monte Carlo: state={state}, train={treatment_train} ===")
        csv = os.path.join(
            output_dir, f"mc_v2_{state}_{treatment_train}_{TAG}_n{num_samples}.csv"
        )
        _gr, csv_path, _png = run_monte_carlo(
            state=state,
            treatment_train=treatment_train,
            effluent_type=effluent_type,
            mc_config=mc_config,
            num_samples=num_samples,
            output_filename=csv,
            seed=None,  # samples are predetermined; no drawing here
            rescue=rescue,
            plot=False,  # deferred to the shared-axis pass below
            verbose=verbose,
            predetermined=predetermined,
        )
        csv_by_state[state] = csv_path

    results = {state: (csv, None) for state, csv in csv_by_state.items()}
    if plot:
        import pandas as pd
        from watertap.flowsheets.potable_reuse.plot_monte_carlo import plot_flow_vs_lcow

        # Common LCOW axis (global min/max + a small margin) across all states.
        lo, hi = float("inf"), float("-inf")
        for csv in csv_by_state.values():
            col = pd.read_csv(csv)["LCOW"].dropna()
            lo, hi = min(lo, col.min()), max(hi, col.max())
        margin = 0.03 * (hi - lo)
        ylim = (lo - margin, hi + margin)

        for state, csv in csv_by_state.items():
            png, _n, _f = plot_flow_vs_lcow(
                csv, label=f"{state} / {treatment_train}", ylim=ylim
            )
            results[state] = (csv, png)
            if verbose:
                print(f"[plot] saved {png}")

    return results


if __name__ == "__main__":
    # SPIRAL RO + added RO constraints: RBAT Monte Carlo for CA / CO / FL on the SAME 1000
    # random input samples (feed_flow 1-100 MGD, feed TOC 7-15 mg/L, feed TDS 0.5-1.0 kg/m3,
    # brine_disposal_cost 0.05-0.66 -- the default RBAT config). Files tagged with TAG so they
    # do not overwrite older results: mc_v2_<STATE>_RBAT_v3_n<N>_n1000.{csv,png}.
    run_monte_carlo_multistate(
        states=("CA", "CO", "FL"),
        treatment_train="RBAT",
        num_samples=1000,
        seed=0,
    )

    # Default grid sweep (1-D feed-flow, matches DPR_sweep_v1 scope): uncomment to run.
    # run_sweep()

    # Example 2-D grid sweep (feed flow x feed TDS): uncomment to run.
    # run_sweep(train="RBAT", num_samples=1000, seed=0)
    #
    #     # Default grid sweep (1-D feed-flow, matches DPR_sweep_v1 scope): uncomment to run.
    #     # run_sweep()
    #
    #     # Example 2-D grid sweep (feed flow x feed TDS): uncomment to run.
    #     # run_sweep(
    #     sweep_config={
    #         "feed_flow": (1 * MGD, 100 * MGD, 21),
    #         "feed_tds": (0.5, 2.0, 5),
    #     }
    # )
