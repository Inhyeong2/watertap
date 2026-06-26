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

from parameter_sweep import LinearSample, parameter_sweep
from idaes.core.util.model_statistics import degrees_of_freedom
import watertap.flowsheets.potable_reuse.DPR_flowsheet_v2 as dpr

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
        outputs["RO recovery"] = m.fs.RO_main.RO.recovery_vol_phase[0, "Liq"]
        outputs["RO pressure (Pa)"] = (
            m.fs.RO_main.pump.control_volume.properties_out[0].pressure
        )
        outputs["RO area (m2)"] = m.fs.RO_main.RO.area
    return outputs


def build_sweep_params(m, sweep_config, treatment_train="RBAT", verbose=True):
    """Turn a {handle_name: (min, max, nx)} config into parameter_sweep LinearSamples,
    with degree-of-freedom safety checks (item 3).

    For each swept target this:
      * finds which get_sweep_handles group it belongs to (inputs/decisions/costing);
      * if it is a *decision* (currently unfixed), notes that fixing it for the sweep
        turns the run into a parametric study (no longer cost-optimal in that var) and
        that it removes one degree of freedom;
      * widens the variable's bounds to cover the sweep range, so fixing it to an
        end-of-range value cannot conflict with a bound left over from
        ``optimize_operation`` (e.g. RO_recovery's [0.30, 0.90]);
    and then predicts the post-sweep DOF, raising a clear error instead of letting an
    over-constrained sweep (DOF < 0) fail later as opaque NaNs.
    """
    handles = dpr.get_sweep_handles(m, treatment_train)
    name_to_group = {n: g for g, d in handles.items() for n in d}

    sweep_params = {}
    n_decision_fixed = 0          # currently-unfixed Vars the sweep will fix -> -1 DOF each
    decision_names = []
    for name, (lo, hi, nx) in sweep_config.items():
        var = dpr.resolve_sweep_handle(handles, name)
        group = name_to_group.get(name, "?")

        # Variable-type targets: handle fixing/bounds. (Costing Params have no DOF
        # impact and no bounds, so skip them.)
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

        sweep_params[name] = LinearSample(var, lo, hi, nx)

    dof_before = degrees_of_freedom(m)
    dof_after = dof_before - n_decision_fixed
    if dof_after < 0:
        raise ValueError(
            f"Sweep is over-constrained: fixing {n_decision_fixed} decision "
            f"variable(s) {decision_names} would drop degrees of freedom from "
            f"{dof_before} to {dof_after}. Remove a decision target from the sweep, "
            f"or free a compensating variable, so DOF stays >= 0."
        )

    if verbose:
        if decision_names:
            print(
                f"[sweep] decision variables fixed for this sweep (parametric, "
                f"NOT cost-optimal in these): {decision_names}"
            )
        kind = "pure evaluation (optimizer has no freedom left)" if dof_after == 0 \
            else "parametric optimization"
        print(
            f"[sweep] DOF: {dof_before} -> {dof_after} after fixing "
            f"{n_decision_fixed} decision var(s); mode: {kind}"
        )

    return sweep_params


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
    m = dpr.main(state=state, treatment_train=treatment_train, verbose=False)[0]

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


if __name__ == "__main__":
    # Default: 1-D feed-flow sweep (matches DPR_sweep_v1 scope).
    run_sweep()

    # Example 2-D sweep (feed flow x feed TDS): uncomment to run.
    # run_sweep(
    #     sweep_config={
    #         "feed_flow": (1 * MGD, 100 * MGD, 21),
    #         "feed_tds": (0.5, 2.0, 5),
    #     }
    # )
