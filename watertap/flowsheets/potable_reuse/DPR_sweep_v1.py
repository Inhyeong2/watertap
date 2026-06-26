"""
Warm-start parameter sweep for DPR_flowsheet_v1 over feed flow (system capacity).

Builds the flowsheet once, then sweeps feed flow from flow_min to flow_max using
WaterTAP's parameter_sweep. The model object is reused across samples, so each
sample is warm-started from the previous (converged) point -- i.e. flow
continuation -- which is far more robust than cold-starting main() per flow.
"""
from parameter_sweep import LinearSample, parameter_sweep
import watertap.flowsheets.potable_reuse.DPR_flowsheet_v1 as dpr


def set_up_sensitivity(m):
    outputs = {
        "LCOW": m.fs.LCOW,
        "Total CAPEX": m.fs.total_capital_cost,
        "Total OPEX": m.fs.total_operating_cost,
        "SEC": m.fs.specific_energy_intensity,
    }
    optimize_kwargs = {"fail_flag": False}
    opt_function = dpr.solve
    return outputs, optimize_kwargs, opt_function


def run_sweep(state="CA", treatment_train="RBAT", nx=21,
              flow_min_mgd=1, flow_max_mgd=100, output_filename=None,
              interpolate_nan_outputs=True):
    MGD = 0.0438126  # m3/s per MGD
    m = dpr.main(state=state, treatment_train=treatment_train)[0]
    outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)

    m.fs.feed.flow_vol[0].unfix()
    sweep_params = {
        "system_capacity": LinearSample(
            m.fs.feed.flow_vol[0], flow_min_mgd * MGD, flow_max_mgd * MGD, nx
        )
    }

    if output_filename is None:
        output_filename = f"sweep_v1_{state}_{treatment_train}.csv"

    global_results = parameter_sweep(
        m,
        sweep_params,
        outputs,
        csv_results_file_name=output_filename,
        optimize_function=opt_function,
        optimize_kwargs=optimize_kwargs,
        # A few off-design flows still fail the RO optimization even with
        # warm-starting (inherent numerical fragility of the RO formulation);
        # interpolate to fill those gaps into a continuous 1-100 MGD curve.
        interpolate_nan_outputs=interpolate_nan_outputs,
    )
    return global_results, output_filename


if __name__ == "__main__":
    run_sweep()
