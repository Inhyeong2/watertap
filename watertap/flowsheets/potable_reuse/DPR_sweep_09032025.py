"""
Non-RO DPR (O3/BAF/UF/Carbon_Adsorption/UV_AOP/Chlorination) and RO DPR (Cl/UF/RO/UV_AOP)
This module contains a zero-order representation of each unit except RO
"""
from pyomo.environ import value, Block
from pyomo.util.calc_var_value import value
from pyomo.environ import units as pyunits
from parameter_sweep import LinearSample, parameter_sweep
import watertap.flowsheets.nonRO_DPR.DPR_flowsheet_ongoing2_11 as dpr_flowsheet
from idaes.core import UnitModelBlockData

def set_up_sensitivity(m, treatment_train=None):
    outputs = {}
    optimize_kwargs = {"fail_flag": False}
    opt_function = dpr_flowsheet.solve

    outputs["LCOW"] = m.fs.LCOW
    outputs["Total CAPEX"] = m.fs.total_capital_cost
    outputs["Total OPEX"] = m.fs.total_operating_cost
    outputs["Opex_fraction"] = 100 * m.fs.total_operating_cost / (m.fs.total_capital_cost * m.fs.zo_costing.capital_recovery_factor + m.fs.total_operating_cost)
    outputs["SEC"] = m.fs.specific_energy_intensity



    return outputs, optimize_kwargs, opt_function

def run_analysis(case_num=1, nx=21, interpolate_nan_outputs=True, output_filename=None):
    case_num = int(case_num)
    nx = int(nx)
    interpolate_nan_outputs = bool(interpolate_nan_outputs)
    sweep_params = {}

    if case_num == 1:
        state = "CO"
        treatment_train = "CBAT"
        description = "Default"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        # res = dpr_flowsheet.cost_output(m, treatment_train=treatment_train)
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1*0.0438, 100*0.0438, nx
        )

    elif case_num == 2:
        state = "FL"
        treatment_train = "CBAT"
        description = "Default"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1*0.0438, 100*0.0438, nx
        )


    elif case_num == 3:
        state = "CO"
        treatment_train = "CBAT"
        description = "TOC_sweep"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)
        m.fs.feed.conc_mass_comp[0, 'toc'].unfix()
        sweep_params["toc conc"] = LinearSample(
            m.fs.feed.conc_mass_comp[0, 'toc'], 0.007, 0.015, nx
        )

    elif case_num == 4:
        state = "FL"
        treatment_train = "CBAT"
        description = "TOC_sweep"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)
        m.fs.feed.conc_mass_comp[0, 'toc'].unfix()
        sweep_params["toc conc"] = LinearSample(
            m.fs.feed.conc_mass_comp[0, 'toc'], 0.007, 0.015, nx
        )

    elif case_num == 5:
        state = "CO"
        treatment_train = "RBAT"
        description = "Default"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1*0.0438, 100*0.0438, nx
        )

    elif case_num == 6:
        state = "FL"
        treatment_train = "RBAT"
        description = "Default"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1*0.0438, 100*0.0438, nx
        )

    elif case_num == 7:
        state = "CA"
        treatment_train = "RBAT"
        description = "Default"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1*0.0438, 100*0.0438, nx
        )

    else:
        raise ValueError("case_num = %d not recognized." % (case_num))

    if output_filename is None:
        output_filename = f"sensitivity_{state}_{treatment_train}_{description}.csv"

    # run sweep
    global_results = parameter_sweep(
        m,
        sweep_params,
        outputs,
        csv_results_file_name=output_filename,
        optimize_function=opt_function,
        optimize_kwargs=optimize_kwargs,
        interpolate_nan_outputs=interpolate_nan_outputs,
    )

    return global_results, sweep_params, m

if __name__ == "__main__":
    results, sweep_params, m = run_analysis()