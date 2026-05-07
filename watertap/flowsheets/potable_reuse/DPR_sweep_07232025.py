"""
Non-RO DPR (O3/BAF/UF/Carbon_Adsorption/UV_AOP/Chlorination) and RO DPR (Cl/UF/RO/UV_AOP)
This module contains a zero-order representation of each unit except RO
"""
from pyomo.environ import value, Block
from pyomo.util.calc_var_value import value
from pyomo.environ import units as pyunits
from parameter_sweep import LinearSample, parameter_sweep
import watertap.flowsheets.nonRO_DPR.DPR_flowsheet_ongoing2_10 as dpr_flowsheet
from idaes.core import UnitModelBlockData

def set_up_sensitivity(m):
    outputs = {}
    optimize_kwargs = {"fail_flag": False}
    opt_function = dpr_flowsheet.solve

    outputs["LCOT_wo_revenue"] = m.fs.LCOT_wo_revenue
    outputs["Total CAPEX"] = m.fs.total_capital_cost
    outputs["Total OPEX"] = m.fs.total_operating_cost
    outputs["Total fixed OPEX"] = m.fs.zo_costing.total_fixed_operating_cost
    outputs["Total fixed OPEX"] = m.fs.zo_costing.total_variable_operating_cost
    outputs["Opex_fraction"] = 100 * m.fs.total_operating_cost / (m.fs.total_capital_cost * m.fs.zo_costing.capital_recovery_factor + m.fs.total_operating_cost)
    outputs["SEC"] = m.fs.specific_energy_intensity

    outputs["O3:TOC"] = m.fs.non_RO.Ozone.O3toTOC[0]
    outputs["HRT_ozone"] = m.fs.non_RO.Ozone.contact_time[0]
    outputs["EBCT_BAC"] = m.fs.non_RO.BAF.EBCT[0]
    outputs["EBCT_GAC"] = m.fs.non_RO.GAC.EBCT[0]
    outputs["Replacement_Frequency"] = m.fs.non_RO.GAC.replacement_frequency[0]
    outputs["UV dose"] = m.fs.non_RO.UV_AOP.uv_dose[0]
    outputs["H2O2 dose"] = m.fs.non_RO.UV_AOP.hydrogen_peroxide_dose[0]
    outputs["UV dose"] = m.fs.non_RO.UV_AOP.uv_dose[0]
    outputs["H2O2 dose"] = m.fs.non_RO.UV_AOP.hydrogen_peroxide_dose[0]
    outputs["Cl dose"] = m.fs.non_RO.Cl.chlorine_dose[0]
    outputs["HRT_Cl"] = m.fs.non_RO.Cl.contact_time[0]

    # ✅ Unit-specific CAPEX/OPEX from registered costing blocks
    for costing_block in m.fs.zo_costing._registered_unit_costing:
        unit_name = costing_block.parent_block().local_name
        outputs[f"{unit_name} CAPEX"] = costing_block.capital_cost * m.fs.zo_costing.total_investment_factor

    flow_types = m.fs.zo_costing.used_flows
    util = m.fs.zo_costing.utilization_factor
    flow_cost_params = {
        ft: getattr(m.fs.zo_costing, f"{ft}_cost") for ft in flow_types
    }

    unit_blocks = {
        name: block for name, block in m.fs.non_RO.component_map(Block).items()
        if isinstance(block, UnitModelBlockData)
    }

    for unit_name, unit_block in unit_blocks.items():
        unit_total_cost = 0.0

        for flow_type in flow_types:
            if hasattr(unit_block, flow_type):
                var = getattr(unit_block, flow_type)
                v = var[0] if var.is_indexed() else var
                flow_cost = v * flow_cost_params[flow_type] * util
                # Add to outputs with formatted key
                outputs[f"{unit_name} {flow_type} cost"] = flow_cost
                unit_total_cost += flow_cost

        outputs[f"{unit_name} flow cost total"] = unit_total_cost
    return outputs, optimize_kwargs, opt_function

def run_analysis(case_num=1, nx=11, interpolate_nan_outputs=True, output_filename=None):

    if output_filename is None:
        output_filename = "sensitivity_" + str(case_num) + ".csv"

    # when from the command line
    case_num = int(case_num)
    nx = int(nx)
    interpolate_nan_outputs = bool(interpolate_nan_outputs)

    # # select flowsheet
    # m = dpr_flowsheet.main(state="FL")[0]

    # # set up sensitivities
    # outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)

    # choose parameter sweep from case structure
    sweep_params = {}

    # Convert MGD to m³/s
    flow_m3s = pyunits.convert(1 * pyunits.Mgallons / pyunits.day,
                               to_units=pyunits.m ** 3 / pyunits.s)

    if case_num == 1:
        m = dpr_flowsheet.main(state="CO", treatment_train="CBAT")[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)

        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], value(flow_m3s)*1, value(flow_m3s)*100, nx
        )
    elif case_num == 2:
        m = dpr_flowsheet.main(state="FL", treatment_train="CBAT")[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], value(flow_m3s)*(1+(100-1)/(nx-1)), value(flow_m3s)*100, nx-1
        )
    elif case_num == 3:
        m = dpr_flowsheet.main(state="CO", treatment_train="CBAT")[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], value(flow_m3s)*1, value(flow_m3s)*100, nx
        )
    elif case_num == 4:
        m = dpr_flowsheet.main(state="FL")[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)
        m.fs.feed.conc_mass_comp[0, 'toc'].unfix()
        sweep_params["toc conc"] = LinearSample(
            m.fs.feed.conc_mass_comp[0, 'toc'], value(flow_m3s)*(1+(100-1)/(nx-1)), value(flow_m3s)*1, nx
        )
    elif case_num == 5:
        m.fs.feed.conc_mass_comp[0, 'toc'].unfix()
        sweep_params["toc conc"] = LinearSample(
            m.fs.feed.conc_mass_comp[0, 'toc'], 0.007, 0.015, nx
        )
    else:
        raise ValueError("case_num = %d not recognized." % (case_num))

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

    # global_results = parameter_sweep(
    #     n,
    #     sweep_params,
    #     outputs,
    #     csv_results_file_name=output_filename,
    #     optimize_function=opt_function,
    #     optimize_kwargs=optimize_kwargs,
    #     interpolate_nan_outputs=interpolate_nan_outputs,
    # )

    #
    # import os
    # abs_path = os.path.abspath(output_filename)
    # print(f"\n✅ Results saved to: {abs_path}\n")

    return global_results, sweep_params, m

if __name__ == "__main__":
    results, sweep_params, m = run_analysis()