"""
Non-RO DPR (O3/BAF/UF/Carbon_Adsorption/UV_AOP/Chlorination) and RO DPR (Cl/UF/RO/UV_AOP)
This module contains a zero-order representation of each unit except RO
"""
from pyomo.environ import value, Block
from pyomo.util.calc_var_value import value
from pyomo.environ import units as pyunits
from parameter_sweep import LinearSample, parameter_sweep
import watertap.flowsheets.nonRO_DPR.DPR_flowsheet_ongoing2_12 as dpr_flowsheet
from idaes.core import UnitModelBlockData

def set_up_sensitivity(m, treatment_train=None):
    outputs = {}
    optimize_kwargs = {"fail_flag": False}
    opt_function = dpr_flowsheet.solve

    flow_types = m.fs.zo_costing.used_flows
    util_zo = m.fs.zo_costing.utilization_factor
    mlc_zo = m.fs.zo_costing.maintenance_labor_chemical_factor
    tif_zo = m.fs.zo_costing.total_investment_factor
    flow_cost_params = {ft: getattr(m.fs.zo_costing, f"{ft}_cost") for ft in flow_types}

    unit_blocks_zo = {
        name: blk for name, blk in m.fs.non_RO.component_map(Block).items()
        if isinstance(blk, UnitModelBlockData)
    }

    capex_blocks_zo = {
        cb.parent_block().local_name: cb for cb in m.fs.zo_costing._registered_unit_costing
    }

    results = {}
    zero_usd_per_year = 0 * (pyunits.MUSD_2020 / pyunits.year)

    for unit_name, unit_block in unit_blocks_zo.items():
        if unit_name not in capex_blocks_zo:
            continue

        cb = capex_blocks_zo[unit_name]
        capex = pyunits.convert(cb.capital_cost * tif_zo * m.fs.zo_costing.capital_recovery_factor,
                                to_units=pyunits.MUSD_2020 / pyunits.year)  # stays with units

        # Flow OPEX (electricity / chemicals / etc. defined as per-flow costs)
        flow_total = zero_usd_per_year
        for ft in flow_types:
            if hasattr(unit_block, ft):
                var = getattr(unit_block, ft)
                v = var[0] if var.is_indexed() else var
                flow_cost_val = pyunits.convert(
                    v * flow_cost_params[ft] * util_zo,
                    to_units=pyunits.MUSD_2020 / pyunits.year
                )
                print(f"  {ft:<15} : {value(flow_cost_val):,.3f} {pyunits.get_units(flow_cost_val)}")
                flow_total += flow_cost_val

        mlc_cost = pyunits.convert(mlc_zo * capex / m.fs.zo_costing.capital_recovery_factor / tif_zo,
                                   to_units=pyunits.MUSD_2020 / pyunits.year)
        opex = flow_total + mlc_cost

        results[unit_name] = {"capex": capex, "opex": opex}

    # RO aggregate (RBAT only)
    if treatment_train == "RBAT":
        # CAPEX: sum all RO unit capital costs * RO total investment factor
        tif_ro = m.fs.ro_costing.total_investment_factor
        mlc_ro = m.fs.ro_costing.maintenance_labor_chemical_factor
        capex_ro = 0 * pyunits.MUSD_2020
        for cb in m.fs.ro_costing._registered_unit_costing:
            capex_ro += pyunits.convert(cb.capital_cost * tif_ro * m.fs.ro_costing.capital_recovery_factor,
                                        to_units=pyunits.MUSD_2020 / pyunits.year)

        # Flow OPEX: pump + ERD electricity (annualized) + membrane replacement (aggregate fixed O&M)
        util_ro = m.fs.ro_costing.utilization_factor
        flow_ro = 0 * (pyunits.MUSD_2020 / pyunits.year)
        flow_ro += pyunits.convert(m.fs.ro_costing.aggregate_flow_costs["electricity"] * util_ro,
                                   to_units=pyunits.MUSD_2020 / pyunits.year)
        flow_ro += pyunits.convert(m.fs.ro_costing.aggregate_flow_costs["membrane_replacement"] * util_ro,
                                   to_units=pyunits.MUSD_2020 / pyunits.year)

        # MLC-based fixed O&M for RO (applied to aggregated RO CAPEX)
        mlc_ro_cost = mlc_ro * capex_ro / m.fs.ro_costing.capital_recovery_factor / tif_ro
        opex_ro = flow_ro + mlc_ro_cost
        brine_cost = pyunits.convert(m.fs.brine_disposal_cost, to_units=pyunits.MUSD_2020 / pyunits.year)

        results["RO"] = {"capex": capex_ro, "opex": opex_ro}
        results["Brine"] = {"capex": 0, "opex": brine_cost}
        total = (m.fs.total_capital_cost * m.fs.zo_costing.capital_recovery_factor + m.fs.total_operating_cost + m.fs.brine_disposal_cost) / 1000000
    else:
        total = (m.fs.total_capital_cost * m.fs.zo_costing.capital_recovery_factor + m.fs.total_operating_cost) / 1000000

    outputs["flowrate_MGD"] = m.fs.feed.flow_vol[0] * 1/0.0438126
    outputs["LCOW"] = m.fs.LCOW
    outputs["Total CAPEX (MGD)"] = m.fs.total_capital_cost / 1000000
    outputs["Total OPEX (MGD/year)"] = m.fs.total_operating_cost / 1000000
    outputs["SEC"] = m.fs.specific_energy_intensity

    for unit_name, vals in results.items():
        if unit_name == "Brine":
            continue
        outputs[f"{unit_name}_CAPEX_frac"] = vals["capex"] / total * 100
        outputs[f"{unit_name}_OPEX_frac"] = vals["opex"] / total * 100
    if treatment_train == "RBAT":
        outputs["brine_frac"] = results["Brine"]["opex"] / total * 100
        outputs["Total_Opex_frac"] = (m.fs.total_operating_cost / 1000000) / total * 100
    else:
        outputs["Total_Opex_frac"] = (m.fs.total_operating_cost / 1000000) / total * 100

    return outputs, optimize_kwargs, opt_function

def run_analysis(case_num=102, nx=21, interpolate_nan_outputs=True, output_filename=None):
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
            m.fs.feed.flow_vol[0], 1*0.0438126, 100*0.0438126, nx
            # m.fs.feed.flow_vol[0], 10 * 0.0438126, 90 * 0.0438126, 5
        )

    elif case_num == 2:
        state = "FL"
        treatment_train = "CBAT"
        description = "Default"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1*0.0438126, 100*0.0438126, nx
            # m.fs.feed.flow_vol[0], 10 * 0.0438126, 90 * 0.0438126, 5
        )


    elif case_num == 3:
        state = "CO"
        treatment_train = "CBAT"
        description = "TOC_sweep"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.conc_mass_comp[0, 'toc'].unfix()
        sweep_params["toc conc"] = LinearSample(
            m.fs.feed.conc_mass_comp[0, 'toc'], 0.007, 0.015, nx
        )

    elif case_num == 4:
        state = "FL"
        treatment_train = "CBAT"
        description = "TOC_sweep"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.conc_mass_comp[0, 'toc'].unfix()
        sweep_params["toc conc"] = LinearSample(
            m.fs.feed.conc_mass_comp[0, 'toc'], 0.007, 0.015, nx
        )

    elif case_num == 5:
        state = "CO"
        treatment_train = "RBAT"
        description = "Default"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1*0.0438126, 100*0.0438126, nx
            # m.fs.feed.flow_vol[0], 10 * 0.0438126, 90 * 0.0438126, 5
        )

    elif case_num == 6:
        state = "FL"
        treatment_train = "RBAT"
        description = "Default"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1*0.0438126, 100*0.0438126, nx
            # m.fs.feed.flow_vol[0], 10 * 0.0438126, 90 * 0.0438126, 5
        )

    elif case_num == 7:
        state = "CA"
        treatment_train = "RBAT"
        description = "Default_brine0.05"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1*0.0438126, 100*0.0438126, nx
            # m.fs.feed.flow_vol[0], 10 * 0.0438126, 90 * 0.0438126, 5
            # m.fs.feed.flow_vol[0], 100 * 0.0438126, 163 * 0.0438126, 4
        )

    elif case_num == 8:
        state = "CA"
        treatment_train = "RBAT"
        description = "flowrate_TDS"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()
        m.fs.feed.conc_mass_comp[0, 'tds'].unfix()

        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1*0.0438126, 100*0.0438126, nx
        )
        sweep_params["tds conc"] = LinearSample(
            m.fs.feed.conc_mass_comp[0, 'tds'], 0.5, 2, 3
        )

    elif case_num == 9:
        state = "CA"
        treatment_train = "RBAT"
        description = "flowrate_brine"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()
        m.fs.zo_costing.brine_disposal_cost.unfix()

        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 5 * 0.0438126, 100 * 0.0438126, nx
        )
        sweep_params["brine_disposal_cost"] = LinearSample(
            m.fs.zo_costing.brine_disposal_cost, 0.5, 2, nx
        )

    elif case_num == 10:
        state = "CA"
        treatment_train = "RBAT"
        description = "Not_Optimized"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()

        m.fs.non_RO.Ozone.contact_time[0].unfix
        m.fs.non_RO.Ozone.contact_time[0].fix(10)
        # m.fs.non_RO.Ozone.O3toTOC[0].unfix
        # m.fs.non_RO.Ozone.O3toTOC[0].fix(1)

        m.fs.non_RO.BAF.EBCT[0].unfix
        m.fs.non_RO.BAF.EBCT[0].fix(30)

        m.fs.non_RO.UV_AOP.hydrogen_peroxide_dose[0].unfix()
        m.fs.non_RO.UV_AOP.hydrogen_peroxide_dose[0].fix(5)

        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1 * 0.0438126, 100 * 0.0438126, nx
        )


    elif case_num == 11:
        state = "CO"
        treatment_train = "CBAT"
        description = "flowrate_TOClimit"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()
        m.fs.non_RO.toc_eff.unfix()

        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 5 * 0.0438126, 100 * 0.0438126, nx
        )
        sweep_params["effluent_toc_limit"] = LinearSample(
            m.fs.non_RO.toc_eff, 0.5, 2, nx
        )

    elif case_num == 12:
        state = "FL"
        treatment_train = "CBAT"
        description = "flowrate_TOClimit"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()
        m.fs.non_RO.toc_eff.unfix()

        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 5 * 0.0438126, 100 * 0.0438126, nx
        )
        sweep_params["effluent_toc_limit"] = LinearSample(
            m.fs.non_RO.toc_eff, 0.5, 2, nx
        )

    #todo: 18/14/15 + BAC, GAC, RO update + # of minimum =3
    elif case_num == 13:
        state = "CA"
        treatment_train = "CBAT"
        description = "Reg_Credit_#_Update_share"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            # m.fs.feed.flow_vol[0], 1*0.0438126, 100*0.0438126, nx
            m.fs.feed.flow_vol[0], 10 * 0.0438126, 90 * 0.0438126, 5
        )

    elif case_num == 14:
        state = "CA"
        treatment_train = "RBAT"
        description = "Reg_Credit_#_Update"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1*0.0438126, 100*0.0438126, nx
            # m.fs.feed.flow_vol[0], 10 * 0.0438126, 90 * 0.0438126, 5
        )

    elif case_num == 15:
        state = "CA"
        treatment_train = "CBAT"
        description = "TOC_TOClimit_80MGD"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)

        m.fs.feed.conc_mass_comp[0, 'toc'].unfix()
        m.fs.non_RO.toc_eff.unfix()

        sweep_params["toc conc"] = LinearSample(
            m.fs.feed.conc_mass_comp[0, 'toc'], 0.007, 0.02, nx
        )
        sweep_params["effluent_toc_limit"] = LinearSample(
            m.fs.non_RO.toc_eff, 0.5, 2, nx
        )

    elif case_num == 100:
        state = "CA"
        treatment_train = "CBAT"
        description = "Flow_TOClimit_toc10mgL"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)

        # m.fs.feed.conc_mass_comp[0, 'toc'].unfix()
        m.fs.feed.flow_vol[0].unfix()
        m.fs.non_RO.toc_eff.unfix()

        # sweep_params["toc conc"] = LinearSample(
        #     m.fs.feed.conc_mass_comp[0, 'toc'], 0.007, 0.02, nx
        # )
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 5 * 0.0438126, 100 * 0.0438126, nx
            # m.fs.feed.flow_vol[0], 10 * 0.0438126, 90 * 0.0438126, 5
        )
        sweep_params["effluent_toc_limit"] = LinearSample(
            m.fs.non_RO.toc_eff, 0.5, 2, nx
        )

    elif case_num == 101:
        state = "CA"
        treatment_train = "RBAT"
        description = "flowrate(atbrine$10m3v2)"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()

        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1 * 0.0438126, 100 * 0.0438126, 41
        )

    elif case_num == 102:
        state = "FL"
        treatment_train = "RBAT"
        description = "flowrate(atbrine$10m3)"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()

        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1 * 0.0438126, 100 * 0.0438126, 41
        )

    elif case_num == 16:
        state = "CA"
        treatment_train = "RBAT"
        description = "TOC_sweep_80MGD"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)

        m.fs.feed.conc_mass_comp[0, 'toc'].unfix()

        sweep_params["toc conc"] = LinearSample(
            m.fs.feed.conc_mass_comp[0, 'toc'], 0.007, 0.02, nx
        )

    elif case_num == 17:
        state = "CA"
        treatment_train = "RBAT"
        description = "applying_FL_case2"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            m.fs.feed.flow_vol[0], 1*0.0438126, 100*0.0438126, nx
            # m.fs.feed.flow_vol[0], 10 * 0.0438126, 90 * 0.0438126, 5
        )

    elif case_num == 18:
        state = "CA"
        treatment_train = "RBAT"
        description = "applying_CO_case2"
        m = dpr_flowsheet.main(state=state, treatment_train=treatment_train)[0]
        outputs, optimize_kwargs, opt_function = set_up_sensitivity(m, treatment_train=treatment_train)
        m.fs.feed.flow_vol[0].unfix()
        sweep_params["system_capacity"] = LinearSample(
            # m.fs.feed.flow_vol[0], 1*0.0438126, 100*0.0438126, nx
            m.fs.feed.flow_vol[0], 10 * 0.0438126, 90 * 0.0438126, 5
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