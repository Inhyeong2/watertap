import numpy as np
from math import log10

from pyomo.environ import (
    ConcreteModel,
    Var,
    NonNegativeReals,
    value,
    Constraint,
    Objective,
    Param,
    Set,
    TransformationFactory,
    assert_optimal_termination,
    units as pyunits,
)
import pyomo.environ as pyo
from pyomo.network import Arc
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from idaes.core import FlowsheetBlock
from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import solve_indexed_blocks, propagate_state
from idaes.models.unit_models import Mixer, Separator, Product, Feed
from idaes.models.unit_models.mixer import MomentumMixingType
from idaes.core import UnitModelCostingBlock
import idaes.core.util.scaling as iscale
from idaes.core.util.scaling import set_scaling_factor, calculate_scaling_factors, set_variable_scaling_from_current_value
import idaes.logger as idaeslog
from idaes.core.util.misc import StrEnum

import watertap.property_models.seawater_prop_pack as props
from watertap.core.membrane_channel_base import (
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    ModuleType,
    PressureChangeType,
)
from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.unit_models.pressure_exchanger import PressureExchanger
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice
from watertap.core.util.initialization import assert_degrees_of_freedom
from watertap.costing import WaterTAPCosting

def main():
    m = build()
    set_operating_conditions(m, flow_vol=50*0.0438126)
    print('DOF after setting operating conditions: ', degrees_of_freedom(m))
    scale_system(m)

    initialize_system(m)
    print('DOF after initialization: ', degrees_of_freedom(m))
    results = solve(m, tee=True)
    assert_optimal_termination(results)

    add_costing(m)
    optimize_set_up(m)
    results = solve(m, tee=True)
    assert_optimal_termination(results)

    # display_RO(m)
    # display_system(m)
    # display_design(m)

    return m


def build(n_stages=3):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = props.SeawaterParameterBlock()

    # Unit processes
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.brine = Product(property_package=m.fs.properties)
    m.fs.pump = Pump(property_package=m.fs.properties)
    m.fs.ERD = EnergyRecoveryDevice(property_package=m.fs.properties)

    # RO Kwargs
    ro_kwargs = {
        "concentration_polarization_type": ConcentrationPolarizationType.calculated,
        "mass_transfer_coefficient": MassTransferCoefficient.calculated,
        "pressure_change_type": PressureChangeType.calculated,
        "module_type": ModuleType.flat_sheet,
        "has_pressure_change": True,
        "has_full_reporting": True,
    }

    # Create RO stages set from n_stages
    stages = list(range(1, n_stages + 1))
    # print("stages", stages)
    m.fs.ro_stages = Set(initialize=stages, doc="RO stages")
    m.fs.RO = ReverseOsmosis0D(m.fs.ro_stages, property_package=m.fs.properties, **ro_kwargs)

    # Mixer for permeate streams from all RO stages
    m.fs.P_mixer = Mixer(
        property_package=m.fs.properties,
        inlet_list=[f"stage_{i}" for i in m.fs.ro_stages],
        momentum_mixing_type=MomentumMixingType.minimize,
    )

    # Arcs for connections
    m.fs.feed_to_pump = Arc(source=m.fs.feed.outlet, destination=m.fs.pump.inlet)

    # Connect RO stages
    for i in m.fs.ro_stages:
        if i == 1:
            setattr(
                m.fs,
                f"ro_inlet_arc_{i}",
                Arc(source=m.fs.pump.outlet, destination=m.fs.RO[i].inlet),
            )
        else:
            setattr(
                m.fs,
                f"ro_inlet_arc_{i}",
                Arc(source=m.fs.RO[i - 1].retentate, destination=m.fs.RO[i].inlet),
            )

        # Permeate mixer connections
        setattr(
            m.fs,
            f"stage_{i}_to_P_mixer",
            Arc(
                source=m.fs.RO[i].permeate,
                destination=getattr(m.fs.P_mixer, f"stage_{i}"),
            ),
        )

    m.fs.P_mixer_to_product = Arc(
        source=m.fs.P_mixer.outlet, destination=m.fs.product.inlet
    )

    last_stage = stages[-1]
    m.fs.retentate_to_erd = Arc(source=m.fs.RO[last_stage].retentate, destination=m.fs.ERD.inlet)
    m.fs.erd_to_brine = Arc(source=m.fs.ERD.outlet, destination=m.fs.brine.inlet)

    # Expand the arcs in the flowsheet
    TransformationFactory("network.expand_arcs").apply_to(m)

    # print("\n=== All Arcs in flowsheet ===")
    # for arc_name, arc in m.fs.component_map(Arc).items():
    #     print(f"{arc_name}: {arc.source.name} --> {arc.destination.name}")

    # Add water recovery variable
    m.fs.water_recovery = Var(
        initialize=0.75,
        domain=NonNegativeReals,
        doc="Water recovery across the RO stages",
        units=pyunits.dimensionless,
    )

    @m.fs.Constraint(doc="Constraint to enforce water recovery across RO stages", )
    def water_recovery_constraint(b):
        return (
                b.water_recovery * b.feed.properties[0].flow_vol_phase["Liq"]
                == b.product.properties[0].flow_vol_phase["Liq"]
        )

    # # costing
    # m.fs.costing = WaterTAPCosting()
    # m.fs.pump.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    # m.fs.RO.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    # m.fs.ERD.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    # m.fs.costing.cost_process()
    # m.fs.costing.add_annual_water_production(m.fs.product.properties[0].flow_vol)
    # m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol)
    # m.fs.costing.add_specific_energy_consumption(m.fs.product.properties[0].flow_vol)
    # m.fs.costing.add_specific_electrical_carbon_intensity(
    #     m.fs.product.properties[0].flow_vol
    # )

    # m.fs.product.properties[0].flow_vol_phase[...] mixer에 flow vol phase를 해야할거같은데
    return m

def set_operating_conditions(
        m, flow_vol = None,
        feed_salinity = 0.5
):
    # Feed
    m.fs.feed.properties[0].pressure.fix(101325)  # feed pressure [Pa]
    m.fs.feed.properties[0].temperature.fix(273.15 + 25)  # feed temperature [K]
    m.fs.feed.properties[0].pressure_osm_phase[...] #touching
    # solve(m.fs.feed)
    calculate_feed_state(m, feed_salinity, flow_vol)
    # m.fs.feed.pprint()

    # Main pump
    m.fs.pump.efficiency_pump.fix(0.80)
    m.fs.pump.outlet.pressure[0].fix(40*101325)

    # RO unit
    for s_id, stage in m.fs.RO.items():
        stage.A_comp[0, "H2O"].fix(4.2e-12)  # m^2/s/bar, water permeability
        stage.B_comp[0, "TDS"].fix(3.5e-8)  # m/s/bar, Salt permeability
        stage.feed_side.channel_height.fix(1e-3)  # 1 mm channel height
        stage.length.fix(6)  # 6 m length of the RO module
        stage.feed_side.spacer_porosity.fix(0.85)  # 85% porosity of the spacer
        stage.mixed_permeate[0].pressure.fix(101325)  # 1 atm
        stage.feed_side.velocity[0,0].fix(0.25)
        stage.feed_side.area.setub(None)
        stage.width.setub(None)
        stage.area.setub(None)
        width_guess = flow_vol / stage.feed_side.velocity[0, 0].value / stage.feed_side.spacer_porosity.value / stage.feed_side.channel_height.value
        stage.width.fix(width_guess)
        stage.width.unfix()
        stage.recovery_vol_phase[0,"Liq"].fix(0.5)
        stage.recovery_vol_phase[0, "Liq"].unfix()
    # m.fs.RO.pprint()

    # ERD unit
    m.fs.ERD.efficiency_pump.fix(0.8)
    m.fs.ERD.control_volume.properties_out[0].pressure.fix(101325)  # Fix ERD outlet pressure to 1 atm

    # for ro_stage_name, ro_stage in m.fs.RO.items():
    #     print(f"\n=== RO Stage: {ro_stage_name} ===")
    #
    #     for pos in ro_stage.feed_side.properties.keys():
    #         print(f"\nPosition: {pos}")
    #
    #         # Feed side
    #         print("Feed side:")
    #         for (p, j), v in ro_stage.feed_side.properties[pos].flow_mass_phase_comp.items():
    #             print(f"  (phase={p}, comp={j}) → var={v}")
    #
    #         # Interface
    #         print("Interface side:")
    #         for (p, j), v in ro_stage.feed_side.properties_interface[pos].flow_mass_phase_comp.items():
    #             print(f"  (phase={p}, comp={j}) → var={v}")
    #
    #         # Permeate side
    #         print("Permeate side:")
    #         for (p, j), v in ro_stage.permeate_side[pos].flow_mass_phase_comp.items():
    #             print(f"  (phase={p}, comp={j}) → var={v}")

    # # Costing
    # m.fs.costing.base_currency = pyunits.USD_2020
    # m.fs.costing.electricity_cost = 0.08
    # m.fs.costing.wacc = 0.09307339771758532
    # m.fs.costing.plant_lifetime = 30
    # m.fs.costing.utilization_factor = 0.9
    # # m.fs.costing.land_cost_percent_FCI = 0
    # # m.fs.costing.working_capital_percent_FCI = 1.307
    # # m.fs.costing.salaries_percent_FCI = 0
    # # m.fs.costing.benefit_percent_of_salary = 0
    # # m.fs.costing.maintenance_costs_percent_FCI = 0.04
    # # m.fs.costing.laboratory_fees_percent_FCI = 0
    # # m.fs.costing.insurance_and_taxes_percent_FCI = 0
    #
    # m.fs.costing.maintenance_labor_chemical_factor = 0.04
    # m.fs.costing.TIC = 1.3
    # m.fs.costing.total_investment_factor = 3/m.fs.costing.TIC.value
    #
    # m.fs.costing.reverse_osmosis.membrane_cost.fix(30 * m.fs.costing.total_investment_factor.value)
    # m.fs.costing.reverse_osmosis.factor_membrane_replacement.fix(0.2 * m.fs.costing.utilization_factor.value) # utilization factor should be considered
    #                                                                                                          # because it is not considered for fixed_operating_cost
    #                                                                                                          # but considered for flow_cost
    # m.fs.pump.costing.costing_package.high_pressure_pump.cost.fix(53 / 1e5 * 3600 *m.fs.costing.total_investment_factor.value)
    # m.fs.ERD.costing.costing_package.energy_recovery_device.pressure_exchanger_cost.fix(535 * m.fs.costing.total_investment_factor.value)
    #
    # # m.fs.RO.costing.del_component(m.fs.RO.costing.capital_cost_constraint)
    # # m.fs.RO.costing.capital_cost_constraint = pyo.Constraint(
    # #     expr=m.fs.RO.costing.capital_cost
    # #          == m.fs.RO.costing.cost_factor
    # #          * 2.307 * m.fs.costing.reverse_osmosis.membrane_cost * m.fs.RO.area  #todo: we delete unit conversion to_units=m.fs.RO.costing.costing_package.base_currency,
    # # )
    # #
    # # m.fs.pump.costing.del_component(m.fs.pump.costing.capital_cost_constraint)
    # # t0 = m.fs.pump.flowsheet().time.first()
    # # m.fs.pump.costing.capital_cost_constraint = pyo.Constraint(
    # #     expr=m.fs.pump.costing.capital_cost
    # #          == m.fs.pump.costing.cost_factor
    # #          * 2.307 * m.fs.pump.costing.costing_package.high_pressure_pump.cost
    # #          * pyunits.convert(m.fs.pump.work_mechanical[t0], pyunits.W)
    # # )
    #
    #
    # # m.fs.RO.costing.capital_cost_constraint = pyo.Constraint(
    # #     expr=m.fs.RO.costing.capital_cost
    # #          == m.fs.RO.costing.cost_factor
    # #          * pyo.units.convert(
    # #         1 * m.fs.costing.reverse_osmosis.membrane_cost * m.fs.RO.area,
    # #         to_units=m.fs.RO.costing.costing_package.base_currency,
    # #     )
    # # )

    return

def scale_system(m):
    # Feed
    set_scaling_factor(m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"], 1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value)
    set_scaling_factor(m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"], 1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value)

    # Pump
    op_pressure = m.fs.pump.control_volume.properties_out[0].pressure.value
    set_scaling_factor(m.fs.pump.control_volume.work, 1 / (
            op_pressure * m.fs.feed.properties[0].flow_vol_phase['Liq'].value /
            m.fs.pump.efficiency_pump[0].value))
    set_scaling_factor(m.fs.pump.work_fluid[0], 1 / (
            op_pressure * m.fs.feed.properties[0].flow_vol_phase['Liq'].value))
    set_scaling_factor(m.fs.pump.control_volume.properties_out[0].pressure, 1 / op_pressure)
    set_scaling_factor(m.fs.pump.control_volume.properties_in[0].pressure, 1 / 101325)
    set_scaling_factor(
        m.fs.pump.control_volume.properties_out[0].flow_vol_phase["Liq"],
        1 / m.fs.feed.properties[0].flow_vol_phase["Liq"].value
    )

    # ERD unit
    set_scaling_factor(
        m.fs.ERD.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "H2O"],
        1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value / 0.1
    )
    set_scaling_factor(
        m.fs.ERD.control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"],
        1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value / 0.1
    )
    set_scaling_factor(
        m.fs.ERD.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "TDS"],
        100 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
    )
    set_scaling_factor(
        m.fs.ERD.control_volume.properties_out[0].flow_mass_phase_comp["Liq", "TDS"],
        100 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
    )
    set_scaling_factor(m.fs.ERD.control_volume.work, 1 / (
            op_pressure * m.fs.feed.properties[0].flow_vol_phase['Liq'].value * 0.1 /
            m.fs.ERD.efficiency_pump[0].value))
    set_scaling_factor(m.fs.pump.work_fluid[0], 1 / (
            op_pressure * m.fs.feed.properties[0].flow_vol_phase['Liq'].value * 0.1)
                       )
    set_scaling_factor(m.fs.ERD.control_volume.properties_in[0].pressure, 1 / op_pressure)
    set_scaling_factor(m.fs.ERD.control_volume.properties_out[0].pressure, 1 / 101325)
    set_scaling_factor(
        m.fs.ERD.control_volume.properties_out[0].flow_vol_phase["Liq"],
        1 / (m.fs.feed.properties[0].flow_vol_phase['Liq'].value * 0.1)
    )

    # Reverse Osmosis (RO) unit models
    for ro_stage in m.fs.RO.values():
        for pos in ro_stage.feed_side.properties.keys():
            # Feed side flow variables
            for (p, j), v in ro_stage.feed_side.properties[
                pos
            ].flow_mass_phase_comp.items():  # Change to 0D Reverse Osmosis
                if p == "Liq" and j == "H2O":
                    set_scaling_factor(v, 1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value)
                elif p == "Liq" and j == "TDS":
                    set_scaling_factor(v, 1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value)
                else:
                    raise ValueError(
                        "Unexpected phase or component in feed side flow variables."
                    )

            # Interface flow variables
            for (p, j), v in ro_stage.feed_side.properties_interface[
                pos
            ].flow_mass_phase_comp.items():
                if p == "Liq" and j == "H2O":
                    set_scaling_factor(v, 1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value)
                elif p == "Liq" and j == "TDS":
                    set_scaling_factor(v, 1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value)
                else:
                    raise ValueError(
                        "Unexpected phase or component in feed side flow variables."
                    )

            # Permeate side flow variables
            for (p, j), v in ro_stage.permeate_side[pos].flow_mass_phase_comp.items():
                if p == "Liq" and j == "H2O":
                    set_scaling_factor(v, 1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value)
                elif p == "Liq" and j == "TDS":
                    set_scaling_factor(v, 10 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value)
                else:
                    raise ValueError(
                        "Unexpected phase or component in feed side flow variables."
                    )

        # Mixed permeate flow variables
        set_scaling_factor(
            ro_stage.mixed_permeate[0].flow_mass_phase_comp["Liq", "H2O"], 1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value
        )
        set_scaling_factor(
            ro_stage.mixed_permeate[0].flow_mass_phase_comp["Liq", "TDS"], 10 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
        )

        set_scaling_factor(ro_stage.length, 1e-1)  # Length of the RO module
        set_scaling_factor(ro_stage.feed_side.channel_height, 1e3)  # Channel height
        set_scaling_factor(ro_stage.feed_side.spacer_porosity, 1)  # Spacer porosity
        set_scaling_factor(ro_stage.A_comp[0, "H2O"], 1e12)  # Water permeability
        set_scaling_factor(ro_stage.B_comp[0, "TDS"], 1e8)  # Salt permeability
        set_scaling_factor(ro_stage.mixed_permeate[0].pressure, 1e-5)
        set_scaling_factor(ro_stage.recovery_vol_phase[0, "Liq"], 2)

        set_variable_scaling_from_current_value(ro_stage.width)
        calculate_variable_from_constraint(ro_stage.area, ro_stage.eq_area)
        set_variable_scaling_from_current_value(ro_stage.area)
        calculate_variable_from_constraint(ro_stage.feed_side.area, ro_stage.feed_side.eq_area)
        set_variable_scaling_from_current_value(ro_stage.feed_side.area)

        calculate_scaling_factors(ro_stage)

        # Mixer
        for i in m.fs.ro_stages:
            state = getattr(m.fs.P_mixer, f"stage_{i}_state")
            set_scaling_factor(state[0].flow_mass_phase_comp["Liq", "H2O"], 1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value)
            set_scaling_factor(state[0].flow_mass_phase_comp["Liq", "TDS"], 1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value)

        set_scaling_factor(
            m.fs.P_mixer.mixed_state[0].flow_mass_phase_comp["Liq", "H2O"], 1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value
        )
        set_scaling_factor(
            m.fs.P_mixer.mixed_state[0].flow_mass_phase_comp["Liq", "TDS"], 10 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
        )

def initialize_system(m, verbose=False):
    # Touch the osmotic pressure variables to ensure they are initialized
    # m.fs.feed.properties[0].pressure_osm_phase[...]
    # m.fs.feed.properties[0].flow_vol_phase[...]
    # m.fs.feed.properties[0].conc_mass_phase_comp[...]

    m.fs.feed.initialize()
    propagate_state(m.fs.feed_to_pump)

    m.fs.pump.initialize()
    # Initialize each RO stage sequentially
    for s_id, stage in m.fs.RO.items():
        if hasattr(m.fs, "ro_inlet_arc_" + str(s_id)):
            if verbose:
                # print the arc
                print(f"Initializing RO stage {s_id} with inlet arc")
                getattr(m.fs, "ro_inlet_arc_" + str(s_id)).pprint()

        propagate_state(getattr(m.fs, "ro_inlet_arc_" + str(s_id)))
        stage.initialize()
        if verbose:
            stage.report()
        # propagate the permeate stream to the mixer
        propagate_state(getattr(m.fs, f"stage_{s_id}_to_P_mixer"))

    # Initialize the permeate mixer after all RO stages are initialized
    m.fs.P_mixer.initialize()
    propagate_state(m.fs.P_mixer_to_product)

    m.fs.product.properties[0].flow_vol_phase[...]
    m.fs.product.properties[0].conc_mass_phase_comp[...]
    m.fs.product.initialize()

    propagate_state(m.fs.retentate_to_erd)

    m.fs.ERD.initialize()
    m.fs.brine.properties[0].flow_vol_phase[...]
    m.fs.brine.properties[0].conc_mass_phase_comp[...]
    propagate_state(m.fs.erd_to_brine)

    m.fs.brine.initialize()

    if verbose:
        m.fs.product.report()
        m.fs.ERD.report()
        m.fs.brine.report()

    # m.fs.costing.initialize()

def add_costing(m):
    # Add Costing
    m.fs.product.properties[0].flow_vol[...]
    m.fs.costing = WaterTAPCosting()
    m.fs.pump.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    for stage in m.fs.RO.values():
        stage.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.ERD.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.product.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.product.properties[0].flow_vol)
    m.fs.costing.initialize()
    return



def optimize_set_up(m):
    # add objective
    m.fs.objective = Objective(expr=m.fs.costing.LCOW)

    """
           Unfixes RO operating conditions and sets solver objective
               - Operating pressure: 1 - 83 bar
               - Crossflow velocity: 20 - 30 cm/s
               - Volumetric recovery: 30 - 80 %
               - Length: 6 - 8 m
               - Area and width were already unfixed
           """

    # RO operating pressure
    m.fs.pump.control_volume.properties_out[0].pressure.unfix()
    m.fs.pump.control_volume.properties_out[0].pressure.setub(8300000)
    m.fs.pump.control_volume.properties_out[0].pressure.setlb(100000)
    # m.fs.pump.deltaP.setlb(0)

    # RO unit
    for s_id, stage in m.fs.RO.items():
        stage.feed_side.velocity[0, 0].unfix()
        stage.feed_side.velocity[0, 0].setub(0.3)
        stage.feed_side.velocity[0, 0].setlb(0.2)
        stage.feed_side.velocity[0, 1].setlb(0.1)

        stage.length.unfix()
        stage.length.setub(8)
        stage.length.setlb(6)

        stage.recovery_vol_phase[0, "Liq"].unfix()
        stage.recovery_vol_phase[0, "Liq"].setub(0.70)
        stage.recovery_vol_phase[0, "Liq"].setlb(0.30)

        stage.mixed_permeate[0].conc_mass_phase_comp["Liq", "TDS"].setub(0.5)
        stage.rejection_phase_comp[0, "Liq", "TDS"].setlb(0.99)

    # m.fs.brine.properties[0].conc_mass_phase_comp






def display_RO(m):
    print('Feed concentration: {} kg/m3'.format(m.fs.feed.properties[0].conc_mass_phase_comp['Liq', 'TDS'].value))
    print('Feed osmotic pressure: {} bar'.format(m.fs.feed.properties[0].pressure_osm_phase['Liq'].value/1e5))
    print(f'Pump outlet pressure: {m.fs.pump.outlet.pressure[0].value/1e5} bar')
    print(f'Membrane area: {m.fs.RO.area.value} m2')
    print(f'Membrane length: {m.fs.RO.length.value} m')
    print(f'Membrane width: {m.fs.RO.width.value} m')
    print(f'Inlet velocity: {m.fs.RO.feed_side.velocity[0,0].value} m/s')
    print('Recovery: {:.2f} %'.format(m.fs.RO.recovery_vol_phase[0, 'Liq'].value*100))
    # print(f'RO capital cost: {m.fs.RO.costing.capital_cost.value/1e6} $M(2018)')
    # print(f'Pump capital cost: {m.fs.pump.costing.capital_cost.value / 1e6} $M(2018)')
    # print(f'ERD capital cost: {m.fs.ERD.costing.capital_cost.value / 1e6} $M(2018)')
    # units = pyunits.get_units(m.fs.RO.costing.capital_cost)
    # print(units)
    print(f'RO capital cost: {value(pyunits.convert(m.fs.RO.costing.capital_cost,to_units= pyunits.MUSD_2020))} $M(2020)')
    print(f'Pump capital cost: {value(pyunits.convert(m.fs.pump.costing.capital_cost, to_units=pyunits.MUSD_2020))} $M(2020)')
    print(f'ERD capital cost: {value(pyunits.convert(m.fs.ERD.costing.capital_cost, to_units=pyunits.MUSD_2020))} $M(2020)')

    print(
        f'RO total capital cost: {value(pyunits.convert(m.fs.RO.costing.capital_cost * m.fs.costing.total_investment_factor, to_units=pyunits.MUSD_2020))} $M(2020)')
    print(
        f'Pump total capital cost: {value(pyunits.convert(m.fs.pump.costing.capital_cost * m.fs.costing.total_investment_factor, to_units=pyunits.MUSD_2020))} $M(2020)')
    print(
        f'ERD total capital cost: {value(pyunits.convert(m.fs.ERD.costing.capital_cost * m.fs.costing.total_investment_factor, to_units=pyunits.MUSD_2020))} $M(2020)')
    print(f'Total capital cost (all): {value(pyunits.convert((m.fs.RO.costing.capital_cost+m.fs.pump.costing.capital_cost+m.fs.ERD.costing.capital_cost)* m.fs.costing.total_investment_factor, to_units=pyunits.MUSD_2020))} $M(2020)')

    print(f'total_investment_factor: {m.fs.costing.total_investment_factor.value}')

    print(f'total installed cost factor (TIC): {m.fs.costing.TIC.value}')
    # print(f'ro costing factor: {value(m.fs.RO.costing.cost_factor)}')
    # print(m.fs.RO.costing.costing_package.base_currency)
    # ratio = value(pyunits.convert(1 * pyunits.USD_2018, to_units=pyunits.USD_2020))
    # print("USD_2018 → USD_2020 ratio:", ratio)

    print(f'Pump work: {m.fs.pump.control_volume.work[0].value}')
    print(f'ERD work: {m.fs.ERD.control_volume.work[0].value}')
    print(pyunits.get_units(m.fs.pump.control_volume.work[0]))

    total_fixed_operating_cost_cal = value(pyunits.convert(
        m.fs.costing.aggregate_fixed_operating_cost * 0 #todo: this should be considered as flow (only RO has this value)
        + m.fs.costing.maintenance_labor_chemical_operating_cost,
        to_units=pyunits.MUSD_2020 / pyunits.year))
    total_variable_operating_cost_vop_cal = value(
        pyunits.convert(m.fs.costing.aggregate_variable_operating_cost,
                        to_units=pyunits.MUSD_2020 / pyunits.year))
    print("Used flows:")
    for flow in m.fs.costing.used_flows:
        print(flow)
    total_flow_cost_cal = value(
        pyunits.convert(
            (sum(m.fs.costing.aggregate_flow_costs[flow] for flow in m.fs.costing.used_flows)
             + m.fs.costing.aggregate_fixed_operating_cost * 1) #todo: this should be considered as flow (only RO has this value)
            * m.fs.costing.utilization_factor,
            to_units=pyunits.MUSD_2020 / pyunits.year
        )
    )
    total_operating_cost_cal = total_fixed_operating_cost_cal + total_variable_operating_cost_vop_cal + total_flow_cost_cal
    total_variable_operating_cost_cal = total_variable_operating_cost_vop_cal + total_flow_cost_cal

    print(f"(Calculated) Total Operating Cost (Cop,tot): {total_operating_cost_cal:.4f} M$/year")
    print(f"(Calcualted) Total fixed operating cost (Cop,fix): {total_fixed_operating_cost_cal:.4f} M$/year")
    print(f"(Calculated) Total variable operating cost (Cop,fix): {total_variable_operating_cost_cal:.4f} M$/year")
    print(
        f"(Calculated) Total variable operating cost from unit models (Cvop,u): {total_variable_operating_cost_vop_cal:.4f} M$/year")
    print(f"(Calculated) Total flow cost (futil*Cflow,tot): {total_flow_cost_cal:.4f} M$/year")



    print(f'LCOW: {value(m.fs.costing.LCOW)} $/m3')
    print(f'SEC: {value(m.fs.costing.specific_energy_consumption)} kWh/m3')
    # m.fs.RO.costing.pprint()
    m.fs.costing.pprint()

def display_system(m):
    print("---system metrics---")
    feed_flow_mass = sum(
        m.fs.feed.flow_mass_phase_comp[0, "Liq", j].value for j in ["H2O", "TDS"]
    )
    feed_mass_frac_TDS = (
        m.fs.feed.flow_mass_phase_comp[0, "Liq", "TDS"].value / feed_flow_mass
    )
    print("Feed: %.2f kg/s, %.0f ppm" % (feed_flow_mass, feed_mass_frac_TDS * 1e6))

    prod_flow_mass = sum(
        m.fs.product.flow_mass_phase_comp[0, "Liq", j].value for j in ["H2O", "TDS"]
    )
    prod_mass_frac_TDS = (
        m.fs.product.flow_mass_phase_comp[0, "Liq", "TDS"].value / prod_flow_mass
    )
    print("Product: %.3f kg/s, %.0f ppm" % (prod_flow_mass, prod_mass_frac_TDS * 1e6))

    print(
        "Volumetric recovery: %.1f%%"
        % (value(m.fs.RO.recovery_vol_phase[0, "Liq"]) * 100)
    )
    print(
        "Water recovery: %.1f%%"
        % (value(m.fs.RO.recovery_mass_phase_comp[0, "Liq", "H2O"]) * 100)
    )
    print(
        "Energy Consumption: %.1f kWh/m3"
        % value(m.fs.costing.specific_energy_consumption)
    )
    print("Levelized cost of water: %.2f $/m3" % value(m.fs.costing.LCOW))

def display_design(m):
    print("---decision variables---")
    print("Operating pressure %.1f bar" % (m.fs.RO.inlet.pressure[0].value / 1e5))
    print("Membrane area %.1f m2" % (m.fs.RO.area.value))

    print("---design variables---")
    print(
        "Pump 1\noutlet pressure: %.1f bar\npower %.2f kW"
        % (
            m.fs.pump.outlet.pressure[0].value / 1e5,
            m.fs.pump.work_mechanical[0].value / 1e3,
        )
    )
    print(
        "ERD\ninlet pressure: %.1f bar\npower recovered %.2f kW"
        % (
            m.fs.ERD.inlet.pressure[0].value / 1e5,
            -1 * m.fs.ERD.work_mechanical[0].value / 1e3,
        )
    )

def calculate_feed_state(m, feed_salinity, flow_vol):
    m.fs.feed.properties[0].flow_mass_phase_comp[...].unfix()  # Unfix the mass flow rates to recalculate them based on the new conditions
    m.fs.feed.properties.calculate_state(
        var_args={
            (
                "conc_mass_phase_comp",
                ("Liq", "TDS"),
            ): feed_salinity,  # feed mass concentration
            ("flow_vol_phase", "Liq"): flow_vol,
        },  # volumetric feed flowrate [-]
        hold_state=True,  # fixes the calculated component mass flow rates
    )
    print(
        f"Fixed the feed conditions to salinity: "
        f"{value(m.fs.feed.properties[0].conc_mass_phase_comp['Liq', 'TDS'])}"
        f"{pyunits.get_units(m.fs.feed.properties[0].conc_mass_phase_comp['Liq', 'TDS'])}"
        f" and volumetric flow rate: {value(m.fs.feed.properties[0].flow_vol_phase['Liq'])}"
        f"{pyunits.get_units(m.fs.feed.properties[0].flow_vol_phase['Liq'])}"
    )
    return

def solve(blk, solver=None, tee=False):
    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=tee)
    return results

if __name__ == "__main__":
    model = main()
    for i, ro_stage in model.fs.RO.items():
        print(f"\n=== RO Stage {i} ===")
        ro_stage.report()
    model.fs.pump.pprint()
    model.fs.P_mixer.pprint()
