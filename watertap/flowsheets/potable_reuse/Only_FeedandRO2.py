import numpy as np
from math import log10

from pyomo.environ import (
    ConcreteModel,
    value,
    Constraint,
    Objective,
    Param,
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
import idaes.logger as idaeslog
from idaes.core.util.misc import StrEnum

import watertap.property_models.seawater_prop_pack as props
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
    set_operating_conditions(m, flow_vol=10*0.0438)
    print('DOF after setting operating conditions: ', degrees_of_freedom(m))
    scale_system(m)

    initialize_system(m)
    print('DOF after initialization: ', degrees_of_freedom(m))
    results = solve(m, tee=True)
    assert_optimal_termination(results)

    # optimize_set_up(m)
    # results = solve(m, tee=True)
    # assert_optimal_termination(results)

    display_RO(m)
    # display_system(m)
    # display_design(m)

    return m


def build():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = props.SeawaterParameterBlock()

    # Unit processes
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.disposal = Product(property_package=m.fs.properties)
    m.fs.pump = Pump(property_package=m.fs.properties)
    m.fs.RO = ReverseOsmosis0D(
        property_package=m.fs.properties,
        has_pressure_change=True,
        pressure_change_type=PressureChangeType.calculated,
        mass_transfer_coefficient=MassTransferCoefficient.calculated,
        concentration_polarization_type=ConcentrationPolarizationType.calculated,
    )
    m.fs.ERD = EnergyRecoveryDevice(property_package=m.fs.properties)
    # m.fs.ERD.pprint()

    # Connections
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.pump.inlet)
    m.fs.s02 = Arc(source=m.fs.pump.outlet, destination=m.fs.RO.inlet)
    m.fs.s03 = Arc(source=m.fs.RO.permeate, destination=m.fs.product.inlet)
    m.fs.s04 = Arc(source=m.fs.RO.retentate, destination=m.fs.ERD.inlet)
    m.fs.s05 = Arc(source=m.fs.ERD.outlet, destination=m.fs.disposal.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    # costing
    m.fs.costing = WaterTAPCosting()
    m.fs.pump.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.RO.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.ERD.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.product.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.product.properties[0].flow_vol)
    m.fs.costing.add_specific_electrical_carbon_intensity(
        m.fs.product.properties[0].flow_vol
    )

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

    # Main pump
    m.fs.pump.efficiency_pump.fix(0.80)
    m.fs.pump.outlet.pressure[0].fix(40*101325)

    # RO unit
    m.fs.RO.A_comp.fix(4.2e-12)  # membrane water permeability
    m.fs.RO.B_comp.fix(3.5e-8)  # membrane salt permeability
    m.fs.RO.feed_side.channel_height.fix(1e-3)  # channel height in membrane stage [m]
    m.fs.RO.length.fix(8)  # membrane length [m]
    m.fs.RO.feed_side.spacer_porosity.fix(0.85)  # spacer porosity in membrane stage [-]
    m.fs.RO.permeate.pressure[0].fix(101325)  # atmospheric pressure [Pa]
    m.fs.RO.feed_side.velocity[0, 0].fix(0.25)

    feed_side_area_guess = flow_vol / m.fs.RO.feed_side.velocity[0, 0].value
    width_guess = feed_side_area_guess / m.fs.RO.feed_side.spacer_porosity.value / m.fs.RO.feed_side.channel_height.value

    m.fs.RO.feed_side.area.setub(None)
    m.fs.RO.width.setub(None)
    m.fs.RO.area.setub(None)
    m.fs.RO.width.fix(width_guess)
    m.fs.RO.width.unfix() # Unfixed variables are width and recovery
    m.fs.RO.recovery_vol_phase[0, "Liq"].fix(0.5)
    m.fs.RO.recovery_vol_phase[0, "Liq"].unfix()

    # ERD unit
    m.fs.ERD.efficiency_pump.fix(0.8)
    m.fs.ERD.control_volume.properties_out[0].pressure.fix(101325)  # Fix ERD outlet pressure to 1 atm

    # Costing
    m.fs.costing.base_currency = pyunits.USD_2020
    m.fs.costing.electricity_cost = 0.08
    m.fs.costing.wacc = 0.09307339771758532
    m.fs.costing.plant_lifetime = 30
    m.fs.costing.utilization_factor = 0.9
    # m.fs.costing.land_cost_percent_FCI = 0
    # m.fs.costing.working_capital_percent_FCI = 1.307
    # m.fs.costing.salaries_percent_FCI = 0
    # m.fs.costing.benefit_percent_of_salary = 0
    # m.fs.costing.maintenance_costs_percent_FCI = 0.04
    # m.fs.costing.laboratory_fees_percent_FCI = 0
    # m.fs.costing.insurance_and_taxes_percent_FCI = 0

    m.fs.costing.maintenance_labor_chemical_factor = 0.04
    m.fs.costing.TIC = 1.3
    m.fs.costing.total_investment_factor = 3/m.fs.costing.TIC.value

    m.fs.costing.reverse_osmosis.membrane_cost.fix(30 * m.fs.costing.total_investment_factor.value)
    m.fs.costing.reverse_osmosis.factor_membrane_replacement.fix(0.2 * m.fs.costing.utilization_factor.value) # utilization factor should be considered
                                                                                                             # because it is not considered for fixed_operating_cost
                                                                                                             # but considered for flow_cost
    m.fs.pump.costing.costing_package.high_pressure_pump.cost.fix(53 / 1e5 * 3600 *m.fs.costing.total_investment_factor.value)
    m.fs.ERD.costing.costing_package.energy_recovery_device.pressure_exchanger_cost.fix(535 * m.fs.costing.total_investment_factor.value)

    # m.fs.RO.costing.del_component(m.fs.RO.costing.capital_cost_constraint)
    # m.fs.RO.costing.capital_cost_constraint = pyo.Constraint(
    #     expr=m.fs.RO.costing.capital_cost
    #          == m.fs.RO.costing.cost_factor
    #          * 2.307 * m.fs.costing.reverse_osmosis.membrane_cost * m.fs.RO.area  #todo: we delete unit conversion to_units=m.fs.RO.costing.costing_package.base_currency,
    # )
    #
    # m.fs.pump.costing.del_component(m.fs.pump.costing.capital_cost_constraint)
    # t0 = m.fs.pump.flowsheet().time.first()
    # m.fs.pump.costing.capital_cost_constraint = pyo.Constraint(
    #     expr=m.fs.pump.costing.capital_cost
    #          == m.fs.pump.costing.cost_factor
    #          * 2.307 * m.fs.pump.costing.costing_package.high_pressure_pump.cost
    #          * pyunits.convert(m.fs.pump.work_mechanical[t0], pyunits.W)
    # )


    # m.fs.RO.costing.capital_cost_constraint = pyo.Constraint(
    #     expr=m.fs.RO.costing.capital_cost
    #          == m.fs.RO.costing.cost_factor
    #          * pyo.units.convert(
    #         1 * m.fs.costing.reverse_osmosis.membrane_cost * m.fs.RO.area,
    #         to_units=m.fs.RO.costing.costing_package.base_currency,
    #     )
    # )

    return

def scale_system(m):
    # Feed
    m.fs.properties.set_default_scaling("flow_mass_phase_comp",
                                        1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value,
                                        index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling("flow_mass_phase_comp",
                                        1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value,
                                        index=("Liq", "TDS"))
    iscale.set_scaling_factor(m.fs.feed.temperature[0], 1e-2)
    iscale.set_scaling_factor(m.fs.feed.pressure[0], 1e-5)

    # Pump
    op_pressure = m.fs.pump.control_volume.properties_out[0].pressure.value
    iscale.set_scaling_factor(m.fs.pump.control_volume.work, 1 / (
            op_pressure * m.fs.feed.properties[0].flow_vol_phase['Liq'].value /
            m.fs.pump.efficiency_pump[0].value))
    iscale.set_scaling_factor(m.fs.pump.work_fluid[0], 1 / (
            op_pressure * m.fs.feed.properties[0].flow_vol_phase['Liq'].value))
    iscale.set_scaling_factor(m.fs.pump.control_volume.properties_out[0].pressure, 1 / op_pressure)
    iscale.set_scaling_factor(m.fs.pump.control_volume.properties_in[0].pressure, 1 / 101325)
    iscale.set_scaling_factor(
        m.fs.pump.control_volume.properties_out[0].flow_vol_phase["Liq"],
        1 / m.fs.feed.properties[0].flow_vol_phase["Liq"].value
    )

    # RO unit
    recovery = m.fs.RO.recovery_vol_phase[0, "Liq"].value
    # === Geometric parameters of RO units ===
    iscale.set_scaling_factor(m.fs.RO.length, 1e-1)
    iscale.set_scaling_factor(m.fs.RO.feed_side.channel_height, 1e3)
    iscale.set_variable_scaling_from_current_value(m.fs.RO.width)
    calculate_variable_from_constraint(m.fs.RO.area, m.fs.RO.eq_area)
    iscale.set_variable_scaling_from_current_value(m.fs.RO.area)
    calculate_variable_from_constraint(
        m.fs.RO.feed_side.area, m.fs.RO.feed_side.eq_area
    )
    iscale.set_variable_scaling_from_current_value(m.fs.RO.feed_side.area)
    iscale.set_scaling_factor(m.fs.RO.feed_side.spacer_porosity, 1)
    iscale.set_scaling_factor(m.fs.RO.A_comp[0, "H2O"], 1e12)
    iscale.set_scaling_factor(m.fs.RO.B_comp[0, "TDS"], 1e8)
    iscale.set_scaling_factor(m.fs.RO.mixed_permeate[0].pressure, 1e-5)
    iscale.set_scaling_factor(m.fs.RO.recovery_vol_phase[0, "Liq"], 2)

    # === Operating conditions ===
    hc = m.fs.RO.feed_side.channel_height.value  # channel height [m]
    width = m.fs.RO.width.value  # width of the module [m]
    length = m.fs.RO.length.value  # length of the module [m]
    q_feed = m.fs.feed.properties[0].flow_vol()  # feed flow rate [m³/s]
    porosity = m.fs.RO.feed_side.spacer_porosity.value  # dimensionless
    dh = 4 * porosity / (2 / hc + (1 - porosity) * 8 / hc)

    # Inlet and outlet velocities
    v_inlet = q_feed / (width * hc)  # inlet velocity [m/s]
    v_exit = v_inlet * (1 - recovery)  # exit velocity [m/s]

    # === Physical properties ===
    dens = m.fs.feed.properties[0].dens_mass_phase["Liq"].value  # kg/m³
    visc = m.fs.feed.properties[0].visc_d_phase["Liq"].value  # Pa·s
    diff = m.fs.feed.properties[0].diffus_phase_comp["Liq", "TDS"].value  # m²/s

    # === Inlet calculations ===
    re_in = dens * v_inlet * 2 * dh / visc
    sc_in = visc / (dens * diff)
    f_in = 0.42 + (189.3 / re_in)

    # === Exit calculations ===
    re_out = dens * v_exit * 2 * dh / visc
    sc_out = sc_in  # usually same properties
    f_out = 0.42 + (189.3 / re_out)

    # === Average values ===
    v_avg = 0.5 * (v_inlet + v_exit)
    re_avg = 0.5 * (re_in + re_out)
    f_avg = 0.5 * (f_in + f_out)
    sc_avg = 0.5 * (sc_in + sc_out)

    # === Pressure drop using Darcy-Weisbach (with averaged values) ===
    dp_in = -dens * v_inlet ** 2 / 2 * f_in / dh  # [Pa/m]
    dp_out = -dens * v_exit ** 2 / 2 * f_out / dh  # [Pa/m]
    dp_avg = 0.5 * (dp_in + dp_out)  # average pressure drop [Pa/m]

    # Set scaling for dimensionless numbers based on representative (inlet) values
    for v in m.fs.RO.feed_side.N_Sc_comp.values():
        iscale.set_scaling_factor(v, 1 / sc_in)
    for v in m.fs.RO.feed_side.N_Re.values():
        iscale.set_scaling_factor(v, 1 / re_avg)
    for v in m.fs.RO.feed_side.friction_factor_darcy.values():
        iscale.set_scaling_factor(v, 1 / f_avg)
    for v in m.fs.RO.feed_side.N_Sh_comp.values():
        sh_avg = re_avg ** 0.36 * sc_avg ** 0.36
        iscale.set_scaling_factor(v, 1 / sh_avg)

    for v in m.fs.RO.feed_side.dP_dx.values():
        iscale.set_scaling_factor(v, length / dp_avg)

    iscale.set_scaling_factor(m.fs.RO.deltaP, 1 / dp_avg)

    # Feed side (bulk)
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "H2O"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value
                              )
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_out[0].flow_mass_phase_comp["Liq", "H2O"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value / (1 - recovery)
                              )
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "TDS"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
                              )
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_out[0].flow_mass_phase_comp["Liq", "TDS"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
                              )

    # Feed side (interface)
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_interface[0, 0].flow_mass_phase_comp["Liq", "H2O"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value
                              )
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_interface[0, 1].flow_mass_phase_comp["Liq", "H2O"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value / (1 - recovery)
                              )
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_interface[0, 0].flow_mass_phase_comp["Liq", "TDS"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
                              )
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_interface[0, 1].flow_mass_phase_comp["Liq", "TDS"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
                              )

    # Permeate side
    iscale.set_scaling_factor(m.fs.RO.permeate_side[0, 0].flow_mass_phase_comp["Liq", "H2O"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value / recovery
                              )
    iscale.set_scaling_factor(m.fs.RO.permeate_side[0, 1].flow_mass_phase_comp["Liq", "H2O"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value / recovery
                              )

    iscale.set_scaling_factor(m.fs.RO.permeate_side[0, 0].flow_mass_phase_comp["Liq", "TDS"],
                              100 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value / recovery
                              )
    iscale.set_scaling_factor(m.fs.RO.permeate_side[0, 1].flow_mass_phase_comp["Liq", "TDS"],
                              100 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value / recovery
                              )

    iscale.set_scaling_factor(
        m.fs.RO.mixed_permeate[0].flow_mass_phase_comp["Liq", "H2O"],
        1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value / recovery
    )
    iscale.set_scaling_factor(
        m.fs.RO.mixed_permeate[0].flow_mass_phase_comp["Liq", "TDS"],
        100 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
    )

    iscale.set_scaling_factor(m.fs.RO.mass_transfer_phase_comp[0, "Liq", "H2O"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value / recovery)
    iscale.set_scaling_factor(m.fs.RO.mass_transfer_phase_comp[0, "Liq", "TDS"],
                              100 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value)

    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_in[0].flow_vol_phase["Liq"],
                              1 / m.fs.feed.properties[0].flow_vol_phase['Liq'].value)
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_out[0].flow_vol_phase["Liq"],
                              1 / m.fs.feed.properties[0].flow_vol_phase['Liq'].value / (1 - recovery))
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_interface[0, 0].flow_vol_phase["Liq"],
                              1 / m.fs.feed.properties[0].flow_vol_phase['Liq'].value)
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_interface[0, 1].flow_vol_phase["Liq"],
                              1 / m.fs.feed.properties[0].flow_vol_phase['Liq'].value / (1 - recovery))
    iscale.set_scaling_factor(m.fs.RO.permeate_side[0, 0].flow_vol_phase["Liq"],
                              1 / m.fs.feed.properties[0].flow_vol_phase['Liq'].value / recovery)
    iscale.set_scaling_factor(m.fs.RO.permeate_side[0, 0].flow_vol_phase["Liq"],
                              1 / m.fs.feed.properties[0].flow_vol_phase['Liq'].value / recovery)

    iscale.calculate_scaling_factors(m.fs.RO)

    # ERD unit
    iscale.set_scaling_factor(
        m.fs.ERD.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "H2O"],
        1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value / (1 - recovery)
    )
    iscale.set_scaling_factor(
        m.fs.ERD.control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"],
        1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value / (1 - recovery)
    )
    iscale.set_scaling_factor(
        m.fs.ERD.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "TDS"],
        100 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
    )
    iscale.set_scaling_factor(
        m.fs.ERD.control_volume.properties_out[0].flow_mass_phase_comp["Liq", "TDS"],
        100 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
    )

    iscale.set_scaling_factor(m.fs.ERD.control_volume.work, 1 / (
            op_pressure * m.fs.feed.properties[0].flow_vol_phase['Liq'].value * (1 - recovery) /
            m.fs.ERD.efficiency_pump[0].value))
    iscale.set_scaling_factor(m.fs.pump.work_fluid[0], 1 / (
            op_pressure * m.fs.feed.properties[0].flow_vol_phase['Liq'].value * (1 - recovery))
                              )
    iscale.set_scaling_factor(m.fs.ERD.control_volume.properties_in[0].pressure, 1 / op_pressure)
    iscale.set_scaling_factor(m.fs.ERD.control_volume.properties_out[0].pressure, 1 / 101325)
    iscale.set_scaling_factor(
        m.fs.ERD.control_volume.properties_out[0].flow_vol_phase["Liq"],
        1 / (m.fs.feed.properties[0].flow_vol_phase['Liq'].value * (1 - recovery))
    )

def initialize_system(m):
    m.fs.feed.initialize()
    propagate_state(m.fs.s01)

    m.fs.pump.initialize()
    propagate_state(m.fs.s02)

    m.fs.RO.initialize()
    propagate_state(m.fs.s03)
    propagate_state(m.fs.s04)

    m.fs.product.initialize()
    m.fs.ERD.initialize()
    propagate_state(m.fs.s05)
    m.fs.disposal.initialize()

    m.fs.costing.initialize()

def optimize_set_up(m):
    # add objective
    m.fs.objective = Objective(expr=m.fs.costing.LCOW)

    """
           Unfixes RO operating conditions and sets solver objective
               - Operating pressure: 1 - 83 bar
               - Crossflow velocity: 10 - 30 cm/s
               - Volumetric recovery: 30 - 80 %
               - Length: 6 - 10 m
               - Area and width were already unfixed
           """

    # RO operating pressure
    m.fs.pump.control_volume.properties_out[0].pressure.unfix()
    m.fs.pump.control_volume.properties_out[0].pressure.setub(8300000)
    m.fs.pump.control_volume.properties_out[0].pressure.setlb(100000)
    # m.fs.pump.deltaP.setlb(0)


    # RO inlet velocity
    m.fs.RO.feed_side.velocity[0, 0].unfix()
    m.fs.RO.feed_side.velocity[0, 0].setub(0.3)
    m.fs.RO.feed_side.velocity[0, 0].setlb(0.1)

    # RO length
    m.fs.RO.length.unfix()
    m.fs.RO.length.setub(10)
    m.fs.RO.length.setlb(6)

    # RO recovery
    m.fs.RO.recovery_vol_phase[0, "Liq"].unfix()
    m.fs.RO.recovery_vol_phase[0, "Liq"].setub(0.80)
    m.fs.RO.recovery_vol_phase[0, "Liq"].setlb(0.30)

    # Permeate salt concentration constraint
    m.fs.RO.mixed_permeate[0].conc_mass_phase_comp["Liq", "TDS"].setub(0.5)
    m.fs.RO.rejection_phase_comp[0, "Liq", "TDS"].setlb(0.99)
    m.fs.disposal.properties[0].conc_mass_phase_comp

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