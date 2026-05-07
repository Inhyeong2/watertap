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

    initialize_system(m)
    print('DOF after initialization: ', degrees_of_freedom(m))
    results = solve(m, tee=True)
    assert_optimal_termination(results)

    adaptive_salinity_sweep(m, target_salinity=0.5)

    # # Optimization after loop
    # optimize_set_up(m)
    # results = solve(m, tee=True)
    # assert_optimal_termination(results)

    display_system(m)
    display_design(m)

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
        m, flow_vol = 0.0438 * 10,  # 10 is MGD
        feed_salinity = 15, # * pyunits.kg / pyunits.m ** 3,
        overpressure = 0.15,
        max_recovery = 0.7,
        mix_recovery = 0.3,
        max_operating_pressure = 8300000
):
    # Feed
    m.fs.feed.properties[0].pressure.fix(101325)  # feed pressure [Pa]
    m.fs.feed.properties[0].temperature.fix(273.15 + 25)  # feed temperature [K]
    m.fs.feed.properties[0].pressure_osm_phase[...] #touching
    calculate_feed_state(m, feed_salinity, flow_vol)

    # Main pump
    m.fs.pump.efficiency_pump.fix(0.80)
    # m.fs.pump.control_volume.properties_out[0].pressure.setub(max_operating_pressure)

    op_pressure, wt_recovery = calculate_operating_pressure_and_recovery(
        feed_state_block=m.fs.feed.properties[0],
        over_pressure=overpressure,
        water_recovery=max_recovery,
        min_recovery=mix_recovery,
        NaCl_passage=0.01,
        max_pressure=max_operating_pressure, #Pa
        recovery_step=0.05,
        solver=None,
    )
    m.fs.pump.outlet.pressure[0].fix(op_pressure)
    print(f"Operating pressure set to {op_pressure * 1e-5:.2f} bar and recovery set to {wt_recovery * 100:.2f}%")

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
    m.fs.RO.recovery_vol_phase[0, "Liq"].fix(wt_recovery)
    m.fs.RO.recovery_vol_phase[0, "Liq"].unfix()
    # m.fs.RO.pprint()

    # ERD unit
    m.fs.ERD.efficiency_pump.fix(0.8)
    m.fs.ERD.control_volume.properties_out[0].pressure.fix(101325)  # Fix ERD outlet pressure to 1 atm

    # Costing
    m.fs.costing.electricity_cost.fix(0.1)

    return

def initialize_system(m):
    # Feed
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1/m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1/m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value, index=("Liq", "TDS"))
    iscale.set_scaling_factor(m.fs.feed.temperature[0], 1e-2)
    iscale.set_scaling_factor(m.fs.feed.pressure[0], 1e-5)

    m.fs.feed.initialize()
    propagate_state(m.fs.s01)

    # Main pump
    op_pressure = m.fs.pump.control_volume.properties_out[0].pressure.value
    iscale.set_scaling_factor(m.fs.pump.control_volume.work, 1 / (
                op_pressure * m.fs.feed.properties[0].flow_vol_phase['Liq'].value /
                m.fs.pump.efficiency_pump[0].value))
    iscale.set_scaling_factor(m.fs.pump.work_fluid[0], 1 / (
            op_pressure * m.fs.feed.properties[0].flow_vol_phase['Liq'].value))
    iscale.set_scaling_factor(m.fs.pump.control_volume.properties_out[0].pressure, 1 / op_pressure)
    iscale.set_scaling_factor(m.fs.pump.control_volume.properties_in[0].pressure, 1 / 101325)
    iscale.set_scaling_factor(
        m.fs.pump.control_volume.properties_out[0].flow_vol_phase["Liq"], 1 / m.fs.feed.properties[0].flow_vol_phase["Liq"].value
    )

    m.fs.pump.initialize()
    propagate_state(m.fs.s02)

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
    porosity =  m.fs.RO.feed_side.spacer_porosity.value # dimensionless
    dh = 4 * porosity / (2/hc + (1-porosity)*8/hc)

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
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_interface[0,0].flow_mass_phase_comp["Liq", "H2O"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value
                              )
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_interface[0,1].flow_mass_phase_comp["Liq", "H2O"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value / (1 - recovery)
                              )
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_interface[0,0].flow_mass_phase_comp["Liq", "TDS"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
                              )
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_interface[0,1].flow_mass_phase_comp["Liq", "TDS"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
                              )

    # Permeate side
    iscale.set_scaling_factor(m.fs.RO.permeate_side[0,0].flow_mass_phase_comp["Liq", "H2O"],
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
        m.fs.RO.mixed_permeate[0].flow_mass_phase_comp["Liq", "H2O"],  1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value / recovery
    )
    iscale.set_scaling_factor(
        m.fs.RO.mixed_permeate[0].flow_mass_phase_comp["Liq", "TDS"], 100 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
    )

    iscale.set_scaling_factor(m.fs.RO.mass_transfer_phase_comp[0, "Liq", "H2O"],
                              1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value / recovery)
    iscale.set_scaling_factor(m.fs.RO.mass_transfer_phase_comp[0, "Liq", "TDS"],
                              100 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value)

    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_in[0].flow_vol_phase["Liq"], 1/m.fs.feed.properties[0].flow_vol_phase['Liq'].value)
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_out[0].flow_vol_phase["Liq"], 1 / m.fs.feed.properties[0].flow_vol_phase['Liq'].value / (1-recovery))
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_interface[0, 0].flow_vol_phase["Liq"], 1 / m.fs.feed.properties[0].flow_vol_phase['Liq'].value)
    iscale.set_scaling_factor(m.fs.RO.feed_side.properties_interface[0, 1].flow_vol_phase["Liq"], 1 / m.fs.feed.properties[0].flow_vol_phase['Liq'].value / (1-recovery))
    iscale.set_scaling_factor(m.fs.RO.permeate_side[0, 0].flow_vol_phase["Liq"], 1 / m.fs.feed.properties[0].flow_vol_phase['Liq'].value / recovery)
    iscale.set_scaling_factor(m.fs.RO.permeate_side[0, 0].flow_vol_phase["Liq"], 1 / m.fs.feed.properties[0].flow_vol_phase['Liq'].value / recovery)

    iscale.calculate_scaling_factors(m.fs.RO)

    m.fs.RO.initialize()
    propagate_state(m.fs.s03)
    propagate_state(m.fs.s04)
    print(f"Area set to {m.fs.RO.area.value:.2f} m2")
    print(f"Width set to {m.fs.RO.width.value:.2f} m")
    print(f"Recovery set to {m.fs.RO.recovery_vol_phase[0, 'Liq'].value :.2f}")

    m.fs.product.initialize()

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

    m.fs.ERD.initialize()
    propagate_state(m.fs.s05)

    m.fs.disposal.initialize()

    m.fs.costing.initialize()

def solve_RO_initial_conditions(m,
                               feed_salinity=0.5,
                               water_recovery=0.7,
                               NaCl_passage=0.01,
                               display=False):
    print('\nSolve for treated wastewater conditions')
    solver = get_solver()
    # update to treated wastewater

    flow_vol = m.fs.feed.properties[0].flow_vol_phase['Liq'].value
    calculate_feed_state(m, feed_salinity, flow_vol)
    m.fs.properties.set_default_scaling("flow_mass_phase_comp",
                                        1 / m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value,
                                        index=("Liq", "TDS"))

    m.fs.feed.initialize(optarg=solver.options)
    propagate_state(m.fs.s01)
    # m.fs.feed.pprint()

    osm_pressure = m.fs.feed.properties[0].pressure_osm_phase['Liq'].value
    # over_pressure_estimate = 1/mass_flow_NaCl*1e-3
    over_pressure_estimate = 10**(-log10(m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value)-2.5)
    print('Overpressure estimate: {:.2f}'.format(over_pressure_estimate))
    try:
        operating_pressure, wt_recovery = calculate_operating_pressure_and_recovery(
            feed_state_block=m.fs.feed.properties[0],
            over_pressure=over_pressure_estimate,
            water_recovery=water_recovery,
            min_recovery=0.3,
            NaCl_passage=NaCl_passage,
            recovery_step=0.05,
            solver=None,
        )
    except:
        print('Failed to solve for operating pressure!')
        # operating_pressure = osm_pressure*over_pressure_estimate
        # wt_recovery = 0.5

        operating_pressure = osm_pressure*10
        wt_recovery = 0.5

    print('New osmotic pressure: {} bar'.format(osm_pressure/1e5))
    print('New operating pressure: {} bar'.format(operating_pressure/1e5))

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

    m.fs.pump.initialize()
    propagate_state(m.fs.s02)

    # RO unit
    # m.fs.RO.length.fix(8) # for iteration
    # m.fs.RO.feed_side.velocity[0, 0].fix(0.25) # for iteration

    m.fs.RO.recovery_vol_phase[0, "Liq"].fix(wt_recovery)
    m.fs.RO.recovery_vol_phase[0, "Liq"].unfix()
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

    m.fs.RO.initialize()
    propagate_state(m.fs.s03)
    propagate_state(m.fs.s04)

    m.fs.product.initialize()

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

    m.fs.ERD.initialize()
    propagate_state(m.fs.s05)

    m.fs.disposal.initialize()

    m.fs.costing.initialize()

    print('DOF: ', degrees_of_freedom(m))
    result = solver.solve(m, tee=False)
    assert_optimal_termination(result)
    if display:
        display_RO(m)

    # print('\nOptimization to minimize LCOW')
    # print('DOF: ', degrees_of_freedom(m))
    # m.fs.objective = Objective(expr=m.fs.costing.LCOW)
    # m.fs.pump.control_volume.properties_out[0].pressure.unfix()
    # m.fs.pump.deltaP.setlb(0)
    # m.fs.RO.length.unfix()
    # m.fs.RO.feed_side.velocity[0, 0].unfix()
    # m.fs.RO.feed_side.velocity[0, 0].setub(0.3)
    # m.fs.RO.feed_side.velocity[0, 0].setlb(0.1)
    #
    # result = solver.solve(m, tee=False)
    # assert_optimal_termination(result)
    # if display:
    #     display_RO(m)


def adaptive_salinity_sweep(m, target_salinity=0.5, initial_salinity=10, step_sizes=None):
    print("###### Adaptive Salinity Sweep #######")
    if step_sizes is None:
        step_sizes = [2, 1, 0.5, 0.25, 0.1]
    current_salinity = initial_salinity
    step_idx = 0

    while True:
        print(f"\n##### Trying salinity: {current_salinity} g/L (step: {step_sizes[step_idx]} g/L) #####")
        try:
            # Try solving the model with the current salinity
            solve_RO_initial_conditions(m, feed_salinity=current_salinity)

            # If we reached the target salinity, we're done
            if current_salinity == target_salinity:
                print(f"Successfully reached the target salinity: {target_salinity} g/L")
                break

            # Tentative next salinity
            tentative_salinity = current_salinity - step_sizes[step_idx]

            if tentative_salinity < target_salinity:
                # If step too big, reduce step size or try target directly
                if step_idx < len(step_sizes) - 1:
                    step_idx += 1
                    print(f"Next step too large to reach target. Reducing step size to {step_sizes[step_idx]}")
                else:
                    print(f"Trying final salinity at exact target: {target_salinity} g/L")
                    current_salinity = target_salinity
            else:
                current_salinity = tentative_salinity

        except Exception as e:
            print(f"Failed at salinity {current_salinity} g/L: {str(e)}")
            step_idx += 1
            if step_idx >= len(step_sizes):
                print("All step sizes exhausted. Stopping loop.")
                break
            else:
                print(f"Reducing step size to {step_sizes[step_idx]}")

    # Final check
    if current_salinity > target_salinity:
        raise RuntimeError(
            f"\n Could not reach the minimum target salinity ({target_salinity} g/L). "
            f"Final salinity tested: {current_salinity} g/L"
        )


def optimize_set_up(m):
    # add objective
    m.fs.objective = Objective(expr=m.fs.costing.LCOW)

    # unfix decision variables and add bounds
    # pump 1 and pump 2
    m.fs.pump.control_volume.properties_out[0].pressure.unfix()
    # m.fs.pump.control_volume.properties_out[0].pressure.setlb(10e5)
    # m.fs.pump.control_volume.properties_out[0].pressure.setub(80e5)
    m.fs.pump.deltaP.setlb(0)

    # RO
    m.fs.RO.length.unfix()
    m.fs.RO.length.setlb(6)
    m.fs.RO.length.setub(10)

    m.fs.RO.feed_side.velocity[0, 0].unfix()
    m.fs.RO.feed_side.velocity[0, 0].setub(0.3)
    m.fs.RO.feed_side.velocity[0, 0].setlb(0.1)

    m.fs.RO.recovery_vol_phase.setub(0.7)
    m.fs.RO.recovery_vol_phase.setlb(0.3)




def display_RO(m):
    print('NaCl mass flow: {} kg/s'.format(m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'NaCl'].value))
    print('Feed concentration: {} kg/m3'.format(m.fs.feed.properties[0].conc_mass_phase_comp['Liq', 'NaCl'].value))
    print('Feed osmotic pressure: {} bar'.format(m.fs.feed.properties[0].pressure_osm_phase['Liq'].value/1e5))
    print(f'Pump outlet pressure: {m.fs.pump.outlet.pressure[0].value/1e5} bar')
    print(f'Membrane area: {m.fs.RO.area.value} m2')
    print(f'Membrane length: {m.fs.RO.length.value} m')
    print(f'Membrane width: {m.fs.RO.width.value} m')
    print(f'Inlet velocity: {m.fs.RO.feed_side.velocity[0,0].value} m/s')
    print('Recovery: {:.2f} %'.format(m.fs.RO.recovery_vol_phase[0, 'Liq'].value*100))
    print(f'LCOW: {value(m.fs.costing.LCOW)} $/m3')
    print(f'SEC: {value(m.fs.costing.specific_energy_consumption)} kWh/m3')

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

def calculate_operating_pressure_and_recovery(
        feed_state_block=None,
        over_pressure=0.15,
        water_recovery=0.7,
        min_recovery=0.3,
        NaCl_passage=0.01,
        max_pressure=8.3e6,
        recovery_step=0.05,
        solver=None,
):
    """Want operating pressure to be (1+over_pressure) x osmotic pressure at given recovery,
    but reduce recovery if pressure exceeds max_pressure."""
    if solver is None:
        solver = get_solver()

    while water_recovery >= min_recovery:
        t = ConcreteModel()  # create temporary m
        prop = feed_state_block.config.parameters
        t.brine = prop.build_state_block([0])

        # specify state block
        t.brine[0].flow_mass_phase_comp["Liq", "H2O"].fix(
            value(feed_state_block.flow_mass_phase_comp["Liq", "H2O"])
            * (1 - water_recovery)
        )
        t.brine[0].flow_mass_phase_comp["Liq", "TDS"].fix(
            value(feed_state_block.flow_mass_phase_comp["Liq", "TDS"]) * (1 - NaCl_passage)
        )
        t.brine[0].pressure.fix(101325)
        t.brine[0].temperature.fix(value(feed_state_block.temperature))

        # calculate osmotic pressure
        t.brine[0].pressure_osm_phase
        results = solve_indexed_blocks(solver, [t.brine])
        assert_optimal_termination(results)

        op_pressure = value(t.brine[0].pressure_osm_phase["Liq"]) * (1 + over_pressure)

        if op_pressure <= max_pressure:
            return op_pressure, water_recovery
        else:
            water_recovery -= recovery_step  # reduce recovery and retry

    raise RuntimeError(
        f"Unable to find recovery meeting max_pressure={max_pressure / 1e6:.2f} MPa "
        f"(minimum tested recovery={water_recovery:.2f})."
    )

def calculate_operating_pressure(
    feed_state_block=None,
    over_pressure=0.15,
    water_recovery=0.7,
    NaCl_passage=0.01,
    solver=None,
):
    """Want operating pressure to be (1+over_pressure) x osmotic pressure at given recovery"""
    if solver is None:
        solver = get_solver()

    t = ConcreteModel()  # create temporary m
    prop = feed_state_block.config.parameters
    t.brine = prop.build_state_block([0])

    # specify state block
    t.brine[0].flow_mass_phase_comp["Liq", "H2O"].fix(
        value(feed_state_block.flow_mass_phase_comp["Liq", "H2O"])
        * (1 - water_recovery)
    )
    t.brine[0].flow_mass_phase_comp["Liq", "TDS"].fix(
        value(feed_state_block.flow_mass_phase_comp["Liq", "TDS"]) * (1 - NaCl_passage)
    )
    t.brine[0].pressure.fix(
        101325
    )  # valid when osmotic pressure is independent of hydraulic pressure
    t.brine[0].temperature.fix(value(feed_state_block.temperature))

    # calculate osmotic pressure
    # since properties are created on demand, we must touch the property to create it
    t.brine[0].pressure_osm_phase
    # solve state block
    results = solve_indexed_blocks(solver, [t.brine])
    assert_optimal_termination(results)

    return value(t.brine[0].pressure_osm_phase["Liq"]) * (1 + over_pressure)



def solve(blk, solver=None, tee=False, check_termination=True):
    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=tee)
    if check_termination:
        assert_optimal_termination(results)
    return results


if __name__ == "__main__":
    model = main()