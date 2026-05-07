'''
This is an RO flowsheet for debugging the failed initialization of the RO unit.
Different initialization values can be tested for treating brackish water.
'''
from pyomo.environ import (
    Var,
    Constraint,
    TransformationFactory,
    Reals,
    ConcreteModel,
    value,
    assert_optimal_termination,
    units as pyunits,
    Objective,
    Block,
    Param
)
from pyomo.network import Arc
# Ideas core components
from idaes.core import FlowsheetBlock
from idaes.core import UnitModelCostingBlock
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    set_scaling_factor,
    constraint_scaling_transform,
)

from idaes.core.util.model_statistics import degrees_of_freedom
from watertap.core.solvers import get_solver

from idaes.core.util.initialization import propagate_state, solve_indexed_blocks

from idaes.models.unit_models import Feed, Product
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# WaterTAP core components
from watertap.property_models.NaCl_prop_pack import NaClParameterBlock
from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.unit_models.pressure_changer import Pump
from watertap.costing import WaterTAPCosting
from math import log10

def main():
    m = build()
    set_operating_conditions(m)
    print('DOF after setting operating conditions: ', degrees_of_freedom(m))
    scale(m)
    initialize(m)
    print('DOF after initialization: ', degrees_of_freedom(m))
    solve(m)
    checking_scaling_values(m)
    # solve_bgw_initial_conditions(m,
    #                            mass_flow_NaCl=0.000035,
    #                            over_pressure=10,
    #                            NaCl_passage=0.0001,
    #                            area_unfix=True)
    # solve_bgw_initial_conditions(m,mass_flow_NaCl=0.00035),over_pressure=10) # Yes
    # solve_bgw_initial_conditions(m,mass_flow_NaCl=0.0035),over_pressure=1) # Yes
    solve_bgw_initial_conditions(m,mass_flow_NaCl=0.00035) #over_pressure=0.1) # No

    checking_scaling_values(m)
    return

def get_bgw_initial_conditions(mass_flow_NaCl=None):
    m = build()
    set_operating_conditions(m)
    scale(m)
    initialize(m)
    solve(m)
    solve_bgw_initial_conditions(m, mass_flow_NaCl=mass_flow_NaCl)
    return m

def build():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = NaClParameterBlock()
    # Unit processes
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.pump = Pump(property_package=m.fs.properties)
    m.fs.RO = ReverseOsmosis0D(
        property_package=m.fs.properties,
        has_pressure_change=True,
        pressure_change_type=PressureChangeType.calculated,
        mass_transfer_coefficient=MassTransferCoefficient.calculated,
        concentration_polarization_type=ConcentrationPolarizationType.calculated,
    )
    # Connections
    m.fs.feed_to_pump = Arc(source=m.fs.feed.outlet, destination=m.fs.pump.inlet)
    m.fs.pump_to_ro = Arc(source=m.fs.pump.outlet, destination=m.fs.RO.inlet)
    m.fs.permeate_to_product = Arc(source=m.fs.RO.permeate, destination=m.fs.product.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    # Costing
    m.fs.costing = WaterTAPCosting()
    m.fs.pump.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.RO.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.product.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.product.properties[0].flow_vol)

    return m

def set_operating_conditions(m):
    # Feed properties
    m.fs.feed.properties[0].temperature.fix(298.15)
    m.fs.feed.properties[0].pressure.fix(101325)
    m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'H2O'].fix(0.965)
    m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'NaCl'].fix(0.035)
    print('H2O flow rate: ', m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'H2O'].value)
    print('NaCl flow rate: ', m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'NaCl'].value)

    m.fs.feed.properties[0].conc_mass_phase_comp[...]
    m.fs.feed.properties[0].pressure_osm_phase[...]

    # Pump
    m.fs.pump.efficiency_pump[0].fix(0.8)

    operating_pressure = calculate_operating_pressure(feed_state_block=m.fs.feed.properties[0],
                                                         )
    print('Operating pressure is {} bar'.format(operating_pressure / 1e5))
    m.fs.pump.outlet.pressure[0].fix(operating_pressure)

    # fix RO values for initialization
    # m.fs.RO.feed_side.velocity[0, 0].fix(0.1)  # fixing inlet velocity
    m.fs.RO.area.fix(100)  # fixing stage area, but length and width are unfixed
    m.fs.RO.length.unfix()
    set_scaling_factor(m.fs.RO.length, 0.1)
    # m.fs.RO.width.unfix()
    m.fs.RO.width.fix(5)
    set_scaling_factor(m.fs.RO.width, 0.1)
    # RO operating conditions
    m.fs.RO.permeate.pressure[0].fix(101325)
    m.fs.RO.feed_side.channel_height.fix(1e-3)
    m.fs.RO.feed_side.spacer_porosity.fix(0.85)
    m.fs.RO.A_comp[0, 'H2O'].fix(1.51 / (3600 * 1000 * 1e5))
    m.fs.RO.B_comp[0, 'NaCl'].fix(0.126 / (3600 * 1000))

    # Costing
    m.fs.costing.electricity_cost.fix(0.1)

    return

def scale(m):
    # Feed
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp",
        1 / m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'H2O'].value,
        index=("Liq", "H2O"),
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp",
        m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'NaCl'],  # approximate scale
        index=("Liq", 'NaCl'),
    )

    # Pump
    set_scaling_factor(m.fs.pump.control_volume.work, 1e-4) # 1e-4
    set_scaling_factor(m.fs.pump.work_fluid[0], 1e-4) # 1e-4
    set_scaling_factor(m.fs.pump.control_volume.properties_out[0].pressure, 1e-5)
    set_scaling_factor(m.fs.pump.control_volume.properties_in[0].pressure, 1e-5)
    set_scaling_factor(
        m.fs.pump.control_volume.properties_out[0].flow_vol_phase["Liq"], 1e4
    )
    # RO
    set_scaling_factor(m.fs.RO.area, 1 / 100)
    set_scaling_factor(m.fs.RO.mass_transfer_phase_comp[0, "Liq", "NaCl"], 1e4)
    set_scaling_factor(
        m.fs.RO.feed_side.mass_transfer_term[0, "Liq", "NaCl"], 1e4 #1e4 # 1e6 when lower salt passage
    )

    return

def initialize(m):
    print('\nSimulation initialization ')
    solver = get_solver()  # get solver
    m.fs.feed.initialize(optarg=solver.options)
    # Feed to pump
    propagate_state(m.fs.feed_to_pump)
    m.fs.pump.initialize(optarg=solver.options)
    # pump to RO
    propagate_state(m.fs.pump_to_ro)
    m.fs.RO.initialize(optarg=solver.options)
    # RO to product
    propagate_state(m.fs.permeate_to_product)

    m.fs.costing.initialize()

    return

def calculate_operating_pressure(feed_state_block=None,
                                 over_pressure=0.15,
                                 water_recovery=0.4,
                                 NaCl_passage=0.1,
                                 solver=None):
    '''Want operating pressure to be (1+over_pressure) x osmotic pressure at given recovery'''
    if solver is None:
        solver=get_solver()

    t = ConcreteModel()  # create temporary m
    prop = feed_state_block.config.parameters
    t.brine = prop.build_state_block([0])

    # specify state block
    t.brine[0].flow_mass_phase_comp["Liq", "H2O"].fix(
        value(feed_state_block.flow_mass_phase_comp["Liq", "H2O"])
        * (1 - water_recovery)
    )
    t.brine[0].flow_mass_phase_comp["Liq", "NaCl"].fix(
        value(feed_state_block.flow_mass_phase_comp["Liq", "NaCl"]) * (1 - NaCl_passage)
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


def solve(m):
    solver = get_solver()
    print('\nSolve simulation')
    result = solver.solve(m, tee=False)
    assert_optimal_termination(result)
    return

def solve_bgw_initial_conditions(m,
                               mass_flow_NaCl=0.00035,
                               water_recovery=0.3,
                               NaCl_passage=0.01,
                               area_unfix=False,
                               display=False):
    print('\nSolve for BGW conditions')
    solver = get_solver()
    # update to bgw equivalent NaCl mass flow
    m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'NaCl'].fix(mass_flow_NaCl)
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp",
        1 / m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'NaCl'].value,  # approximate scale
        index=("Liq", 'NaCl'),
    )
    m.fs.feed.initialize(optarg=solver.options)
    osm_pressure = m.fs.feed.properties[0].pressure_osm_phase['Liq'].value
    # over_pressure_estimate = 1/mass_flow_NaCl*1e-3
    over_pressure_estimate = 10**(-log10(mass_flow_NaCl)-2.5)
    print('Overpressure estimate: {:.2f}'.format(over_pressure_estimate))
    try:
        operating_pressure = calculate_operating_pressure(feed_state_block=m.fs.feed.properties[0],
                                                      water_recovery=water_recovery,
                                                      over_pressure=over_pressure_estimate,#over_pressure,
                                                      NaCl_passage=NaCl_passage,
                                                      solver=solver)
    except:
        print('Failed to solve for operating pressure!')
        operating_pressure = osm_pressure*over_pressure_estimate
    # rescale_model(m)
    print('New osmotic pressure: {} bar'.format(osm_pressure/1e5))
    print('New operating pressure: {} bar'.format(operating_pressure/1e5))
    m.fs.pump.outlet.pressure[0].fix(operating_pressure)
    m.fs.RO.feed_side.velocity[0,0].unfix()
    m.fs.RO.width.unfix()
    m.fs.RO.feed_side.velocity[0,0].setub(0.25)
    if area_unfix:
        m.fs.RO.area.unfix()

    print('DOF: ', degrees_of_freedom(m))
    result = solver.solve(m, tee=False)
    assert_optimal_termination(result)
    if display:
        display_RO(m)

    print('\nOptimization to minimize LCOW')
    print('DOF: ', degrees_of_freedom(m))
    m.fs.RO.area.unfix()
    m.fs.pump.outlet.pressure[0].unfix()
    m.fs.RO.recovery_vol_phase[0, 'Liq'].fix(water_recovery)
    m.fs.objective = Objective(expr=m.fs.costing.LCOW)
    result = solver.solve(m, tee=False)
    assert_optimal_termination(result)
    if display:
        display_RO(m)

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


def checking_scaling_values(m):
    print('\nChecking scaling values')
    print(f'Pump work: {m.fs.pump.control_volume.work[0].value}')  # 1e-4
    print('Pump work fluid: {}'.format(m.fs.pump.work_fluid[0].value))  # 1
    print('Pump flow: {}'.format(m.fs.pump.control_volume.properties_out[0].flow_vol_phase["Liq"].value)) # 1
    # RO
    print('RO mass transfer: {}'.format(m.fs.RO.mass_transfer_phase_comp[0, "Liq", "NaCl"].value))# 1e4

def rescale_model(m):
    set_scaling_factor(m.fs.pump.control_volume.work, 1e-3)  # 1e-4
    set_scaling_factor(m.fs.pump.work_fluid[0], 1e-3)  # 1e-4
    set_scaling_factor(m.fs.RO.mass_transfer_phase_comp[0, "Liq", "NaCl"], 1e6)
    set_scaling_factor(
        m.fs.RO.feed_side.mass_transfer_term[0, "Liq", "NaCl"], 1e6  # 1e4 # 1e6 when lower salt passage
    )

if __name__ == '__main__':
    main()