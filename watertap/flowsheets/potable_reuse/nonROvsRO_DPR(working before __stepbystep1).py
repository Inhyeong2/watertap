"""
Non-RO DPR (O3/BAF/UF/Carbon_Adsorption/UV_AOP/Chlorination)
This module contains a zero-order representation of each unit in non-RO DPR.
"""

import os, math
import idaes.logger as idaeslog
from pyomo.environ import (
    assert_optimal_termination,
    check_optimal_termination,
    ConcreteModel,
    Block,
    Expression,
    Objective,
    value,
    Var,
    TransformationFactory,
    units as pyunits,
)
from pyomo.network import Arc, SequentialDecomposition
from pyomo.util.check_units import assert_units_consistent

from idaes.core import (
    FlowsheetBlock,
    MomentumBalanceType,
    UnitModelBlockData,
)

from watertap.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state

import idaes.core.util.scaling as iscale
from idaes.models.unit_models import (
    Mixer,
    Separator,
    Product,
    Translator,
    MomentumMixingType,
)

from idaes.core import UnitModelCostingBlock

from watertap.unit_models.pressure_exchanger import PressureExchanger
from watertap.unit_models.pressure_changer import Pump
from watertap.core.util.initialization import assert_degrees_of_freedom

from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.unit_models.reverse_osmosis_1D import ReverseOsmosis1D

from idaes.models.unit_models import Feed, Product
from watertap.core.zero_order_properties import WaterParameterBlock
from watertap.core.wt_database import Database
from watertap.unit_models.zero_order import (
    FeedZO,
    OzoneZO,
    BioActiveFiltrationZO,
    UltraFiltrationZO,
    GACZO,
    UVAOPZO,
    ChlorinationZO,
)
from watertap.flowsheets.nonRO_DPR.imaginary_separator_zo import ImaginarySeparatorZO

from watertap.costing.zero_order_costing import ZeroOrderCosting
from watertap.costing import WaterTAPCosting

# Set up logger
_log = idaeslog.getLogger(__name__)


def main(working_directory=None):
    # build, set, and initialize
    m, solute_list = build_nonRO(working_directory=working_directory)
    n = build_RO(working_directory=working_directory, solute_list=solute_list)
    set_operating_conditions(m, is_ro=False)  # non-RO DPR
    set_operating_conditions(n, is_ro=True)  # RO DPR
    assert_units_consistent(m)
    assert_units_consistent(n)

    initialize_system(m, is_ro=False)
    initialize_system(n, is_ro=True)
    assert_degrees_of_freedom(m, 0)
    assert_degrees_of_freedom(n, 0)

    results_nonRO = solve(m, checkpoint="solve flowsheet (non-RO DPR) after initializing system", tee=True)
    assert_optimal_termination(results_nonRO)
    results_RO = solve(n, checkpoint="solve flowsheet (RO DPR) after initializing system", tee=True)
    assert_optimal_termination(results_RO)

    add_costing(m, is_ro=False)
    add_costing(n, is_ro=True)

    initialize_costing(m, is_ro=False)
    initialize_costing(n, is_ro=True)
    assert_degrees_of_freedom(m, 0)  # ensures problem is square
    assert_degrees_of_freedom(n, 0)  # ensures problem is square

    optimize_operation(n)

    results_nonRO = solve(m, checkpoint="solve flowsheet (non-RO DPR) after costing", tee=True)
    assert_optimal_termination(results_nonRO)
    results_RO = solve(n, checkpoint="solve flowsheet (RO DPR) after costing", tee=True)
    assert_optimal_termination(results_RO)

    display_results_nonRO_DPR(m)
    display_results_RO_DPR(n)
    display_costing_nonRO_DPR(m)
    display_costing_RO_DPR(n)

    return m, n, results_nonRO, results_RO


def build_nonRO(working_directory=None):
    # flowsheet set up
    m = ConcreteModel()
    if working_directory == None:
        working_directory = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "data",
                "techno_economic",
            )
    elif working_directory == "local":
        working_directory = os.path.join(os.getcwd())
    else:
        raise TypeError(
            "Cannot find a working directory. working_directory should be either None or local."
        )
    m.db = Database(dbpath=working_directory)
    m.fs = FlowsheetBlock(dynamic=False)

    # Define solute list
    solute_list = [
        "cryptosporidium",
        "toc",
        "giardia_lamblia",
         "eeq",
         "total_coliforms_fecal_ecoli",
         "viruses_enteric",
         "tss",
         "tds"
    ]
    # define property packages
    m.fs.prop_zo = WaterParameterBlock(solute_list=solute_list)

    # define blocks
    non_RO = m.fs.non_RO = Block()

    # define flowsheet inlets and outlets
    m.fs.feed = FeedZO(property_package=m.fs.prop_zo) #water_sources.yaml

    # non-RO DPR components
    non_RO.Ozone = OzoneZO(property_package=m.fs.prop_zo, database=m.db)
    non_RO.BAF = BioActiveFiltrationZO(property_package=m.fs.prop_zo, database=m.db)
    non_RO.UF = UltraFiltrationZO(property_package=m.fs.prop_zo, database=m.db)
    non_RO.GAC = GACZO(property_package=m.fs.prop_zo, database=m.db)
    non_RO.UV_AOP = UVAOPZO(property_package=m.fs.prop_zo, database=m.db)
    non_RO.Cl = ChlorinationZO(property_package=m.fs.prop_zo, database=m.db)

    m.fs.byproduct_BAF = Product(property_package=m.fs.prop_zo)
    m.fs.byproduct_UF = Product(property_package=m.fs.prop_zo)
    m.fs.byproduct_GAC = Product(property_package=m.fs.prop_zo)
    m.fs.treated_nonRO = Product(property_package=m.fs.prop_zo)

    # connections (non-RO DPR)
    m.fs.s_non_RO_feed = Arc(source=m.fs.feed.outlet, destination=non_RO.Ozone.inlet)
    non_RO.s01 = Arc(source=non_RO.Ozone.treated, destination=non_RO.BAF.inlet)
    non_RO.s02 = Arc(source=non_RO.BAF.treated, destination=non_RO.UF.inlet)
    non_RO.s03 = Arc(source=non_RO.UF.treated, destination=non_RO.GAC.inlet)
    non_RO.s04 = Arc(source=non_RO.GAC.treated, destination=non_RO.UV_AOP.inlet)
    non_RO.s05 = Arc(source=non_RO.UV_AOP.treated, destination=non_RO.Cl.inlet)
    non_RO.s06 = Arc(source=non_RO.Cl.treated, destination=m.fs.treated_nonRO.inlet)

    m.fs.s01 = Arc(source=non_RO.BAF.byproduct, destination=m.fs.byproduct_BAF.inlet)
    m.fs.s02 = Arc(source=non_RO.UF.byproduct, destination=m.fs.byproduct_UF.inlet)
    m.fs.s03 = Arc(source=non_RO.GAC.byproduct, destination=m.fs.byproduct_GAC.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    return m, solute_list

def build_RO(working_directory=None, solute_list=None): # TODO: [defalut] solute_list is identical to that in build_nonRO
    # flowsheet set up
    m = ConcreteModel()
    if working_directory == None:
        working_directory = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "data",
                "techno_economic",
            )
    elif working_directory == "local":
        working_directory = os.path.join(os.getcwd())
    else:
        raise TypeError(
            "Cannot find a working directory. working_directory should be either None or local."
        )
    m.db = Database(dbpath=working_directory)
    m.fs = FlowsheetBlock(dynamic=False)

    # TODO: Should define specific solutes that cannot be perfectly removed by RO
    specific_solutes = ["toc", "tss"]

    # define property packages
    m.fs.prop_zo = WaterParameterBlock(solute_list=solute_list)
    m.fs.prop_ro = SeawaterParameterBlock()

    # define blocks
    RO_pre = m.fs.RO_pre = Block() # include zero-order chlorination and ultrafiltration
    RO_main = m.fs.RO_main = Block()

    RO_sep = m.fs.RO_sep = Block()
    RO_mix = m.fs.RO_mix = Block()

    #RO_post = m.fs.RO_post = Block() # include zero-order UV-AOP

    # define flowsheet inlets and outlets
    m.fs.feed = FeedZO(property_package=m.fs.prop_zo) #water_sources.yaml

    # RO DPR components
    RO_pre.Cl = ChlorinationZO(property_package=m.fs.prop_zo, database=m.db)
    RO_pre.UF = UltraFiltrationZO(property_package=m.fs.prop_zo, database=m.db)
    RO_sep.img_sep = ImaginarySeparatorZO(property_package=m.fs.prop_zo, database=m.db)
    m.fs.byproduct_UF = Product(property_package=m.fs.prop_zo)
    # m.fs.treated_RO = Product(property_package=m.fs.prop_ro)

    # TODO : delete it
    m.fs.byproduct_imgsep = Product(property_package=m.fs.prop_zo)

    RO_main.P1 = Pump(property_package=m.fs.prop_ro)
    RO_main.RO = ReverseOsmosis0D(
                 property_package=m.fs.prop_ro,
                 has_pressure_change=True,
                 pressure_change_type=PressureChangeType.calculated,
                 mass_transfer_coefficient=MassTransferCoefficient.calculated,
                 concentration_polarization_type=ConcentrationPolarizationType.calculated,
            )

    RO_main.RO.width.setub(2000)
    RO_main.RO.area.setub(20000)

    RO_main.S1 = Separator(property_package=m.fs.prop_ro, outlet_list=["P1", "PXR"])
    RO_main.M1 = Mixer(
        property_package=m.fs.prop_ro,
        momentum_mixing_type=MomentumMixingType.equality,
        inlet_list=["P1", "P2"],
    )
    RO_main.PXR = PressureExchanger(property_package=m.fs.prop_ro)
    RO_main.P2 = Pump(property_package=m.fs.prop_ro)

    m.fs.permeate = Product(property_package=m.fs.prop_ro)
    m.fs.brine = Product(property_package=m.fs.prop_ro)

    # translator blocks
    m.fs.tb_pre_main = Translator(
        inlet_property_package=m.fs.prop_zo, outlet_property_package=m.fs.prop_ro
    )

    @m.fs.tb_pre_main.Constraint(["H2O", "tds"])
    def eq_flow_mass_comp(blk, j):
        if j == "tds":
            return (
                    blk.properties_in[0].flow_mass_comp["tds"]
                    == blk.properties_out[0].flow_mass_phase_comp["Liq", "TDS"]
            )
        else:
            return (
                    blk.properties_in[0].flow_mass_comp["H2O"]
                    == blk.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]
            )

    #RO_post.UV_AOP = UVAOPZO(property_package=m.fs.prop_zo, database=m.db) ##### property_package check

    # connections (RO DPR)
    m.fs.s_feed_Cl = Arc(source=m.fs.feed.outlet, destination=RO_pre.Cl.inlet)
    RO_pre.s01 = Arc(source=RO_pre.Cl.treated, destination=RO_pre.UF.inlet)
    m.fs.s_UF_sep = Arc(source=RO_pre.UF.treated, destination=RO_sep.img_sep.inlet)
    m.fs.s_sep_tb1 = Arc(source=RO_sep.img_sep.treated, destination=m.fs.tb_pre_main.inlet)

    m.fs.s_UFby = Arc(source=RO_pre.UF.byproduct, destination=m.fs.byproduct_UF.inlet)
    m.fs.s_sepby = Arc(source=RO_sep.img_sep.byproduct, destination=m.fs.byproduct_imgsep.inlet)

    m.fs.s_tb1_RO = Arc(source=m.fs.tb_pre_main.outlet, destination=RO_main.S1.inlet)
    RO_main.s01 = Arc(source=RO_main.S1.P1, destination=RO_main.P1.inlet)
    RO_main.s02 = Arc(source=RO_main.P1.outlet, destination=RO_main.M1.P1)
    RO_main.s03 = Arc(source=RO_main.M1.outlet, destination=RO_main.RO.inlet)
    RO_main.s04 = Arc(source=RO_main.RO.retentate, destination=RO_main.PXR.brine_inlet)
    RO_main.s05 = Arc(source=RO_main.S1.PXR, destination=RO_main.PXR.feed_inlet)
    RO_main.s06 = Arc(source=RO_main.PXR.feed_outlet, destination=RO_main.P2.inlet)
    RO_main.s07 = Arc(source=RO_main.P2.outlet, destination=RO_main.M1.P2)

    m.fs.s_disposal = Arc(source=RO_main.PXR.brine_outlet, destination=m.fs.brine.inlet)
    m.fs.s_permeate = Arc(source=RO_main.RO.permeate, destination=m.fs.permeate.inlet)

    # RO arcs expansion
    TransformationFactory("network.expand_arcs").apply_to(m)

    # scaling
    m.fs.prop_ro.set_default_scaling("flow_mass_phase_comp", 1e-3, index=("Liq", "H2O"))
    m.fs.prop_ro.set_default_scaling("flow_mass_phase_comp", 1e-1, index=("Liq", "TDS"))

    # set unit model values
    iscale.set_scaling_factor(RO_main.P1.control_volume.work, 1e-6)
    iscale.set_scaling_factor(RO_main.P1.control_volume.deltaP, 1e-7)
    iscale.set_scaling_factor(
        RO_main.P1.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "H2O"],
        1e-1,
    )
    iscale.set_scaling_factor(RO_main.RO.area, 1e-4)
    iscale.set_scaling_factor(RO_main.P2.control_volume.work, 1e-5)
    iscale.set_scaling_factor(RO_main.PXR.feed_side.work, 1e-5)
    iscale.set_scaling_factor(RO_main.PXR.brine_side.work, 1e-5)

    # calculate and propagate scaling factors
    iscale.calculate_scaling_factors(m.fs.RO_main)
    return m

def set_operating_conditions(model, is_ro=False):
    # feed
    feed_temperature = (273.15 + 25) * pyunits.K
    feed_pressure = 101325 * pyunits.Pa
    feed_flow_vol = 0.0004101*100 * pyunits.m ** 3 / pyunits.s
    feed_conc_mass_toc = 0.005 * pyunits.kg / pyunits.m ** 3  # 1 mg/L = 0.001 kg/m3
    feed_conc_mass_tss = 0.01 * pyunits.kg / pyunits.m ** 3
    feed_conc_mass_tds = 0.1 * pyunits.kg / pyunits.m ** 3

    model.fs.feed.flow_vol[0].fix(feed_flow_vol)
    model.fs.feed.conc_mass_comp[0, "toc"].fix(feed_conc_mass_toc)
    model.fs.feed.conc_mass_comp[0, "tss"].fix(feed_conc_mass_tss)
    model.fs.feed.conc_mass_comp[0, "tds"].fix(feed_conc_mass_tds)
    model.fs.feed.conc_mass_comp[0, "cryptosporidium"].fix(1)
    model.fs.feed.conc_mass_comp[0, "giardia_lamblia"].fix(1)
    model.fs.feed.conc_mass_comp[0, "eeq"].fix(1)
    model.fs.feed.conc_mass_comp[0, "total_coliforms_fecal_ecoli"].fix(1)
    model.fs.feed.conc_mass_comp[0, "viruses_enteric"].fix(1)

    iscale.set_variable_scaling_from_current_value(model.fs.feed, descend_into=False)
    solve(model.fs.feed, checkpoint="solve feed block")

    if not is_ro:  # non-RO DPR
        non_RO = model.fs.non_RO

        # Unit processes in non-RO DPR
        non_RO.Ozone.load_parameters_from_database(use_default_removal=True)
        non_RO.BAF.load_parameters_from_database(use_default_removal=True)
        non_RO.UF.load_parameters_from_database(use_default_removal=True)
        non_RO.GAC.load_parameters_from_database(use_default_removal=True)
        non_RO.UV_AOP.load_parameters_from_database(use_default_removal=True)
        non_RO.Cl.load_parameters_from_database(use_default_removal=True)

    else:  # RO DPR
        RO_pre = model.fs.RO_pre
        RO_main = model.fs.RO_main
        RO_sep = model.fs.RO_sep

        RO_pre.Cl.load_parameters_from_database(use_default_removal=True)
        RO_pre.UF.load_parameters_from_database(use_default_removal=True)
        RO_sep.img_sep.load_parameters_from_database(use_default_removal=True)

        RO_main.P1.efficiency_pump.fix(0.80)
        operating_pressure = 70e5 * pyunits.Pa
        RO_main.P1.control_volume.properties_out[0].pressure.fix(operating_pressure)
        RO_main.RO.A_comp.fix(4.2e-12)  # membrane water permeability
        RO_main.RO.B_comp.fix(3.5e-8)  # membrane salt permeability
        RO_main.RO.feed_side.channel_height.fix(1e-3)  # channel height in membrane stage [m]
        RO_main.RO.feed_side.spacer_porosity.fix(0.97)  # spacer porosity in membrane stage [-]
        RO_main.RO.permeate.pressure[0].fix(feed_pressure)  # atmospheric pressure [Pa]
        RO_main.RO.feed_side.velocity[0, 0].fix(0.25)
        RO_main.RO.recovery_vol_phase[0, "Liq"].fix(0.5)
        model.fs.tb_pre_main.properties_out[0].temperature.fix(feed_temperature)
        model.fs.tb_pre_main.properties_out[0].pressure.fix(feed_pressure)

        # pressure exchanger
        RO_main.PXR.efficiency_pressure_exchanger.fix(0.95)
        # booster pump
        RO_main.P2.efficiency_pump.fix(0.80)

    return

def initialize_system(model, is_ro=False):
    # initialize feed
    solve(model.fs.feed, checkpoint="solve flowsheet after initializing feed")

    seq = SequentialDecomposition()
    seq.options.tear_set = []
    seq.options.iterLim = 1

    if not is_ro:  # non-RO DPR
        non_RO = model.fs.non_RO
        propagate_state(model.fs.s_non_RO_feed)

        # Initialize non-RO DPR
        seq.run(non_RO, lambda u: u.initialize())

    else:  # RO DPR
        RO_pre = model.fs.RO_pre
        RO_main = model.fs.RO_main
        RO_sep = model.fs.RO_sep

        # Propagate state from feed to Chlorination
        propagate_state(model.fs.s_feed_Cl)
        seq.run(RO_pre, lambda u: u.initialize())

        # Propagate state from UF to img_sep
        propagate_state(model.fs.s_UF_sep)
        RO_sep.img_sep.initialize()

        # Propagate state from img_sep to tb_pre_main
        propagate_state(model.fs.s_sep_tb1)

        # Translate flow from tb_pre_main to RO_main
        model.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "H2O"] = value(
            model.fs.tb_pre_main.properties_in[0].flow_mass_comp["H2O"]
        )
        model.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"] = value(
            model.fs.tb_pre_main.properties_in[0].flow_mass_comp["tds"]
        )

        propagate_state(model.fs.s_tb1_RO)

        # Initialize RO_main
        RO_main.RO.inlet.flow_mass_phase_comp[0, "Liq", "H2O"] = value(
            model.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]
        )
        RO_main.RO.inlet.temperature[0] = value(
            model.fs.tb_pre_main.properties_out[0].temperature
        )
        RO_main.RO.inlet.pressure[0] = value(
            RO_main.P1.control_volume.properties_out[0].pressure
        )

        RO_main.RO.initialize()
        solve(RO_main, checkpoint="solve flowsheet after initializing RO_main")

    return

def optimize_operation(m):
    """
    Unfixes RO operating conditions and sets solver objective
        - Operating pressure: 1 - 83 bar
        - Crossflow velocity: 10 - 30 cm/s
        - Membrane area: 50 - 5000 m2
        - Volumetric recovery: 10 - 75 %
    """
    RO_main = m.fs.RO_main

    # RO operating pressure
    RO_main.P1.control_volume.properties_out[0].pressure.unfix()
    RO_main.P1.control_volume.properties_out[0].pressure.setub(
        8300000
    )  # pressure vessel burst pressure
    RO_main.P1.control_volume.properties_out[0].pressure.setlb(100000)

    # RO inlet velocity
    RO_main.RO.feed_side.velocity[0, 0].unfix()
    RO_main.RO.feed_side.velocity[0, 0].setub(0.3)
    RO_main.RO.feed_side.velocity[0, 0].setlb(0.1)

    # RO membrane area
    RO_main.RO.area.unfix()
    RO_main.RO.area.setub(5000)
    RO_main.RO.area.setlb(50)

    # RO recovery - likely limited by operating pressure
    RO_main.RO.recovery_vol_phase[0, "Liq"].unfix()
    RO_main.RO.recovery_vol_phase[0, "Liq"].setub(0.99)
    RO_main.RO.recovery_vol_phase[0, "Liq"].setlb(0.1)

    # Permeate salt concentration constraint
    m.fs.permeate.properties[0].conc_mass_phase_comp["Liq", "TDS"].setub(0.5)
    m.fs.brine.properties[0].conc_mass_phase_comp
    m.fs.objective = Objective(expr=m.fs.LCOT)
    return

def solve(blk, solver=None, checkpoint=None, tee=False, fail_flag=True):
    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=tee)
    return results

def add_costing(model, is_ro=False):
    # Zero order costing
    source_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "nonRO_DPR_global_costing.yaml",
    )

    if not is_ro:  # non-RO DPR costing
        non_RO = model.fs.non_RO
        model.fs.zo_costing_nonRO = ZeroOrderCosting(case_study_definition=source_file)

        # non-RO DPR: cost of each unit process
        non_RO.Ozone.costing = UnitModelCostingBlock(flowsheet_costing_block=model.fs.zo_costing_nonRO)
        non_RO.BAF.costing = UnitModelCostingBlock(flowsheet_costing_block=model.fs.zo_costing_nonRO)
        non_RO.UF.costing = UnitModelCostingBlock(flowsheet_costing_block=model.fs.zo_costing_nonRO)
        non_RO.GAC.costing = UnitModelCostingBlock(flowsheet_costing_block=model.fs.zo_costing_nonRO)
        non_RO.UV_AOP.costing = UnitModelCostingBlock(flowsheet_costing_block=model.fs.zo_costing_nonRO)
        non_RO.Cl.costing = UnitModelCostingBlock(flowsheet_costing_block=model.fs.zo_costing_nonRO)

        # Aggregate unit level costs and calculate overall process costs
        model.fs.zo_costing_nonRO.cost_process()

        feed_flowrate = model.fs.feed.flow_vol[0]
        model.fs.zo_costing_nonRO.add_electricity_intensity(feed_flowrate)
        model.fs.specific_energy_intensity = Expression(
            expr=(model.fs.zo_costing_nonRO.electricity_intensity),
            doc="Specific energy consumption of the non-RO DPR treatment train on a feed flowrate basis [kWh/m3]",
        )

        # Water recovery revenue
        # TODO: check recovered_water_cost for BAF, UF, and GAC byproduct (backwash); currently this value is based on dye_desalination yaml file
        model.fs.water_recovery_revenue = Expression(
            expr=(
                -1 * model.fs.zo_costing_nonRO.utilization_factor
                * model.fs.zo_costing_nonRO.recovered_water_cost
                * pyunits.convert(
                    model.fs.byproduct_BAF.properties[0].flow_vol
                    + model.fs.byproduct_UF.properties[0].flow_vol
                    + model.fs.byproduct_GAC.properties[0].flow_vol,
                    to_units=pyunits.m**3 / model.fs.zo_costing_nonRO.base_period,
                )
            ),
            doc="Savings from water recovered (BAF, UF, and GAC backwashed water (byproduct)) back to the plant",
        )

        # Combine results from costing packages and calculate overall metrics
        @model.fs.Expression(doc="Total capital cost of the non-RO DPR treatment train")
        def total_capital_cost(b):
            return pyunits.convert(
                model.fs.zo_costing_nonRO.total_capital_cost, to_units=pyunits.USD_2020
            )

        @model.fs.Expression(doc="Total operating cost of the non-RO DPR treatment train")
        def total_operating_cost(b):
            return pyunits.convert(
                model.fs.zo_costing_nonRO.total_fixed_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year,
            ) + pyunits.convert(
                model.fs.zo_costing_nonRO.total_variable_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year,
            )

        @model.fs.Expression(doc="Total cost of water recovery")
        def total_externalities(b): # can be either - or +
            return pyunits.convert(model.fs.water_recovery_revenue, to_units=pyunits.USD_2020 / pyunits.year)

        @model.fs.Expression(
            doc="Levelized cost of the non-RO DPR treatment with respect to volumetric feed flow"
        )
        def LCOT(b):
            return (
                b.total_capital_cost * b.zo_costing_nonRO.capital_recovery_factor
                + b.total_operating_cost
                + b.total_externalities
            ) / (
                pyunits.convert(
                    b.feed.properties[0].flow_vol,
                    to_units=pyunits.m ** 3 / pyunits.year,
                )
                * b.zo_costing_nonRO.utilization_factor
            )

        @model.fs.Expression(
            doc="Levelized cost of the non-RO DPR treatment with respect to volumetric feed flow, not including externalities (water recovery)"
        )
        def LCOT_wo_revenue(b):
            return (
                    b.total_capital_cost * b.zo_costing_nonRO.capital_recovery_factor
                    + b.total_operating_cost
            ) / (
                    pyunits.convert(
                        b.feed.properties[0].flow_vol,
                        to_units=pyunits.m ** 3 / pyunits.year,
                    )
                    * b.zo_costing_nonRO.utilization_factor
            )

        @model.fs.Expression(
            doc="Levelized cost of water (non-RO DPR) with respect to volumetric treated water flow"
        )
        def LCOW(b):
            return (
                    b.total_capital_cost * b.zo_costing_nonRO.capital_recovery_factor
                    + b.total_operating_cost
                    + b.total_externalities
            ) / (
                    pyunits.convert(
                        b.treated_nonRO.properties[0].flow_vol,
                        to_units=pyunits.m ** 3 / pyunits.year,
                    )
                    * b.zo_costing_nonRO.utilization_factor
            )

        @model.fs.Expression(
            doc="Levelized cost of water (non-RO DPR) with respect to volumetric treated water flow, not including externalities (water recovery)"
        )
        def LCOW_wo_revenue(b):
            return (
                b.total_capital_cost * b.zo_costing_nonRO.capital_recovery_factor
                + b.total_operating_cost
            ) / (
                pyunits.convert(
                    b.treated_nonRO.properties[0].flow_vol,
                    to_units=pyunits.m ** 3 / pyunits.year,
                )
                * b.zo_costing_nonRO.utilization_factor
            )

    else:  # RO DPR costing
        RO_pre = model.fs.RO_pre
        model.fs.zo_costing_RO_pre = ZeroOrderCosting(case_study_definition=source_file)
        model.fs.ro_costing = WaterTAPCosting()

        # RO DPR: cost of each unit process (RO_pre)
        RO_pre.Cl.costing = UnitModelCostingBlock(flowsheet_costing_block=model.fs.zo_costing_RO_pre)
        RO_pre.UF.costing = UnitModelCostingBlock(flowsheet_costing_block=model.fs.zo_costing_RO_pre)

        # Aggregate unit level costs and calculate overall process costs
        model.fs.zo_costing_RO_pre.cost_process()

        feed_flowrate = model.fs.feed.flow_vol[0]
        model.fs.zo_costing_RO_pre.add_electricity_intensity(feed_flowrate)

        # RO DPR: cost of RO (RO_main)
        # RO equipment is costed using more detailed costing package
        RO_main = model.fs.RO_main
        RO_main.P1.costing = UnitModelCostingBlock(
            flowsheet_costing_block=model.fs.ro_costing,
            costing_method_arguments={"cost_electricity_flow": True},
        )
        RO_main.RO.costing = UnitModelCostingBlock(
            flowsheet_costing_block=model.fs.ro_costing
        )

        RO_main.M1.costing = UnitModelCostingBlock(
            flowsheet_costing_block=model.fs.ro_costing
        )
        RO_main.PXR.costing = UnitModelCostingBlock(
            flowsheet_costing_block=model.fs.ro_costing
        )
        RO_main.P2.costing = UnitModelCostingBlock(
            flowsheet_costing_block=model.fs.ro_costing,
            costing_method_arguments={"cost_electricity_flow": True},
        )

        model.fs.ro_costing.electricity_cost = value(model.fs.zo_costing_RO_pre.electricity_cost)
        model.fs.ro_costing.base_currency = pyunits.USD_2020

        model.fs.ro_costing.cost_process()

        model.fs.ro_costing.add_specific_energy_consumption(feed_flowrate)

        model.fs.specific_energy_intensity = Expression(
            expr=(
                    model.fs.zo_costing_RO_pre.electricity_intensity
                    + model.fs.ro_costing.specific_energy_consumption
            ),
            doc="Specific energy consumption of the RO DPR treatment train on a feed flowrate basis [kWh/m3]",
        )

        # Water recovery revenue
        # TODO: check recovered_water_cost for UF byproduct (backwash); currently this value is based on dye_desalination yaml file
        model.fs.water_recovery_revenue = Expression(
            expr=(
                    -1 * model.fs.zo_costing_RO_pre.utilization_factor
                    * model.fs.zo_costing_RO_pre.recovered_water_cost
                    * pyunits.convert(model.fs.byproduct_UF.properties[0].flow_vol,
                to_units=pyunits.m ** 3 / model.fs.zo_costing_RO_pre.base_period,
            )
            ),
            doc="Savings from water recovered (UF backwashed water (byproduct)) back to the plant",
        )

        # Combine results from costing packages and calculate overall metrics
        @model.fs.Expression(doc="Total capital cost of the RO DPR treatment train")
        def total_capital_cost(b):
            return pyunits.convert(
                model.fs.zo_costing_RO_pre.total_capital_cost, to_units=pyunits.USD_2020
            ) + pyunits.convert(
                model.fs.ro_costing.total_capital_cost, to_units=pyunits.USD_2020
            )

        @model.fs.Expression(doc="Total operating cost of the RO DPR treatment train")
        def total_operating_cost(b):
            return pyunits.convert(
                model.fs.zo_costing_RO_pre.total_fixed_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year,
            ) + pyunits.convert(
                model.fs.zo_costing_RO_pre.total_variable_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year,
            ) + pyunits.convert(
                model.fs.ro_costing.total_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year,
            )

        @model.fs.Expression(doc="Total cost of water recovery")
        def total_externalities(b): # can be either - or +
            return pyunits.convert(model.fs.water_recovery_revenue, to_units=pyunits.USD_2020 / pyunits.year)

        @model.fs.Expression(
            doc="Levelized cost of the RO DPR treatment with respect to volumetric feed flow"
        )
        def LCOT(b):
            return (
                b.total_capital_cost * b.zo_costing_RO_pre.capital_recovery_factor
                + b.total_operating_cost
                + b.total_externalities
            ) / (
                pyunits.convert(
                    b.feed.properties[0].flow_vol,
                    to_units=pyunits.m ** 3 / pyunits.year,
                )
                * b.zo_costing_RO_pre.utilization_factor
            )

        @model.fs.Expression(
            doc="Levelized cost of the RO DPR treatment with respect to volumetric feed flow, not including externalities"
        )
        def LCOT_wo_revenue(b):
            return (
                    b.total_capital_cost * b.zo_costing_RO_pre.capital_recovery_factor
                    + b.total_operating_cost
            ) / (
                    pyunits.convert(
                        b.feed.properties[0].flow_vol,
                        to_units=pyunits.m ** 3 / pyunits.year,
                    )
                    * b.zo_costing_RO_pre.utilization_factor
            )

        @model.fs.Expression(
            doc="Levelized cost of water (RO DPR) with respect to volumetric treated water flow"
        )
        def LCOW(b):
            return (
                b.total_capital_cost * b.zo_costing_RO_pre.capital_recovery_factor
                + b.total_operating_cost
                + b.total_externalities
            ) / (
                pyunits.convert(
                    b.permeate.properties[0].flow_vol,
                    to_units=pyunits.m ** 3 / pyunits.year,
                )
                * b.zo_costing_RO_pre.utilization_factor
            )

        @model.fs.Expression(
            doc="Levelized cost of water (RO DPR) with respect to volumetric treated water flow, not including externalities"
        )
        def LCOW_wo_revenue(b):
            return (
                b.total_capital_cost * b.zo_costing_RO_pre.capital_recovery_factor
                + b.total_operating_cost
            ) / (
                pyunits.convert(
                    b.permeate.properties[0].flow_vol,
                    to_units=pyunits.m ** 3 / pyunits.year,
                )
                * b.zo_costing_RO_pre.utilization_factor
            )

    return


def initialize_costing(model, is_ro=False):
    if not is_ro:
        model.fs.zo_costing_nonRO.initialize()
    else:
        model.fs.zo_costing_RO_pre.initialize()
        model.fs.ro_costing.initialize()
    return

def display_results_nonRO_DPR(m):
    print("\n----------Unit models in non-RO DPR ----------")
    unit_models = {name: block for name, block in m.fs.non_RO.component_map(Block).items() if
                   isinstance(block, UnitModelBlockData)}
    for unit_name, unit_block in unit_models.items():
        unit_block.report()

    # Flow rate
    print("\nFeed flow rate in each unit process")
    print(
        f"Feed flow rate: {value(pyunits.convert(m.fs.feed.flow_vol[0], to_units=pyunits.m ** 3 / pyunits.hr)): .3f} m3/hr")
    for unit_name, unit_block in unit_models.items():
        unit_flow_rate = value(pyunits.convert(unit_block.properties_in[0].flow_vol, to_units=pyunits.m ** 3 / pyunits.hr))
        print(f"Feed flow rate in {unit_name}: {unit_flow_rate: .3f} m3/hr")

    # TOC removal
    print("\nTOC concentration in each unit process")
    print(f"TOC in feed: {value(pyunits.convert(m.fs.feed.conc_mass_comp[0, 'toc'], to_units=pyunits.mg / pyunits.L)): .3f} mg/L")
    for unit_name, unit_block in unit_models.items():
        toc_in = value(pyunits.convert(unit_block.properties_in[0].conc_mass_comp["toc"], to_units=pyunits.mg / pyunits.L))
        toc_out = value(pyunits.convert(unit_block.properties_treated[0].conc_mass_comp["toc"], to_units=pyunits.mg / pyunits.L))
        print(f"TOC before {unit_name}: {toc_in: .3f} mg/L")
        print(f"TOC after {unit_name}: {toc_out: .3f} mg/L")

        toc_in_flow_mass = value(pyunits.convert(unit_block.properties_in[0].flow_vol, to_units=pyunits.L / pyunits.s)) * toc_in
        toc_out_flow_mass = value(pyunits.convert(unit_block.properties_treated[0].flow_vol, to_units=pyunits.L / pyunits.s)) * toc_out

        # Check if TOC removal is defined in the unit's YAML file
        if "toc" in unit_block.config.database.get_unit_operation_parameters(unit_block._tech_type)["removal_frac_mass_comp"]:
            removal_toc = 100 * (1 - toc_out_flow_mass / toc_in_flow_mass)
            print(f"% removal of TOC by {unit_name}: {removal_toc:.2f}%")
        else:
            print(f"{unit_name} does not remove TOC.")
    print(
        f"total % TOC concentration removal: {100 * (1 - value(m.fs.treated_nonRO.properties[0].conc_mass_comp['toc']) / value(m.fs.feed.conc_mass_comp[0, 'toc'])): .2f}%")

    # TSS removal
    print("\nTSS concentration in each unit process")
    print(f"TSS in feed: {value(pyunits.convert(m.fs.feed.conc_mass_comp[0, 'tss'], to_units=pyunits.mg / pyunits.L)): .3f} mg/L")
    for unit_name, unit_block in unit_models.items():
        tss_in = value(pyunits.convert(unit_block.properties_in[0].conc_mass_comp["tss"], to_units=pyunits.mg / pyunits.L))
        tss_out = value(pyunits.convert(unit_block.properties_treated[0].conc_mass_comp["tss"], to_units=pyunits.mg / pyunits.L))

        print(f"TSS before {unit_name}: {tss_in: .3f} mg/L")
        print(f"TSS after {unit_name}: {tss_out: .3f} mg/L")
    print(
        f"total % TSS concentration removal: {100 * (1 - value(m.fs.treated_nonRO.properties[0].conc_mass_comp['tss']) / value(m.fs.feed.conc_mass_comp[0, 'tss'])): .2f}%")

    # Cryptosporidium % removal
    print("\nCryptosporidium % removal in each unit process")
    for unit_name, unit_block in unit_models.items():
        crypto_in = value(pyunits.convert(unit_block.properties_in[0].conc_mass_comp["cryptosporidium"], to_units=pyunits.mg / pyunits.L))
        crypto_out = value(pyunits.convert(unit_block.properties_treated[0].conc_mass_comp["cryptosporidium"], to_units=pyunits.mg / pyunits.L))

        crypto_in_flow_mass = value(pyunits.convert(unit_block.properties_in[0].flow_vol, to_units=pyunits.L / pyunits.s)) * crypto_in
        crypto_out_flow_mass = value(pyunits.convert(unit_block.properties_treated[0].flow_vol, to_units=pyunits.L / pyunits.s)) * crypto_out

        # Check if Cryptosporidium removal is defined in the unit's YAML file
        if "cryptosporidium" in unit_block.config.database.get_unit_operation_parameters(unit_block._tech_type)[
            "removal_frac_mass_comp"]:
            removal_crypto = 100 * (1 - crypto_out_flow_mass / crypto_in_flow_mass)
            print(f"% removal of Cryptosporidium by {unit_name}: {removal_crypto:.6f}%")
    print(
        f"Total log removal ratio of Cryptosporidium (concentration ratio): {-math.log10(value(m.fs.treated_nonRO.properties[0].conc_mass_comp['cryptosporidium']) / value(m.fs.feed.conc_mass_comp[0, 'cryptosporidium'])): .2f} log")

    # Overall system recovery
    print("\n----------System Recovery (non-RO DPR)----------\n")
    sys_water_recovery = (
            m.fs.treated_nonRO.properties[0].flow_mass_comp["H2O"]()
            / m.fs.feed.flow_mass_comp[0, "H2O"]()
    )
    print(f"System water recovery: {sys_water_recovery * 100 : .3f}%")

    solute_list = m.fs.prop_zo.solute_set
    for solute in solute_list:
        sys_recovery = (
                m.fs.treated_nonRO.properties[0].flow_mass_comp[solute]()
                / m.fs.feed.flow_mass_comp[0, solute]()
        )
        print(f"System {solute} recovery: {sys_recovery * 100: .9f}%")


def display_results_RO_DPR(m):
    print("\n----------Unit models in RO DPR ----------")
    unit_models = {name: block for name, block in m.fs.RO_pre.component_map(Block).items() if
                   isinstance(block, UnitModelBlockData)}
    for unit_name, unit_block in unit_models.items():
        unit_block.report()
    m.fs.RO_sep.img_sep.report()

    # Flow rate
    print("\nFeed flow rate in each unit process")
    print(f"Feed flow rate: {value(pyunits.convert(m.fs.feed.flow_vol[0], to_units=pyunits.m ** 3 / pyunits.hr)): .3f} m3/hr")
    for unit_name, unit_block in unit_models.items():
        unit_flow_rate = value(
            pyunits.convert(unit_block.properties_in[0].flow_vol, to_units=pyunits.m ** 3 / pyunits.hr))
        print(f"Feed flow rate in {unit_name}: {unit_flow_rate: .3f} m3/hr")
    print(f"Feed flow rate in RO: {value(pyunits.convert(m.fs.RO_main.RO.feed_side.properties[0, 0].flow_vol_phase['Liq'], to_units=pyunits.m ** 3 / pyunits.hr)): .3f} m3/hr")
    print(f"Flow rate of RO permeate: {value(pyunits.convert(m.fs.RO_main.RO.mixed_permeate[0].flow_vol_phase['Liq'], to_units=pyunits.m ** 3 / pyunits.hr)): .3f} m3/hr")

    # TOC removal
    print("\nTOC concentration in each unit process")
    print(f"TOC in feed: {value(pyunits.convert(m.fs.feed.conc_mass_comp[0, 'toc'], to_units=pyunits.mg / pyunits.L)): .3f} mg/L")
    for unit_name, unit_block in unit_models.items():
        toc_in = value(pyunits.convert(unit_block.properties_in[0].conc_mass_comp["toc"], to_units=pyunits.mg / pyunits.L))
        toc_out = value(pyunits.convert(unit_block.properties_treated[0].conc_mass_comp["toc"], to_units=pyunits.mg / pyunits.L))
        print(f"TOC before {unit_name}: {toc_in: .3f} mg/L")
        print(f"TOC after {unit_name}: {toc_out: .3f} mg/L")

        toc_in_flow_mass = value(pyunits.convert(unit_block.properties_in[0].flow_vol, to_units=pyunits.L / pyunits.s)) * toc_in
        toc_out_flow_mass = value(pyunits.convert(unit_block.properties_treated[0].flow_vol, to_units=pyunits.L / pyunits.s)) * toc_out

        # Check if TOC removal is defined in the unit's YAML file
        if "toc" in unit_block.config.database.get_unit_operation_parameters(unit_block._tech_type)["removal_frac_mass_comp"]:
            removal_toc = 100 * (1 - toc_out_flow_mass / toc_in_flow_mass)
            print(f"% removal of TOC by {unit_name}: {removal_toc:.2f}%")
        else:
            print(f"{unit_name} does not remove TOC.")

    # print(
    #     f"TOC before imaginary separator: {value(pyunits.convert(m.fs.RO_sep.img_sep.properties_in[0].conc_mass_comp['toc'], to_units=pyunits.mg / pyunits.L)): .3f} mg/L")
    # print(
    #     f"TOC after imaginary separator: {value(pyunits.convert(m.fs.RO_sep.img_sep.properties_treated[0].conc_mass_comp['toc'], to_units=pyunits.mg / pyunits.L)): .3f} mg/L")

    # print(
    #     f"total % TOC concentration removal: {100 * (1 - value(m.fs.treated_nonRO.properties[0].conc_mass_comp['toc']) / value(m.fs.feed.conc_mass_comp[0, 'toc'])): .2f}%")


    # Overall system recovery
    print("\n----------System Recovery (RO DPR)----------\n")
    sys_water_recovery = (
            m.fs.permeate.properties[0].flow_mass_phase_comp["Liq", "H2O"]()
            / m.fs.feed.flow_mass_comp[0, "H2O"]()
    )
    print(f"System water recovery: {sys_water_recovery * 100 : .3f}%")
    sys_tds_recovery = (
            m.fs.permeate.properties[0].flow_mass_phase_comp["Liq", "TDS"]()
            / m.fs.feed.flow_mass_comp[0, "tds"]()
    )
    print(f"System TDS recovery: {sys_tds_recovery * 100: .3f}%")

    # solute_list = m.fs.prop_zo.solute_set
    # for solute in solute_list:
    #     sys_recovery = (
    #             m.fs.treated_RO.properties[0].flow_mass_comp[solute]()
    #             / m.fs.feed.flow_mass_comp[0, solute]()
    #     )
    #     print(f"System {solute} recovery: {sys_recovery * 100: .9f}%")


def display_costing_nonRO_DPR(m):
    # Capex
    print("\n----------System costing metrics (non-RO DPR)----------\n")
    capex = value(pyunits.convert(m.fs.total_capital_cost, to_units=pyunits.MUSD_2020))
    print(f"Total Capital Cost: {capex:.4f} M$")

    print("\n----------Unit Capital Costs----------")
    total_unit_capex_cal = 0
    for u in m.fs.zo_costing_nonRO._registered_unit_costing:
        unit_name = u.parent_block().local_name
        print(f"{unit_name} capital cost: {value(pyunits.convert(u.capital_cost, to_units=pyunits.USD_2020)):.3f} $")
        total_unit_capex_cal += value(pyunits.convert(u.capital_cost, to_units=pyunits.USD_2020))
    print(f"(Calculated) Sum of capital costs of unit processes: {total_unit_capex_cal / 1e6:.4f} M$")
    print("---------------------------")

    # Opex
    opex = value(pyunits.convert(m.fs.total_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))
    total_fixed_operating_cost = value(pyunits.convert(m.fs.zo_costing_nonRO.total_fixed_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))
    total_variable_operating_cost = value(pyunits.convert(m.fs.zo_costing_nonRO.total_variable_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))

    print(f"\nTotal Operating Cost (Cop,tot): {opex:.4f} M$/year")
    print(f"Total fixed operating cost (Cop,fix): {total_fixed_operating_cost:.4f} M$/year")
    print(f"Total variable operating cost (Cop,fix): {total_variable_operating_cost:.4f} M$/year\n")

    # Comparison with calculated values
    total_fixed_operating_cost_cal = value(pyunits.convert(m.fs.zo_costing_nonRO.aggregate_fixed_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))
    total_variable_operating_cost_vop_cal = value(pyunits.convert(m.fs.zo_costing_nonRO.aggregate_variable_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))
    # print("Used flows:")
    # for flow in m.fs.zo_costing_nonRO.used_flows:
    #    print(flow)
    total_flow_cost_cal = value(
        pyunits.convert(
            sum(m.fs.zo_costing_nonRO.aggregate_flow_costs[flow] for flow in m.fs.zo_costing_nonRO.used_flows)
            * m.fs.zo_costing_nonRO.utilization_factor,
            to_units=pyunits.MUSD_2020 / pyunits.year
        )
    )
    total_operating_cost_cal = total_fixed_operating_cost_cal + total_variable_operating_cost_vop_cal + total_flow_cost_cal
    total_variable_operating_cost_cal = total_variable_operating_cost_vop_cal + total_flow_cost_cal

    print(f"(Calculated) Total Operating Cost (Cop,tot): {total_operating_cost_cal:.4f} M$")
    print(f"(Calcualted) Total fixed operating cost (Cop,fix): {total_fixed_operating_cost_cal:.4f} M$/year")
    print(f"(Calculated) Total variable operating cost (Cop,fix): {total_variable_operating_cost_cal:.4f} M$/year")
    print(f"(Calculated) Total variable operating cost from unit models (Cvop,u): {total_variable_operating_cost_vop_cal:.4f} M$/year")
    print(f"(Calculated) Total flow cost (futil*Cflow,tot): {total_flow_cost_cal:.4f} M$/year")

    print("\n----------Unit Operating Costs----------")
    Ozone_opex = value(
        pyunits.convert(m.fs.non_RO.Ozone.electricity[0] * m.fs.zo_costing_nonRO.electricity_cost * m.fs.zo_costing_nonRO.utilization_factor,
                        to_units=pyunits.USD_2020 / pyunits.year,
                        )
    )

    BAF_opex = value(
        pyunits.convert(
            m.fs.non_RO.BAF.electricity[0] * m.fs.zo_costing_nonRO.electricity_cost * m.fs.zo_costing_nonRO.utilization_factor,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    )

    UF_opex = value(
        pyunits.convert(
            m.fs.non_RO.UF.electricity[0] * m.fs.zo_costing_nonRO.electricity_cost * m.fs.zo_costing_nonRO.utilization_factor,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    )

    GAC_opex = value(pyunits.convert(
        (
                pyunits.convert(m.fs.non_RO.GAC.electricity[0] * m.fs.zo_costing_nonRO.electricity_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour) +
                pyunits.convert(m.fs.non_RO.GAC.activated_carbon_demand[0] * m.fs.zo_costing_nonRO.activated_carbon_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour)
        ) * m.fs.zo_costing_nonRO.utilization_factor,
        to_units=pyunits.USD_2020 / pyunits.year)
    )

    UV_AOP_opex = value(pyunits.convert(
        (
                pyunits.convert(m.fs.non_RO.UV_AOP.electricity[0] * m.fs.zo_costing_nonRO.electricity_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour) +
                pyunits.convert(m.fs.non_RO.UV_AOP.chemical_flow_mass[0] * m.fs.zo_costing_nonRO.hydrogen_peroxide_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour)
        ) * m.fs.zo_costing_nonRO.utilization_factor,
        to_units=pyunits.USD_2020 / pyunits.year)
    )

    chem_flow_mass = (
            m.fs.non_RO.Cl.chlorine_dose[0] * m.fs.non_RO.Cl.properties_in[0].flow_vol
    )
    Cl_opex = value(pyunits.convert(
        (
                pyunits.convert(m.fs.non_RO.Cl.electricity[0] * m.fs.zo_costing_nonRO.electricity_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour) +
                pyunits.convert(chem_flow_mass * m.fs.zo_costing_nonRO.chlorine_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour)
        ) * m.fs.zo_costing_nonRO.utilization_factor,
        to_units=pyunits.USD_2020 / pyunits.year)
    )

    Opex_dict = {"Ozone": Ozone_opex,"BAF": BAF_opex,"UF": UF_opex,"GAC": GAC_opex,"UV_AOP": UV_AOP_opex,"Cl": Cl_opex}
    for unit, unit_opex in Opex_dict.items():
        print(f"{unit} Opex: {unit_opex:.4f} USD/year")
    print("---------------------------")

    externalities = value(pyunits.convert(m.fs.total_externalities, to_units=pyunits.MUSD_2020 / pyunits.year))
    wrr = value(pyunits.convert(m.fs.water_recovery_revenue, to_units=pyunits.USD_2020 / pyunits.year))

    # normalized costs
    feed_flowrate = value(
        pyunits.convert(
            m.fs.feed.properties[0].flow_vol, to_units=pyunits.m**3 / pyunits.hr
        )
    )

    capex_norm = (
        value(pyunits.convert(m.fs.total_capital_cost, to_units=pyunits.USD_2020))
        / feed_flowrate
    )

    annual_investment = value(
        pyunits.convert(
            m.fs.total_capital_cost * m.fs.zo_costing_nonRO.capital_recovery_factor
            + m.fs.total_operating_cost,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    )

    opex_fraction = (
        100
        * value(
            pyunits.convert(
                m.fs.total_operating_cost, to_units=pyunits.USD_2020 / pyunits.year
            )
        )
        / annual_investment
    )

    lcot = value(pyunits.convert(m.fs.LCOT, to_units=pyunits.USD_2020 / pyunits.m ** 3))
    lcot_wo_revenue = value(pyunits.convert(m.fs.LCOT_wo_revenue, to_units=pyunits.USD_2020 / pyunits.m ** 3))
    lcow = value(pyunits.convert(m.fs.LCOW, to_units=pyunits.USD_2020 / pyunits.m ** 3))
    lcow_wo_revenue = value(pyunits.convert(m.fs.LCOW_wo_revenue, to_units=pyunits.USD_2020 / pyunits.m ** 3))
    sec = m.fs.specific_energy_intensity()

    # print(f"\nTotal Capital Cost: {capex:.4f} M$")
    # print(f"\nTotal Operating Cost: {opex:.4f} M$/year")
    print(f"\nTotal Externalities: {externalities:.4f} M$/year")
    print(f"Water Recovery Revenue: {wrr: .4f} USD/year")
    print(f"\nTotal Annual Cost: {annual_investment : .4f} $/year")
    print(f"Normalized Capital Cost: {capex_norm:.4f} $/(m3feed/hr)")
    print(f"Opex Fraction of Annual Cost:{opex_fraction : .4f} %")
    print(f"Levelized cost of treatment: {lcot:.4f} $/m3feed")
    print(f"Levelized cost of treatment without revenue: {lcot_wo_revenue:.4f} $/m3feed")
    print(f"Levelized cost of water: {lcow:.4f} $/m3treated")
    print(f"Levelized cost of water without revenue: {lcow_wo_revenue:.4f} $/m3treated")
    print(f"Specific energy intensity (given inlet flow): {sec:.3f} kWh/m3feed")


def display_costing_RO_DPR(m):
    # Capex
    print("\n----------System costing metrics (RO DPR)----------\n")
    capex = value(pyunits.convert(m.fs.total_capital_cost, to_units=pyunits.MUSD_2020))
    print(f"Total Capital Cost: {capex:.4f} M$")

    print("\n----------Unit Capital Costs----------")
    for u in m.fs.zo_costing_RO_pre._registered_unit_costing:
        print(
            u.name,
            " : {price:0.3f} $".format(
                price=value(pyunits.convert(u.capital_cost, to_units=pyunits.USD_2020))
            ),
        )
    for z in m.fs.ro_costing._registered_unit_costing:
        print(
            z.name,
            " : {price:0.3f} $".format(
                price=value(pyunits.convert(z.capital_cost, to_units=pyunits.USD_2020))
            ),
        )

    total_unit_capex_cal = 0
    # RO_pre
    unit_names = ["Cl", "UF"]
    for unit in unit_names:
        unit_capex = value(pyunits.convert(getattr(m.fs.RO_pre, unit).costing.capital_cost, to_units=pyunits.USD_2020))
        print(f"{unit} capex: {unit_capex:0.3f} $")
        total_unit_capex_cal += unit_capex

    # RO_main
    unit_names = ["P1", "RO", "M1", "PXR", "P2"]
    total_ro_capex_cal = 0
    for unit in unit_names:
        unit_capex = value(pyunits.convert(getattr(m.fs.RO_main, unit).costing.capital_cost, to_units=pyunits.USD_2020))
        total_ro_capex_cal += unit_capex
    print(f"RO capex: {total_ro_capex_cal:0.3f} $")

    total_unit_capex_cal += total_ro_capex_cal
    print(f"(Calculated) Sum of capital costs of unit processes: {total_unit_capex_cal / 1e6:.4f} M$")
    print("---------------------------")

    # Opex
    opex = value(pyunits.convert(m.fs.total_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))
    total_fixed_operating_cost = value(pyunits.convert(m.fs.zo_costing_RO_pre.total_fixed_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))  # excluding RO
    total_variable_operating_cost = value(pyunits.convert(m.fs.zo_costing_RO_pre.total_variable_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))  # excluding RO
    ro_opex = value(pyunits.convert(m.fs.ro_costing.total_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))

    print(f"\nTotal Operating Cost (Cop,tot): {opex:.4f} M$/year")
    print(f"\nOperating Cost of RO: {ro_opex:.4f} M$/year")

    print(f"Total fixed operating cost, excluding RO  (Cop,fix): {total_fixed_operating_cost:.4f} M$/year")
    print(f"Total variable operating cost, excluding RO (Cop,fix): {total_variable_operating_cost:.4f} M$/year\n")

    # Comparison with calculated values
    total_fixed_operating_cost_cal = value(pyunits.convert(m.fs.zo_costing_RO_pre.aggregate_fixed_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))  # excluding RO
    total_variable_operating_cost_vop_cal = value(pyunits.convert(m.fs.zo_costing_RO_pre.aggregate_variable_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))  # excluding RO
    # print("Used flows:")
    # for flow in m.fs.zo_costing_nonRO.used_flows:
    #    print(flow)
    total_flow_cost_cal = value(  # excluding RO
        pyunits.convert(
            sum(m.fs.zo_costing_RO_pre.aggregate_flow_costs[flow] for flow in m.fs.zo_costing_RO_pre.used_flows)
            * m.fs.zo_costing_RO_pre.utilization_factor,
            to_units=pyunits.MUSD_2020 / pyunits.year
        )
    )

    total_operating_cost_cal = total_fixed_operating_cost_cal + total_variable_operating_cost_vop_cal + total_flow_cost_cal  # excluding RO
    total_variable_operating_cost_cal = total_variable_operating_cost_vop_cal + total_flow_cost_cal  # excluding flow cost used in RO

    print(f"(Calculated) Total Operating Cost, excluding RO (Cop,tot): {total_operating_cost_cal:.4f} M$")
    print(f"(Calcualted) Total fixed operating cost, excluding RO (Cop,fix): {total_fixed_operating_cost_cal:.4f} M$/year")
    print(f"(Calculated) Total variable operating cost, excluding RO (Cop,fix): {total_variable_operating_cost_cal:.4f} M$/year")
    print(f"(Calculated) Total variable operating cost from unit models, excluding RO (Cvop,u): {total_variable_operating_cost_vop_cal:.4f} M$/year")
    print(f"(Calculated) Total flow cost (futil*Cflow,tot), excluding flow cost used in RO: {total_flow_cost_cal:.4f} M$/year")

    print("\n----------Unit Operating Costs----------")
    chem_flow_mass = (
            m.fs.RO_pre.Cl.chlorine_dose[0] * m.fs.RO_pre.Cl.properties_in[0].flow_vol
    )
    Cl_opex = value(pyunits.convert(
        (
                pyunits.convert(m.fs.RO_pre.Cl.electricity[0] * m.fs.zo_costing_RO_pre.electricity_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour) +
                pyunits.convert(chem_flow_mass * m.fs.zo_costing_RO_pre.chlorine_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour)
        ) * m.fs.zo_costing_RO_pre.utilization_factor,
        to_units=pyunits.USD_2020 / pyunits.year)
    )

    UF_opex = value(
        pyunits.convert(
            m.fs.RO_pre.UF.electricity[0] * m.fs.zo_costing_RO_pre.electricity_cost * m.fs.zo_costing_RO_pre.utilization_factor,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    )

    Opex_dict = {"Cl": Cl_opex, "UF": UF_opex, "RO": ro_opex*1e6}
    for unit, unit_opex in Opex_dict.items():
        print(f"{unit} Opex: {unit_opex:.4f} USD/year")
    print("---------------------------")

    externalities = value(pyunits.convert(m.fs.total_externalities, to_units=pyunits.MUSD_2020 / pyunits.year))
    wrr = value(pyunits.convert(m.fs.water_recovery_revenue, to_units=pyunits.USD_2020 / pyunits.year))

    # normalized costs
    feed_flowrate = value(
        pyunits.convert(
            m.fs.feed.properties[0].flow_vol, to_units=pyunits.m**3 / pyunits.hr
        )
    )

    capex_norm = (
        value(pyunits.convert(m.fs.total_capital_cost, to_units=pyunits.USD_2020))
        / feed_flowrate
    )

    annual_investment = value(
        pyunits.convert(
            m.fs.total_capital_cost * m.fs.zo_costing_RO_pre.capital_recovery_factor
            + m.fs.total_operating_cost,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    )

    opex_fraction = (
        100
        * value(
            pyunits.convert(
                m.fs.total_operating_cost, to_units=pyunits.USD_2020 / pyunits.year
            )
        )
        / annual_investment
    )

    lcot = value(pyunits.convert(m.fs.LCOT, to_units=pyunits.USD_2020 / pyunits.m ** 3))
    lcot_wo_revenue = value(pyunits.convert(m.fs.LCOT_wo_revenue, to_units=pyunits.USD_2020 / pyunits.m ** 3))
    lcow = value(pyunits.convert(m.fs.LCOW, to_units=pyunits.USD_2020 / pyunits.m ** 3))
    lcow_wo_revenue = value(pyunits.convert(m.fs.LCOW_wo_revenue, to_units=pyunits.USD_2020 / pyunits.m ** 3))
    sec = m.fs.specific_energy_intensity()


    # print(f"\nTotal Capital Cost: {capex:.4f} M$")
    # print(f"\nTotal Operating Cost: {opex:.4f} M$/year")
    print(f"\nTotal Externalities: {externalities:.4f} M$/year")
    print(f"Water Recovery Revenue: {wrr: .4f} USD/year")
    print(f"\nTotal Annual Cost: {annual_investment : .4f} $/year")
    print(f"Normalized Capital Cost: {capex_norm:.4f} $/(m3feed/hr)")
    print(f"Opex Fraction of Annual Cost:{opex_fraction : .4f} %")
    print(f"Levelized cost of treatment: {lcot:.4f} $/m3feed")
    print(f"Levelized cost of treatment without revenue: {lcot_wo_revenue:.4f} $/m3feed")
    print(f"Levelized cost of water: {lcow:.4f} $/m3treated")
    print(f"Levelized cost of water without revenue: {lcow_wo_revenue:.4f} $/m3treated")
    print(f"Specific energy intensity (given inlet flow): {sec:.3f} kWh/m3feed")


if __name__ == "__main__":
    model_nonRO, model_RO, results_nonRO, results_RO = main(working_directory="local")