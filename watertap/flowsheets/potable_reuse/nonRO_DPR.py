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
)

from watertap.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state

import idaes.core.util.scaling as iscale

from idaes.core import UnitModelCostingBlock

from idaes.core.util.model_statistics import degrees_of_freedom  ####in ozonation
from watertap.core.util.initialization import assert_degrees_of_freedom

from idaes.core.util.exceptions import ConfigurationError  ####in ozonation
from idaes.core.util.testing import initialization_tester  ####in ozonation

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
from watertap.unit_models.gac import GAC

from watertap.costing.zero_order_costing import ZeroOrderCosting
from watertap.costing import WaterTAPCosting

# Set up logger
_log = idaeslog.getLogger(__name__)


def main(working_directory=None):
    # build, set, and initialize
    m = build(working_directory=working_directory)
    set_operating_conditions(m)

    assert_units_consistent(m)

    initialize_system(m)
    assert_degrees_of_freedom(m, 0)

    results = solve(m, checkpoint="solve flowsheet after initializing system", tee=True)
    assert_optimal_termination(results)

    add_costing(m)
    initialize_costing(m)
    assert_degrees_of_freedom(m, 0)  # ensures problem is square

    results = solve(m, checkpoint="solve flowsheet after costing")
    assert_optimal_termination(results)

    display_results(m)
    display_costing(m)

    return m, results


def build(working_directory=None):
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

    # define property packages
    m.fs.properties = WaterParameterBlock(
        solute_list=[
            "cryptosporidium",
            "toc",
            "giardia_lamblia",
            "eeq",
            "total_coliforms_fecal_ecoli",
            "viruses_enteric",
            "tss",
        ]
    )

    # define blocks
    non_RO = m.fs.non_RO = Block()
    #   RO = m.fs.RO = Block()

    # define flowsheet inlets and outlets
    m.fs.feed = FeedZO(property_package=m.fs.properties) #water_sources.yaml

    # non-RO DPR components
    non_RO.Ozone = OzoneZO(property_package=m.fs.properties, database=m.db)
    non_RO.BAF = BioActiveFiltrationZO(property_package=m.fs.properties, database=m.db)
    non_RO.UF = UltraFiltrationZO(property_package=m.fs.properties, database=m.db)
    non_RO.GAC = GACZO(property_package=m.fs.properties, database=m.db)
    non_RO.UV_AOP = UVAOPZO(property_package=m.fs.properties, database=m.db)
    non_RO.Cl = ChlorinationZO(property_package=m.fs.properties, database=m.db)
    m.fs.treated = Product(property_package=m.fs.properties)

    # RO DPR components

    # connections
    m.fs.s_feed = Arc(source=m.fs.feed.outlet, destination=non_RO.Ozone.inlet)
    non_RO.s01 = Arc(source=non_RO.Ozone.treated, destination=non_RO.BAF.inlet)
    non_RO.s02 = Arc(source=non_RO.BAF.treated, destination=non_RO.UF.inlet)
    non_RO.s03 = Arc(source=non_RO.UF.treated, destination=non_RO.GAC.inlet)
    non_RO.s04 = Arc(source=non_RO.GAC.treated, destination=non_RO.UV_AOP.inlet)
    non_RO.s05 = Arc(source=non_RO.UV_AOP.treated, destination=non_RO.Cl.inlet)
    non_RO.s06 = Arc(source=non_RO.Cl.treated, destination=m.fs.treated.inlet)


    TransformationFactory("network.expand_arcs").apply_to(m)

    # scaling
    """
    """

    # set unit model values

    # calculate and propagate scaling factors
    #iscale.calculate_scaling_factors(m)
    return m


def set_operating_conditions(m):
    non_RO = m.fs.non_RO

    # feed
    feed_temperature = (273.15 + 25) * pyunits.K
    feed_pressure = 101325 * pyunits.Pa
    feed_flow_vol = 0.0004101*100 * pyunits.m ** 3 / pyunits.s
    feed_conc_mass_toc = 0.005 * pyunits.kg / pyunits.m ** 3  # 1 mg/L = 0.001 kg/m3
    feed_conc_mass_tss = 0.01 * pyunits.kg / pyunits.m ** 3

    m.fs.feed.flow_vol[0].fix(feed_flow_vol)
    m.fs.feed.conc_mass_comp[0, "toc"].fix(feed_conc_mass_toc)
    m.fs.feed.conc_mass_comp[0, "tss"].fix(feed_conc_mass_tss)
    m.fs.feed.conc_mass_comp[0, "cryptosporidium"].fix(1)
    m.fs.feed.conc_mass_comp[0, "giardia_lamblia"].fix(1)
    m.fs.feed.conc_mass_comp[0, "eeq"].fix(1)
    m.fs.feed.conc_mass_comp[0, "total_coliforms_fecal_ecoli"].fix(1)
    m.fs.feed.conc_mass_comp[0, "viruses_enteric"].fix(1)

    iscale.set_variable_scaling_from_current_value(m.fs.feed, descend_into=False)

    """""
    iscale.set_scaling_factor(m.fs.feed.flow_vol[0], 1/value(feed_flow_vol))
    iscale.set_scaling_factor(m.fs.feed.conc_mass_comp[0, "toc"], 1/value(feed_conc_mass_toc))
    iscale.set_scaling_factor(m.fs.feed.conc_mass_comp[0, "tss"], 1/value(feed_conc_mass_tss))
    iscale.set_scaling_factor(m.fs.feed.conc_mass_comp[0, "cryptosporidium"], 1)
    iscale.set_scaling_factor(m.fs.feed.conc_mass_comp[0, "giardia_lamblia"], 1)
    iscale.set_scaling_factor(m.fs.feed.conc_mass_comp[0, "eeq"], 1)
    iscale.set_scaling_factor(m.fs.feed.conc_mass_comp[0, "total_coliforms_fecal_ecoli"], 1)
    iscale.set_scaling_factor(m.fs.feed.conc_mass_comp[0, "viruses_enteric"], 1)
    """""
    solve(m.fs.feed, checkpoint="solve feed block")

    # ozonation
    non_RO.Ozone.load_parameters_from_database(use_default_removal=True)
    # bio-active filtration
    non_RO.BAF.load_parameters_from_database(use_default_removal=True)
    # ultrafiltration
    non_RO.UF.load_parameters_from_database(use_default_removal=True)
    # GAC
    non_RO.GAC.load_parameters_from_database(use_default_removal=True)
    # UV-AOP
    non_RO.UV_AOP.load_parameters_from_database(use_default_removal=True)
    # Chlorination
    non_RO.Cl.load_parameters_from_database(use_default_removal=True)

    return


def initialize_system(m):
    non_RO = m.fs.non_RO

    # initialize feed
    solve(m.fs.feed, checkpoint="solve flowsheet after initializing feed")

    propagate_state(m.fs.s_feed)
    seq = SequentialDecomposition()
    seq.options.tear_set = []
    seq.options.iterLim = 1

    # initialize ozonation, BAF, and UF
    seq.run(non_RO, lambda u: u.initialize())

    return


def solve(blk, solver=None, checkpoint=None, tee=False, fail_flag=True):
    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=tee)
    return results


def add_costing(m):
    non_RO = m.fs.non_RO

    # Zero order costing
    source_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "nonRO_DPR_global_costing.yaml",
    )

    m.fs.zo_costing = ZeroOrderCosting(case_study_definition=source_file)

    # cost of each unit process
    non_RO.Ozone.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    non_RO.BAF.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    non_RO.UF.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    non_RO.GAC.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    non_RO.UV_AOP.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
    non_RO.Cl.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)

    # Aggregate unit level costs and calculate overall process costs
    m.fs.zo_costing.cost_process()

    # Add annual water production, specific energy consumption, and specific electrical carbon intensity
    feed_flowrate = m.fs.feed.flow_vol[0]
    treated_flowrate = m.fs.treated.properties[0].flow_vol
    m.fs.zo_costing.add_annual_water_production(treated_flowrate)
    m.fs.annual_water_production = Expression(
        expr=(m.fs.zo_costing.annual_water_production),
        doc="Annual water production based on flowrate basis [m3/year]",
    )

    m.fs.zo_costing.add_electricity_intensity(feed_flowrate) # m.fs.zo_costing.electricity_intensity is an expression
    m.fs.specific_energy_intensity = Expression(
        expr=(m.fs.zo_costing.electricity_intensity),
        doc="Specific energy consumption of the treatment train on a feed flowrate basis [kWh/m3]",
    )

    @m.fs.Expression(doc="Total capital cost of the treatment train")
    def total_capital_cost(b):
        return pyunits.convert(
            m.fs.zo_costing.total_capital_cost, to_units=pyunits.USD_2020
        )

    @m.fs.Expression(doc="Total operating cost of the treatment train")
    def total_operating_cost(b):
        return pyunits.convert(
            m.fs.zo_costing.total_fixed_operating_cost,
            to_units=pyunits.USD_2020 / pyunits.year,
        ) + pyunits.convert(
            m.fs.zo_costing.total_variable_operating_cost,
            to_units=pyunits.USD_2020 / pyunits.year,
        )

    @m.fs.Expression(
        doc="Levelized cost of treatment with respect to volumetric feed flow"
    )
    def LCOT(b):
        return (
                b.total_capital_cost * b.zo_costing.capital_recovery_factor
                + b.total_operating_cost
        ) / (
                pyunits.convert(
                    b.feed.properties[0].flow_vol,  # can use b.feed.flow_vol[0] as well
                    to_units=pyunits.m ** 3 / pyunits.year,
                )
                * b.zo_costing.utilization_factor
        )

    @m.fs.Expression(
        doc="Levelized cost of water with respect to volumetric treated water flow"
    )
    def LCOW(b):
        return (
            b.total_capital_cost * b.zo_costing.capital_recovery_factor
            + b.total_operating_cost
        ) / (
            pyunits.convert(
                b.treated.properties[0].flow_vol,
                to_units=pyunits.m**3 / pyunits.year,
            )
            * b.zo_costing.utilization_factor
        )


def initialize_costing(m):
    m.fs.zo_costing.initialize()
    return


def display_results(m):
    print("\n----------Unit models----------")
    unit_names = ["Ozone", "BAF", "UF", "GAC", "UV_AOP", "Cl"]
    for unit in unit_names:
        getattr(m.fs.non_RO, unit).report()

    # Flow rate
    print("\nFeed flow rate in each unit process")
    print(f"Feed flow rate: {value(pyunits.convert(m.fs.feed.flow_vol[0], to_units=pyunits.m**3 / pyunits.hr,)): .3f} m3/hr")
    for unit in unit_names:
        unit_flow_rate = value(pyunits.convert(getattr(m.fs.non_RO, unit).properties_in[0].flow_vol, to_units=pyunits.m ** 3 / pyunits.hr))
        print(f"Feed flow rate in {unit}: {unit_flow_rate: .3f} m3/hr")

    # TOC removal
    print("\nTOC concentration in each unit process")
    print(f"TOC in feed: {value(pyunits.convert(m.fs.feed.conc_mass_comp[0, 'toc'], to_units=pyunits.mg / pyunits.L)): .3f} mg/L")
    for unit in unit_names:
        unit_block = getattr(m.fs.non_RO, unit)
        toc_in = value(
            pyunits.convert(unit_block.properties_in[0].conc_mass_comp["toc"], to_units=pyunits.mg / pyunits.L))
        toc_out = value(
            pyunits.convert(unit_block.properties_treated[0].conc_mass_comp["toc"], to_units=pyunits.mg / pyunits.L))

        print(f"TOC before {unit}: {toc_in: .3f} mg/L")
        print(f"TOC after {unit}: {toc_out: .3f} mg/L")

       ## Check if TOC removal is defined in the unit's YAML file
       #if "toc" in unit_block.config.database.get_unit_operation_parameters(unit_block._tech_type)[
       #    "removal_frac_mass_comp"]:
       #    removal_toc = 100 * (1 - toc_out / toc_in) * unit_block.recovery_frac_mass_H2O[0].value
       #    print(f"% removal of TOC by {unit}: {removal_toc:.2f}%")
       #else:
       #    print(f"{unit} does not remove TOC.")

    print(
        f"total % TOC concentration removal: {100 * (1 - value(m.fs.treated.properties[0].conc_mass_comp['toc']) / value(m.fs.feed.conc_mass_comp[0, 'toc'])): .2f}%")

    # TSS removal
    print("\nTSS concentration in each unit process")
    print(f"TSS in feed: {value(pyunits.convert(m.fs.feed.conc_mass_comp[0, 'tss'], to_units=pyunits.mg / pyunits.L)): .3f} mg/L")
    for unit in unit_names:
        unit_block = getattr(m.fs.non_RO, unit)
        tss_in = value(
            pyunits.convert(unit_block.properties_in[0].conc_mass_comp["tss"], to_units=pyunits.mg / pyunits.L))
        tss_out = value(
            pyunits.convert(unit_block.properties_treated[0].conc_mass_comp["tss"], to_units=pyunits.mg / pyunits.L))

        print(f"TSS before {unit}: {tss_in: .3f} mg/L")
        print(f"TSS after {unit}: {tss_out: .3f} mg/L")
    print(
        f"total % TSS concentration removal: {100 * (1 - value(m.fs.treated.properties[0].conc_mass_comp['tss']) / value(m.fs.feed.conc_mass_comp[0, 'tss'])): .2f}%")

    # Cryptosporidium % removal
    print("\nCryptosporidium % removal in each unit process")
    for unit in unit_names:
        unit_block = getattr(m.fs.non_RO, unit)
        crypto_in = value(pyunits.convert(unit_block.properties_in[0].conc_mass_comp["cryptosporidium"],
                                          to_units=pyunits.mg / pyunits.L))
        crypto_out = value(pyunits.convert(unit_block.properties_treated[0].conc_mass_comp["cryptosporidium"],
                                           to_units=pyunits.mg / pyunits.L))

        # Check if Cryptosporidium removal is defined in the unit's YAML file
        if "cryptosporidium" in unit_block.config.database.get_unit_operation_parameters(unit_block._tech_type)[
            "removal_frac_mass_comp"]:
            removal_crypto = 100 * (1 - crypto_out / crypto_in) * unit_block.recovery_frac_mass_H2O[0].value
            print(f"% removal of Cryptosporidium by {unit}: {removal_crypto:.6f}%")
        else:
            pass
    print(
        f"Total log removal ratio of Cryptosporidium: "
        f"{-math.log10(value(m.fs.treated.properties[0].conc_mass_comp['cryptosporidium']) / value(m.fs.feed.conc_mass_comp[0, 'cryptosporidium'])): .2f} log"
    )

    # Overall system recovery
    print("\n----------System Recovery----------\n")
    sys_water_recovery = (
            m.fs.treated.properties[0].flow_mass_comp["H2O"]()
            / m.fs.feed.flow_mass_comp[0, "H2O"]()
    )
    print(f"System water recovery: {sys_water_recovery * 100 : .3f}%")

    solute_list = m.fs.properties.solute_set
    for solute in solute_list:
        sys_recovery = (
                m.fs.treated.properties[0].flow_mass_comp[solute]()
                / m.fs.feed.flow_mass_comp[0, solute]()
        )
        print(f"System {solute} recovery: {sys_recovery * 100: .3f}%")

    """
    # check scaling factors
    print("Scaling factor of feed flow rate:", iscale.get_scaling_factor(m.fs.feed.flow_vol[0]))
    print("Scaling factor of toc conc_mass:", iscale.get_scaling_factor(m.fs.feed.conc_mass_comp[0, "toc"]))
    print("Scaling factor of cryptosporidium conc_mass:",iscale.get_scaling_factor(m.fs.feed.conc_mass_comp[0, "cryptosporidium"]))
    """

def display_costing(m):
    # Capex
    print("\n----------Costing----------\n")
    capex = value(pyunits.convert(m.fs.total_capital_cost, to_units=pyunits.MUSD_2020))
    print(f"Total Capital Cost: {capex:.4f} M$")

    print("\n----------Unit Capital Costs----------")
    unit_names = ["Ozone", "BAF", "UF", "GAC", "UV_AOP", "Cl"]

    total_unit_capex_cal = 0
    for unit in unit_names:
        unit_capex = value(pyunits.convert(getattr(m.fs.non_RO, unit).costing.capital_cost, to_units=pyunits.USD_2020))
        print(f"{unit} capex: {unit_capex:0.3f} $")
        total_unit_capex_cal += unit_capex
    print(f"(Calculated) Sum of capital costs of unit processes: {total_unit_capex_cal/1e6:.4f} M$")
    print("---------------------------")


    # Opex
    opex = value(pyunits.convert(m.fs.total_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))
    total_fixed_operating_cost = value(pyunits.convert(m.fs.zo_costing.total_fixed_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))
    total_variable_operating_cost = value(pyunits.convert(m.fs.zo_costing.total_variable_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))

    print(f"\nTotal Operating Cost (Cop,tot): {opex:.4f} M$/year")
    print(f"Total fixed operating cost (Cop,fix): {total_fixed_operating_cost:.4f} M$/year")
    print(f"Total variable operating cost (Cop,fix): {total_variable_operating_cost:.4f} M$/year\n")

    # Comparison with calculated values
    total_fixed_operating_cost_cal = value(pyunits.convert(m.fs.zo_costing.aggregate_fixed_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))
    total_variable_operating_cost_vop_cal = value(pyunits.convert(m.fs.zo_costing.aggregate_variable_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))
    #print("Used flows:")
    #for flow in m.fs.zo_costing.used_flows:
    #    print(flow)
    total_flow_cost_cal = value(
        pyunits.convert(
            (
                    m.fs.zo_costing.aggregate_flow_costs["electricity"] +
                    m.fs.zo_costing.aggregate_flow_costs["activated_carbon"] +
                    m.fs.zo_costing.aggregate_flow_costs["chlorine"] +
                    m.fs.zo_costing.aggregate_flow_costs["hydrogen_peroxide"]
            ) * m.fs.zo_costing.utilization_factor,
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
        pyunits.convert(m.fs.non_RO.Ozone.electricity[0] * m.fs.zo_costing.electricity_cost * m.fs.zo_costing.utilization_factor,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    )

    BAF_opex = value(
        pyunits.convert(
            m.fs.non_RO.BAF.electricity[0] * m.fs.zo_costing.electricity_cost * m.fs.zo_costing.utilization_factor,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    )

    UF_opex = value(
        pyunits.convert(
            m.fs.non_RO.UF.electricity[0] * m.fs.zo_costing.electricity_cost * m.fs.zo_costing.utilization_factor,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    )

    GAC_opex = value(pyunits.convert(
        (
                pyunits.convert(m.fs.non_RO.GAC.electricity[0] * m.fs.zo_costing.electricity_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour) +
                pyunits.convert(m.fs.non_RO.GAC.activated_carbon_demand[0] * m.fs.zo_costing.activated_carbon_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour)
        ) * m.fs.zo_costing.utilization_factor,
        to_units=pyunits.USD_2020 / pyunits.year)
    )

    UV_AOP_opex = value(pyunits.convert(
        (
                pyunits.convert(m.fs.non_RO.UV_AOP.electricity[0] * m.fs.zo_costing.electricity_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour) +
                pyunits.convert(m.fs.non_RO.UV_AOP.chemical_flow_mass[0] * m.fs.zo_costing.hydrogen_peroxide_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour)
        ) * m.fs.zo_costing.utilization_factor,
        to_units=pyunits.USD_2020 / pyunits.year)
    )

    chem_flow_mass = (
            m.fs.non_RO.Cl.chlorine_dose[0] * m.fs.non_RO.Cl.properties_in[0].flow_vol
    )
    Cl_opex = value(pyunits.convert(
        (
                pyunits.convert(m.fs.non_RO.Cl.electricity[0] * m.fs.zo_costing.electricity_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour) +
                pyunits.convert(chem_flow_mass * m.fs.zo_costing.chlorine_cost,
                                to_units=pyunits.USD_2020 / pyunits.hour)
        ) * m.fs.zo_costing.utilization_factor,
        to_units=pyunits.USD_2020 / pyunits.year)
    )

    Opex_dict = {"Ozone": Ozone_opex,"BAF": BAF_opex,"UF": UF_opex,"GAC": GAC_opex,"UV_AOP": UV_AOP_opex,"Cl": Cl_opex,}
    for unit, unit_opex in Opex_dict.items():
        print(f"{unit} Opex: {unit_opex:.4f} USD/year")
    print("---------------------------\n")

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
            m.fs.total_capital_cost * m.fs.zo_costing.capital_recovery_factor
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

    annual_water_production = value(pyunits.convert( m.fs.annual_water_production, to_units=pyunits.m ** 3 / pyunits.year))
    lcot = value(pyunits.convert(m.fs.LCOT, to_units=pyunits.USD_2020 / pyunits.m ** 3))
    lcow = value(pyunits.convert(m.fs.LCOW, to_units=pyunits.USD_2020 / pyunits.m ** 3))
    sec = m.fs.specific_energy_intensity()

    #opex = value(pyunits.convert(m.fs.zo_costing.total_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))


    print("\nSystem costing metrics:")
    print(f"\nTotal Capital Cost: {capex:.4f} M$")
    print(f"\nTotal Operating Cost: {opex:.4f} M$/year")
    print(f"\nTotal Annual Cost: {annual_investment : .4f} $/year")
    print(f"Normalized Capital Cost: {capex_norm:.4f} $/(m3feed/hr)")
    print(f"Opex Fraction of Annual Cost:{opex_fraction : .4f} %")
    print(f"\nAnnual water production:{annual_water_production : .4f} m3treated/year")
    print(f"Levelized cost of treatment: {lcot:.4f} $/m3feed")
    print(f"Levelized cost of water: {lcow:.4f} $/m3treated")
    print(f"Specific energy intensity (given inlet flow): {sec:.3f} kWh/m3feed")


if __name__ == "__main__":
    model, results = main(working_directory="local")
