"""
Non-RO DPR (O3/BAF/UF/Carbon_Adsorption/UV/Chlorination)
This module contains a zero-order representation of a Ozone reactor unit.
"""

import os
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
    UVZO,
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

 #   add_costing(m)
  #  initialize_costing(m)
 #   assert_degrees_of_freedom(m, 0)  # ensures problem is square

 #   results = solve(m, checkpoint="solve flowsheet after costing")
 #  assert_optimal_termination(results)

    display_results(m)
 #   display_costing(m)

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
    m.fs.treated = Product(property_package=m.fs.properties)

    #   non_RO.BAF = BioActiveFiltrationZO(property_package=m.fs.properties, database=m.db)
    #   non_RO.UF = UltraFiltrationZO(property_package=m.fs.properties, database=m.db)
    #   non_RO.GAC = GACZO(property_package=m.fs.properties, database=m.db)
    #   non_RO.UV = UVZO(property_package=m.fs.properties, database=m.db)
    #   non_RO.Cl = ChlorinationZO(property_package=m.fs.properties, database=m.db)

    # RO DPR components

    # connections
    m.fs.s_feed = Arc(source=m.fs.feed.outlet, destination=non_RO.Ozone.inlet)
    non_RO.s01 = Arc(source=non_RO.Ozone.treated, destination=m.fs.treated.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

#    non_RO.Ozone.pprint()
#    non_RO.s01.pprint()
#    m.fs.treated.pprint()

    # scaling

    # set unit m values

    # calculate and propagate scaling factors
    #   iscale.calculate_scaling_factors(m)
    return m


def set_operating_conditions(m):
    non_RO = m.fs.non_RO

    # feed
    feed_temperature = (273.15 + 25) * pyunits.K
    feed_pressure = 101325 * pyunits.Pa
    feed_flow_vol = 0.0004101 * pyunits.m ** 3 / pyunits.s
    feed_conc_mass_toc = 0.005 * pyunits.kg / pyunits.m ** 3  # 1 mg/L = 0.001 kg/m3
    feed_conc_mass_tss = 0.01 * pyunits.kg / pyunits.m ** 3

    m.fs.feed.flow_vol[0].fix(feed_flow_vol)
    m.fs.feed.conc_mass_comp[0, "toc"].fix(feed_conc_mass_toc)
    m.fs.feed.conc_mass_comp[0, "tss"].fix(feed_conc_mass_tss)
    m.fs.feed.conc_mass_comp[0, "cryptosporidium"].fix(1e-14)
    m.fs.feed.conc_mass_comp[0, "giardia_lamblia"].fix(1e-14)
    m.fs.feed.conc_mass_comp[0, "eeq"].fix(1e-14)
    m.fs.feed.conc_mass_comp[0, "total_coliforms_fecal_ecoli"].fix(1e-14)
    m.fs.feed.conc_mass_comp[0, "viruses_enteric"].fix(1e-14)

    solve(m.fs.feed, checkpoint="solve feed block")

    # ozonation
    non_RO.Ozone.load_parameters_from_database(use_default_removal=True)

 #   non_RO.Ozone.pprint()
    non_RO.Ozone.properties_treated[0].flow_mass_comp["toc"].pprint()
    return


def initialize_system(m):
    non_RO = m.fs.non_RO

    # initialize feed
    solve(m.fs.feed, checkpoint="solve flowsheet after initializing feed")

    propagate_state(m.fs.s_feed)
    seq = SequentialDecomposition()
    seq.options.tear_set = []
    seq.options.iterLim = 1

    # initialize ozonation
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

    # cost ozonation
    non_RO.Ozone.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.zo_costing
    )

    # Aggregate unit level costs and calculate overall process costs
    m.fs.zo_costing.cost_process()
    # Add specific energy consumption
    feed_flowrate = m.fs.feed.flow_vol[0]
    m.fs.zo_costing.add_electricity_intensity(feed_flowrate)
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
        doc="Levelized cost of water with respect to volumetric flow"
    )
    def LCOW(b):
        return (
            b.total_capital_cost * b.zo_costing.capital_recovery_factor
            + b.total_operating_cost
        ) / (
            pyunits.convert(
                b.feed.properties[0].flow_vol,
                to_units=pyunits.m**3 / pyunits.year,
            )
            * b.zo_costing.utilization_factor
        )


def initialize_costing(m):
    m.fs.zo_costing.initialize()
    return


def display_results(m):
    print("Feed flow rate:", m.fs.feed.flow_vol[0].value)
    print("TOC concentration in feed:", m.fs.feed.conc_mass_comp[0, "toc"].value)
    print("Ozonation unit outlet flow rate:", m.fs.non_RO.Ozone.properties_treated[0].flow_vol())
    print("TOC concentration after ozonation:", m.fs.non_RO.Ozone.properties_treated[0].conc_mass_comp["toc"]())
    print("% removal of TOC:", 1-m.fs.non_RO.Ozone.properties_treated[0].conc_mass_comp["toc"]()/m.fs.feed.conc_mass_comp[0, "toc"].value)

#   m.fs.non_RO.Ozone.pprint()


def display_costing(m):
    capex = value(pyunits.convert(m.fs.total_capital_cost, to_units=pyunits.MUSD_2020))

    Ozone_capex = value(
        pyunits.convert(
            m.fs.non_RO.Ozone.costing.capital_cost,
            to_units=pyunits.USD_2020,
        )
    )

    opex = value(
        pyunits.convert(
            m.fs.total_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year
        )
    )

    Ozone_opex = value(
            pyunits.convert(
                m.fs.zo_costing.total_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year,
            )
        )

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

    lcow = value(pyunits.convert(m.fs.LCOW, to_units=pyunits.USD_2020 / pyunits.m ** 3))
    sec = m.fs.specific_energy_intensity()



if __name__ == "__main__":
    model, results = main(working_directory="local")
