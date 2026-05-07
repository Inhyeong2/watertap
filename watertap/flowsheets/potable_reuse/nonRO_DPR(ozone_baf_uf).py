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
    m.fs.treated = Product(property_package=m.fs.properties)

    #   non_RO.GAC = GACZO(property_package=m.fs.properties, database=m.db) # GAC
    #   non_RO.UV = UVZO(property_package=m.fs.properties, database=m.db)
    #   non_RO.Cl = ChlorinationZO(property_package=m.fs.properties, database=m.db)

    # RO DPR components

    # connections
    m.fs.s_feed = Arc(source=m.fs.feed.outlet, destination=non_RO.Ozone.inlet)
    non_RO.s01 = Arc(source=non_RO.Ozone.treated, destination=non_RO.BAF.inlet)
    non_RO.s02 = Arc(source=non_RO.BAF.treated, destination=non_RO.UF.inlet)
    non_RO.s03 = Arc(source=non_RO.UF.treated, destination=m.fs.treated.inlet)

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

    # cost ozonation
    non_RO.Ozone.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.zo_costing
    )

    # cost BAF
    non_RO.BAF.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.zo_costing
    )

    # cost UF
    non_RO.UF.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.zo_costing
    )

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
    print("Feed flow rate:", m.fs.feed.flow_vol[0].value)
    print("TOC concentration in feed:", m.fs.feed.conc_mass_comp[0, "toc"].value)
    print("Ozonation unit outlet flow rate:", m.fs.non_RO.Ozone.properties_treated[0].flow_vol())
    print("TOC concentration after ozonation:", m.fs.non_RO.Ozone.properties_treated[0].conc_mass_comp["toc"]())
    print("% removal of TOC by ozonation:", 1-m.fs.non_RO.Ozone.properties_treated[0].conc_mass_comp["toc"]()/m.fs.feed.conc_mass_comp[0, "toc"].value)

    print("TOC concentration after BAF:", m.fs.non_RO.BAF.properties_treated[0].conc_mass_comp["toc"]())
    print("TOC concentration after UF:", m.fs.non_RO.UF.properties_treated[0].conc_mass_comp["toc"]())
    print("% removal of TOC by UF:", 1-m.fs.non_RO.UF.properties_treated[0].conc_mass_comp["toc"]()/m.fs.non_RO.BAF.properties_treated[0].conc_mass_comp["toc"]())

    print("% removal of Crypto by ozonation:", 1 - m.fs.non_RO.Ozone.properties_treated[0].conc_mass_comp["cryptosporidium"]() / m.fs.feed.conc_mass_comp[0, "cryptosporidium"].value)
    print("% removal of Crypto by UF:",
          1 - m.fs.non_RO.UF.properties_treated[0].conc_mass_comp["cryptosporidium"]() / m.fs.non_RO.BAF.properties_treated[0].conc_mass_comp["cryptosporidium"]())

    print("feed_flowrate_Var:", m.fs.feed.flow_vol[0].value)
    print("feed_flowrate_Exp:", m.fs.feed.properties[0].flow_vol())
    print("treated_flowrate_Exp:", m.fs.treated.properties[0].flow_vol())

    """
    print(type(m.fs.feed.properties[0].flow_vol))  # pyomo.core.base.expression.ScalarExpression type
    print(type(m.fs.feed.properties[0].flow_vol()))  # float type
    print(type(m.fs.feed.flow_vol[0])) # pyomo.core.base.var.VarData type
    print(type(m.fs.feed.flow_vol[0].value)) # float type

    print(type(m.fs.treated.properties[0].flow_vol))  # pyomo.core.base.expression.ScalarExpression type
    print(type(m.fs.treated.properties[0].flow_vol()))  # float type

    # print("treated_flowrate:", m.fs.treated.flow_vol[0].value) #--> not existing
    # print(type(m.fs.treated.flow_vol[0])) #--> not existing
    """

    # check scaling factors
    print("Scaling factor of feed flow rate:", iscale.get_scaling_factor(m.fs.feed.flow_vol[0]))
    print("Scaling factor of toc conc_mass:", iscale.get_scaling_factor(m.fs.feed.conc_mass_comp[0, "toc"]))
    print("Scaling factor of cryptosporidium conc_mass:",iscale.get_scaling_factor(m.fs.feed.conc_mass_comp[0, "cryptosporidium"]))

    print("Ozone_consumption:", value(m.fs.non_RO.Ozone.ozone_consumption[0]))

#    m.fs.non_RO.Ozone.pprint()
#    non_RO.Ozone.properties_treated[0].flow_mass_comp["toc"].pprint()


def display_costing(m):
    capex = value(pyunits.convert(m.fs.total_capital_cost, to_units=pyunits.MUSD_2020))

    Ozone_capex = value(
        pyunits.convert(
            m.fs.non_RO.Ozone.costing.capital_cost,
            to_units=pyunits.USD_2020,
        )
    )

    BAF_capex = value(
        pyunits.convert(
            m.fs.non_RO.BAF.costing.capital_cost,
            to_units=pyunits.USD_2020,
        )
    )

    UF_capex = value(
        pyunits.convert(
            m.fs.non_RO.UF.costing.capital_cost,
            to_units=pyunits.USD_2020,
        )
    )

    print("Total Capex:", capex)
    print("Ozone Capex:", Ozone_capex)
    print("BAF Capex:", BAF_capex)
    print("UF Capex:", UF_capex)

    print("Total capital cost:", m.fs.zo_costing.total_capital_cost())
    print("Aggregated capital cost:", m.fs.zo_costing.aggregate_capital_cost())
    print("Total investment factor:", m.fs.zo_costing.total_investment_factor())
   #m.fs.zo_costing.pprint()

#   m.fs.total_capital_cost.pprint()

#   m.fs.non_RO.Ozone.costing.pprint()
#   m.fs.non_RO.BAF.costing.pprint()
#   m.fs.non_RO.UF.costing.pprint()


    opex = value(
        pyunits.convert(
            m.fs.zo_costing.total_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year
        )
    )


    tot_mlc_factor = (m.fs.zo_costing.salaries_percent_FCI
                      + m.fs.zo_costing.benefit_percent_of_salary * m.fs.zo_costing.salaries_percent_FCI
                      + m.fs.zo_costing.maintenance_costs_percent_FCI
                      + m.fs.zo_costing.laboratory_fees_percent_FCI
                      + m.fs.zo_costing.insurance_and_taxes_percent_FCI)

    Ozone_opex = value(
        #pyunits.convert(
        #    m.fs.non_RO.Ozone.costing.fixed_operating_cost,
        #    to_units=pyunits.USD_2020 / pyunits.year,
        #)
        #+ pyunits.convert(
        #    tot_mlc_factor * Ozone_capex + m.fs.non_RO.Ozone.costing.variable_operating_cost,
        #    to_units=pyunits.USD_2020 / pyunits.year,
        #)
        + pyunits.convert(m.fs.non_RO.Ozone.electricity[0] * m.fs.zo_costing.electricity_cost * m.fs.zo_costing.utilization_factor,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    )

    BAF_opex = value(
        #pyunits.convert(
        #    m.fs.non_RO.BAF.costing.fixed_operating_cost,
        #    to_units=pyunits.USD_2020 / pyunits.year,
        #)
        #+ pyunits.convert(
        #    tot_mlc_factor * BAF_capex + m.fs.non_RO.BAF.costing.variable_operating_cost,
        #    to_units=pyunits.USD_2020 / pyunits.year,
        #)
        + pyunits.convert(
            m.fs.non_RO.BAF.electricity[0] * m.fs.zo_costing.electricity_cost * m.fs.zo_costing.utilization_factor,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    )

    UF_opex = value(
        #pyunits.convert(
        #    m.fs.non_RO.UF.costing.fixed_operating_cost,
        #    to_units=pyunits.USD_2020 / pyunits.year,
        #)
        #+ pyunits.convert(
        #    tot_mlc_factor * UF_capex + m.fs.non_RO.UF.costing.variable_operating_cost,
        #    to_units=pyunits.USD_2020 / pyunits.year,
        #)
        + pyunits.convert(
            m.fs.non_RO.UF.electricity[0] * m.fs.zo_costing.electricity_cost * m.fs.zo_costing.utilization_factor,
            to_units=pyunits.USD_2020 / pyunits.year,
        )
    )

    print("Total Opex:", opex)
    print("Ozone Opex:", Ozone_opex)
    print("BAF Opex:", BAF_opex)
    print("UF Opex:", UF_opex)

    print("Electricity_Ozone(kW):", m.fs.non_RO.Ozone.electricity[0].value)
    print("Electricity_BAF(kW):", m.fs.non_RO.BAF.electricity[0].value)
    print("Electricity_UF(kW):", m.fs.non_RO.UF.electricity[0].value)

    print("Plant utilization factor:", m.fs.zo_costing.utilization_factor())
#   m.fs.non_RO.Ozone.pprint()
    print("Flow_vol in ozonation:", m.fs.non_RO.Ozone.properties_in[0].flow_vol())
#    print("Flow_vol in ozonation:", m.fs.non_RO.Ozone.properties[0].flow_vol())  not working
    print("Flow_vol in BAF:", m.fs.non_RO.BAF.properties_in[0].flow_vol())
    print("Flow_vol in UF:", m.fs.non_RO.UF.properties_in[0].flow_vol())
#   print(m.fs.zo_costing.electricity_cost())

    # check how the operating cost of unit processes is calculated
    print("Ozone Opex:", value(
        pyunits.convert(
        m.fs.non_RO.Ozone.specific_energy_coeff[0]
        * m.fs.non_RO.Ozone.ozone_consumption[0]
        * m.fs.non_RO.Ozone.properties_in[0].flow_vol
        * m.fs.zo_costing.electricity_cost
        * m.fs.zo_costing.utilization_factor,
        to_units=pyunits.USD_2020 / pyunits.year)
    ))

    print("BAF Opex:", value(pyunits.convert(
        m.fs.non_RO.BAF.properties_in[0].flow_vol
        * m.fs.non_RO.BAF.energy_electric_flow_vol_inlet
        * m.fs.zo_costing.electricity_cost
        * m.fs.zo_costing.utilization_factor,
        to_units=pyunits.USD_2020 / pyunits.year)
    ))

    print("UF Opex:", value(pyunits.convert(
        m.fs.non_RO.UF.properties_in[0].flow_vol
        * m.fs.non_RO.UF.energy_electric_flow_vol_inlet
        * m.fs.zo_costing.electricity_cost
        * m.fs.zo_costing.utilization_factor,
        to_units=pyunits.USD_2020 / pyunits.year)
    ))

    print("Aggregated fixed operating cost:", m.fs.zo_costing.aggregate_fixed_operating_cost())
    print("Aggregated variable operating cost:", m.fs.zo_costing.aggregate_variable_operating_cost())


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

    lcot = value(pyunits.convert(m.fs.LCOT, to_units=pyunits.USD_2020 / pyunits.m ** 3))
    lcow = value(pyunits.convert(m.fs.LCOW, to_units=pyunits.USD_2020 / pyunits.m ** 3))
    sec = m.fs.specific_energy_intensity()

    print("\n System costing metrics:")
    print(f"\nTotal Capital Cost: {capex:.4f} M$")

    print("\n----------Unit Capital Costs----------\n")
    for u in m.fs.zo_costing._registered_unit_costing:
        print(
            u.name,
            " : {price:0.3f} $".format(
                price=value(pyunits.convert(u.capital_cost, to_units=pyunits.USD_2020))
            ),
        )

    print(f"\nTotal Operating Cost: {opex:.4f} M$/year")
    print(f"Ozonation Operating Cost: {Ozone_opex:.4f} $/yr")
    print(f"BAF Operating Cost: {BAF_opex:.4f} $/yr")
    print(f"UF Operating Cost: {UF_opex:.4f} $/yr")

    print(f"\nTotal Annual Cost: {annual_investment : .4f} $/year")
    print(f"Normalized Capital Cost: {capex_norm:.4f} $/m3feed/hr")
    print(f"Opex Fraction of Annual Cost:{opex_fraction : .4f} %")

    print(f"Levelized cost of treatment: {lcot:.4f} $/m3 feed")
    print(f"Levelized cost of water: {lcow:.4f} $/m3 feed")
    print(f"Specific energy intensity: {sec:.3f} kWh/m3 feed")

    print(m.fs.zo_costing._registered_flows)



if __name__ == "__main__":
    model, results = main(working_directory="local")
