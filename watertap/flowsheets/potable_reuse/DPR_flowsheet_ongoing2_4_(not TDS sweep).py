"""
Non-RO DPR (O3/BAF/UF/Carbon_Adsorption/UV_AOP/Chlorination) and RO DPR (Cl/UF/RO/UV_AOP)
This module contains a zero-order representation of each unit except RO
"""

import os, math
import idaes.logger as idaeslog
import pyomo.environ as pyo
from pyomo.environ import (
    assert_optimal_termination,
    check_optimal_termination,
    ConcreteModel,
    Constraint,
    Block,
    Expression,
    Objective,
    value,
    Var,
    Reference,
    TransformationFactory,
    units as pyunits,
)
from pyomo.network import Arc, SequentialDecomposition
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.check_units import assert_units_consistent

from idaes.core import (
    FlowsheetBlock,
    MomentumBalanceType,
    UnitModelBlockData,
)

from watertap.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state, solve_indexed_blocks

import idaes.core.util.scaling as iscale
from idaes.core.util.exceptions import ConfigurationError
from idaes.models.unit_models import (
    Feed,
    Mixer,
    Separator,
    Product,
    Translator,
    MomentumMixingType,
)

from idaes.core import UnitModelCostingBlock

from watertap.unit_models.pressure_exchanger import PressureExchanger
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice
from watertap.core.util.initialization import assert_degrees_of_freedom

from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.unit_models.reverse_osmosis_1D import ReverseOsmosis1D

from watertap.core.zero_order_properties import WaterParameterBlock
from watertap.core.wt_database import Database
from watertap.unit_models.zero_order import (
    FeedZO,
    OzoneDPRZOv0,
    BioActiveFiltrationDPRZO,
    UltraFiltrationDPRZO,
    GACDPRZO,
    UVAOPDPRZO,
    ChlorinationDPRZO,
)

from watertap.costing.zero_order_costing import ZeroOrderCosting
from watertap.costing import WaterTAPCosting

import pprint

# Some more information about this module
__author__ = "Inhyeong Jeon"

# Set up logger
_log = idaeslog.getLogger(__name__)

def main(state="FL", treatment_train="RBAT", effluent_type="TERTIARY"):
    # Define initial setting
    LRV_req, details, state, treatment_train, effluent_type, solute_list = DPR_initial_setting(state=state, treatment_train=treatment_train, effluent_type=effluent_type)

    # build, set, and initialize
    if treatment_train == "CBAT":
        m = build_nonRO(solute_list=solute_list,
                        state=state,
                        effluent_type=effluent_type,
                        LRVO3_req=LRV_req["ozone_crypto_lrv_required"],
                        LRVClvirus_req=LRV_req["cl2_virus_lrv_required"],
                        LRVClgiardia_req=LRV_req["cl2_giardia_lrv_required"],)
    elif treatment_train == "RBAT":
        m = build_RO(solute_list=solute_list,
                        state=state,
                        effluent_type=effluent_type,
                        LRVO3_req=LRV_req["ozone_crypto_lrv_required"],
                        LRVClvirus_req=LRV_req["cl2_virus_lrv_required"],
                        LRVClgiardia_req=LRV_req["cl2_giardia_lrv_required"],)

    set_operating_conditions(m, flow_vol=10*0.0438, solute_list=solute_list, treatment_train=treatment_train)
    scale_system(m, treatment_train=treatment_train)

    initialize_system(m, solute_list=solute_list, treatment_train=treatment_train)
    assert_degrees_of_freedom(m, 0)
    results = solve(m, checkpoint="solve flowsheet after initializing system", tee=True)
    assert_optimal_termination(results)

    if treatment_train == "RBAT":
        adaptive_overpressure_sweep(m, solute_list=solute_list, target_tds=value(solute_list["tds"]))

    # add_costing(m, treatment_train=treatment_train)
    # initialize_costing(m, treatment_train=treatment_train)
    # assert_degrees_of_freedom(m, 0)  # ensures problem is square
    #
    # optimize_operation(m, state=state, effluent_type=effluent_type, treatment_train=treatment_train)
    #
    # results = solve(m, checkpoint="solve flowsheet after optimization", tee=True)
    # assert_optimal_termination(results)
    #
    # display_initial_setting(LRV_req=LRV_req, process_details=details, solute_list=solute_list)
    # display_results(m, treatment_train=treatment_train)
    # display_cost(m, treatment_train=treatment_train)

    return m, results

def DPR_initial_setting(state=None, treatment_train=None, effluent_type=None):  #In the future, we might need to consider another type of tratment_train for Secondary effluent (for nutrient removal)
    # Water quality
    if effluent_type.upper() not in ["TERTIARY", "SECONDARY"]:
        raise ValueError("Unsupported type of effluent. Choose 'TERTIARY' or 'SECONDARY'.")

    solute_list = {
        "TERTIARY": {
            "toc": 0.01 * pyunits.kg / pyunits.m ** 3, # 1kg/m3 = 1000 mg/L
            "tds": 0.5 * pyunits.kg / pyunits.m ** 3,
            "nitrate": 0.01 * (62.0049/14.0067) * pyunits.kg / pyunits.m ** 3,
            "nitrite": 0.01 * (46.0055/14.0067) * pyunits.kg / pyunits.m ** 3,
        },
        "SECONDARY": {
            "toc": 0.01 * pyunits.kg / pyunits.m ** 3,
            "tds": 1 * pyunits.kg / pyunits.m ** 3,
            "nitrate": 0.01 * (62.0049 / 14.0067) * pyunits.kg / pyunits.m ** 3,
            "nitrite": 0.01 * (46.0055 / 14.0067) * pyunits.kg / pyunits.m ** 3,
        },
    }
    selected_solutes = solute_list[effluent_type]

    # Regulatory requirements
    state_requirements = {
        "CA": {"virus": 20, "giardia": 14, "crypto": 15, "min_processes": 4},
        "CO": {"virus": 12, "giardia": 10, "crypto": 10, "min_processes": 3},
        "FL": {"virus": 14, "giardia": 12, "crypto": 12, "min_processes": 3},
        "AZ": {"virus": 14, "giardia": 12, "crypto": 12, "min_processes": 3},
    }

    if state.upper() not in ["CA", "CO", "FL", "AZ"]:
        raise ValueError("Unsupported state. Choose 'CA', 'CO', 'FL' or 'AZ'.")

    fixed_LRV = {
        "BAC": {"virus": 0, "giardia": 0, "crypto": 0},
        "UF": {"virus": 0, "giardia": 4, "crypto": 4},
        "RO": {"virus": 2, "giardia": 2, "crypto": 2},
        "GAC": {"virus": 0, "giardia": 0, "crypto": 0},
        "UV-AOP": {"virus": 6, "giardia": 6, "crypto": 6},
    }

    if treatment_train.upper() == "RBAT":
        train = ["Ozone", "BAC", "UF", "RO", "UV-AOP", "Cl2"]
    elif treatment_train.upper() == "CBAT":
        train = ["Ozone", "BAC", "UF", "GAC", "UV-AOP", "Cl2"]
    else:
        raise ValueError("Unsupported treatment train. Choose 'RBAT' or 'CBAT'.")

    reg = state_requirements[state.upper()]
    LRV_req = {}
    process_details = {}

    for pathogen in ["virus", "giardia", "crypto"]:
        total_fixed = 0
        used_processes = 0
        contributing_units = []

        for proc in train:
            if proc == "Ozone":
                val = 6 if pathogen != "crypto" else 0
            elif proc == "Cl2":
                val = 0
            else:
                val = fixed_LRV.get(proc, {}).get(pathogen, 0)

            total_fixed += val
            if val >= 1:
                used_processes += 1
                contributing_units.append(f"{proc} LRV {val}")

        required_total = reg[pathogen]
        remaining_lrv = required_total - total_fixed
        status = "OK"

        if pathogen == "crypto":
            ozone_crypto = 1 if remaining_lrv <= 0 else min(remaining_lrv, 6)
            LRV_req["ozone_crypto_lrv_required"] = round(ozone_crypto)
            total = total_fixed + ozone_crypto
            process_ct = used_processes + 1
            contributing_units.append(f"Ozone LRV {round(ozone_crypto)}")

        else:
            cl2_lrv = 0 if remaining_lrv <= 0 else min(remaining_lrv, 6)
            if cl2_lrv == 0 and used_processes < reg["min_processes"]:
                cl2_lrv = 1
            LRV_req[f"cl2_{pathogen}_lrv_required"] = round(cl2_lrv)
            total = total_fixed + cl2_lrv
            process_ct = used_processes + (1 if cl2_lrv >= 1 else 0)
            if cl2_lrv >= 1:
                contributing_units.append(f"Cl2 LRV {round(cl2_lrv)}")

        if total < required_total and process_ct < reg["min_processes"]:
            status = (f"Fails: only {round(total)} LRV, needs {required_total}. only {process_ct} processes, needs {reg['min_processes']}")
        elif total < required_total and process_ct >= reg["min_processes"]:
            status = (f"Fails: only {round(total)} LRV, needs {required_total}")
        elif process_ct < reg["min_processes"]:
            status = f"Fails: only {process_ct} processes, needs {reg['min_processes']}"

        if "Fails" in status:
            raise ValueError(f"  {status}")

        process_details[pathogen] = {
            "Processes Used": contributing_units,
            "Number of Processes": process_ct,
            "Required LRV": required_total,
            "Min Required Processes": reg["min_processes"],
            "Sum of LRV": round(total),
            "Regulatory Status": status
        }

    if LRV_req["cl2_giardia_lrv_required"] >= 1 and LRV_req["cl2_virus_lrv_required"] == 0:
        LRV_req["cl2_virus_lrv_required"] = 6
        process_details["virus"]["Processes Used"].append("Cl2 LRV 6")
        process_details["virus"]["Number of Processes"] += 1
        process_details["virus"]["Sum of LRV"] += 6

    elif LRV_req["cl2_giardia_lrv_required"] >= 1 and LRV_req["cl2_virus_lrv_required"] > 0:
        LRV_req0 = LRV_req["cl2_virus_lrv_required"]
        LRV_req["cl2_virus_lrv_required"] = 6

        # NEW LRV update
        process_details["virus"]["Processes Used"] = [
            entry for entry in process_details["virus"]["Processes Used"]
            if not entry.startswith("Cl2 LRV")
        ]
        process_details["virus"]["Processes Used"].append("Cl2 LRV 6")
        process_details["virus"]["Sum of LRV"] += 6 - LRV_req0

   # CTCl_giardia = 15.257 * LRV_req["cl2_giardia_lrv_required"] + 0.1333
   # CTCl_virus = 0.5 * LRV_req["cl2_virus_lrv_required"]

    return LRV_req, process_details, state, treatment_train, effluent_type, selected_solutes

def display_initial_setting(LRV_req=None, process_details=None, solute_list=None):
    print("\n--- Required LRV Results ---")
    pprint.pprint(LRV_req)
    print("\n--- Process Summary by Pathogen ---")
    for k, v in process_details.items():
        print(f"\n{k.upper()}:")
        for key, val in v.items():
            print(f"  {key}: {val}")
    print("\n--- Solute List (Water Quality) ---")
    pprint.pprint(solute_list)

def build_nonRO(working_directory="local", state=None, solute_list=None, effluent_type=None, LRVO3_req=None, LRVClvirus_req=None, LRVClgiardia_req=None):
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
    m.fs.prop_zo = WaterParameterBlock(solute_list=solute_list)

    # define blocks
    non_RO = m.fs.non_RO = Block()

    # define flowsheet inlets and outlets
    m.fs.feed = FeedZO(property_package=m.fs.prop_zo)
    non_RO.Ozone = OzoneDPRZOv0(property_package=m.fs.prop_zo, database=m.db, state=state, effluent_type=effluent_type,
                              LRVO3_required=LRVO3_req)
    non_RO.BAF = BioActiveFiltrationDPRZO(property_package=m.fs.prop_zo, database=m.db, state=state)
    non_RO.UF = UltraFiltrationDPRZO(property_package=m.fs.prop_zo, database=m.db)
    non_RO.GAC = GACDPRZO(property_package=m.fs.prop_zo, database=m.db)
    non_RO.UV_AOP = UVAOPDPRZO(property_package=m.fs.prop_zo, database=m.db)
    non_RO.Cl = ChlorinationDPRZO(property_package=m.fs.prop_zo, database=m.db, LRVCl_required_for_virus=LRVClvirus_req, LRVCl_required_for_giardia=LRVClgiardia_req)

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
    return m

def build_RO(working_directory="local", state=None, solute_list=None, effluent_type=None, LRVO3_req=None, LRVClvirus_req=None, LRVClgiardia_req=None):
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
    m.fs.prop_zo = WaterParameterBlock(solute_list=solute_list)
    m.fs.prop_ro = SeawaterParameterBlock()

    # define blocks
    RO_pre = m.fs.RO_pre = Block()
    RO_main = m.fs.RO_main = Block()
    # RO_post = m.fs.RO_post = Block()

    # define flowsheet inlets and outlets
    m.fs.feed = FeedZO(property_package=m.fs.prop_zo)
    m.fs.treated_RO = Product(property_package=m.fs.prop_ro)

    # RO DPR components
    RO_pre.Ozone = OzoneDPRZOv0(property_package=m.fs.prop_zo, database=m.db, state=state, effluent_type=effluent_type,
                              LRVO3_required=LRVO3_req)
    RO_pre.BAF = BioActiveFiltrationDPRZO(property_package=m.fs.prop_zo, database=m.db, state=state)
    RO_pre.UF = UltraFiltrationDPRZO(property_package=m.fs.prop_zo, database=m.db)

    m.fs.byproduct_BAF = Product(property_package=m.fs.prop_zo)
    m.fs.byproduct_UF = Product(property_package=m.fs.prop_zo)

    RO_main.pump = Pump(property_package=m.fs.prop_ro)
    RO_main.RO = ReverseOsmosis0D(
                 property_package=m.fs.prop_ro,
                 has_pressure_change=True,
                 pressure_change_type=PressureChangeType.calculated,
                 mass_transfer_coefficient=MassTransferCoefficient.calculated,
                 concentration_polarization_type=ConcentrationPolarizationType.calculated,
            )

    RO_main.ERD = EnergyRecoveryDevice(property_package=m.fs.prop_ro)
    m.fs.brine = Product(property_package=m.fs.prop_ro)

    # RO_post.UV_AOP = UVAOPZO(property_package=m.fs.prop_zo, database=m.db)

    # translator blocks
    m.fs.tb_pre_main = Translator(
        inlet_property_package=m.fs.prop_zo, outlet_property_package=m.fs.prop_ro
    )

    @m.fs.tb_pre_main.Constraint(["H2O"])
    def eq_flow_mass_comp(blk, j):
        return blk.properties_in[0].flow_mass_comp[j] == blk.properties_out[0].flow_mass_phase_comp["Liq", j]

    # m.fs.tb_main_post = Translator(
    #     inlet_property_package=m.fs.prop_ro, outlet_property_package=m.fs.prop_zo
    # )
    #
    # # filtered_solute_list = [s for s in solute_list if s not in ["tds", "H2O"]]
    # # TODO: Should define specific solutes that cannot be perfectly removed by RO
    # specific_solutes = {"toc": 0.99, "nitrate": 0.90, "nitrite": 0.90} # solutes & rejection ratio (= 1-C_permeate/C_feed)
    #
    # @m.fs.tb_main_post.Constraint(solute_list)
    # def eq_flow_mass_comp_main_post(blk, j):
    #     if j == "TDS":
    #         return (
    #                 blk.properties_in[0].flow_mass_phase_comp["Liq", "TDS"]
    #                 == blk.properties_out[0].flow_mass_comp["tds"]
    #         )
    #     elif j == "H2O":
    #         return (
    #                 blk.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
    #                 == blk.properties_out[0].flow_mass_comp["H2O"]
    #         )
    #     elif j in specific_solutes:
    #         return blk.properties_out[0].flow_mass_comp[j] == (1 - specific_solutes[j]) * RO_main.RO.recovery_vol_phase[0, "Liq"] * m.fs.tb_pre_main.properties_in[0].flow_mass_comp[j]
    #     else:
    #         return blk.properties_out[0].flow_mass_comp[j] == 0

    # connections (RO DPR)
    m.fs.s_RO_feed = Arc(source=m.fs.feed.outlet, destination=RO_pre.Ozone.inlet)
    RO_pre.s01 = Arc(source=RO_pre.Ozone.treated, destination=RO_pre.BAF.inlet)
    RO_pre.s02 = Arc(source=RO_pre.BAF.treated, destination=RO_pre.UF.inlet)
    m.fs.s_UF_tb1 = Arc(source=RO_pre.UF.treated, destination=m.fs.tb_pre_main.inlet)

    m.fs.s_BAFby = Arc(source=RO_pre.BAF.byproduct, destination=m.fs.byproduct_BAF.inlet)
    m.fs.s_UFby = Arc(source=RO_pre.UF.byproduct, destination=m.fs.byproduct_UF.inlet)

    m.fs.s_tb1_RO = Arc(source=m.fs.tb_pre_main.outlet, destination=RO_main.pump.inlet)
    RO_main.s01 = Arc(source=RO_main.pump.outlet, destination=RO_main.RO.inlet)
    RO_main.s02 = Arc(source=RO_main.RO.permeate, destination=m.fs.treated_RO.inlet)
    RO_main.s03 = Arc(source=RO_main.RO.retentate, destination=RO_main.ERD.inlet)
    m.fs.s_disposal  = Arc(source=RO_main.ERD.outlet, destination=m.fs.brine.inlet)


    # m.fs.s_RO_tb2 = Arc(source=RO_main.RO.permeate, destination=m.fs.tb_main_post.inlet)
    # m.fs.s_tb2_prod = Arc(source=m.fs.tb_main_post.outlet, destination=m.fs.treated_RO.inlet)

    # m.fs.s_RO_prod = Arc(source=RO_main.RO.permeate, destination=m.fs.treated_RO.inlet)


    # m.fs.s_tb2_UVAOP = Arc(source=m.fs.tb_main_post.outlet, destination=RO_post.UV_AOP.inlet)
    # m.fs.s_UVAOP_prod = Arc(source=RO_post.UV_AOP.treated, destination=m.fs.treated_RO.inlet)

    # RO arcs expansion
    TransformationFactory("network.expand_arcs").apply_to(m)

    return m

def set_operating_conditions(m, flow_vol=10 * 0.0438, solute_list=None, treatment_train=None):
    # feed
    feed_temperature = (273.15 + 25) * pyunits.K
    feed_pressure = 101325 * pyunits.Pa
    m.fs.feed.flow_vol[0].fix(flow_vol)

    # fix solute concentrations
    for solute, value_with_unit in solute_list.items():
        m.fs.feed.conc_mass_comp[0, solute].fix(value_with_unit)
    solve(m.fs.feed)

    if treatment_train == "CBAT":  # non-RO DPR
        non_RO = m.fs.non_RO

        # Unit processes in non-RO DPR
        non_RO.Ozone.load_parameters_from_database(use_default_removal=True)
        non_RO.Ozone.contact_time[0].fix(5)

        non_RO.BAF.load_parameters_from_database(use_default_removal=True)
        non_RO.BAF.EBCT[0].fix(20)

        non_RO.UF.load_parameters_from_database(use_default_removal=True)

        non_RO.GAC.load_parameters_from_database(use_default_removal=True)
        non_RO.GAC.EBCT[0].fix(15)
        non_RO.GAC.required_BV[0].fix(20000)

        non_RO.UV_AOP.load_parameters_from_database(use_default_removal=True)
        non_RO.UV_AOP.hydrogen_peroxide_dose[0].fix(5)

        non_RO.Cl.load_parameters_from_database(use_default_removal=True)
        non_RO.Cl.contact_time[0].fix(30)

        iscale.set_variable_scaling_from_current_value(non_RO, descend_into=False)

    elif treatment_train == "RBAT":  # RO DPR
        RO_pre = m.fs.RO_pre
        RO_main = m.fs.RO_main
        # RO_post = m.fs.RO_post

        RO_pre.Ozone.load_parameters_from_database(use_default_removal=True)
        RO_pre.Ozone.contact_time[0].fix(5)

        RO_pre.BAF.load_parameters_from_database(use_default_removal=True)
        RO_pre.BAF.EBCT[0].fix(20)

        RO_pre.UF.load_parameters_from_database(use_default_removal=True)

        m.fs.tb_pre_main.properties_out[0].temperature.fix(feed_temperature)
        m.fs.tb_pre_main.properties_out[0].pressure.fix(feed_pressure)
        m.fs.tb_pre_main.properties_out[0].pressure_osm_phase[...] #touching
        flow_vol_to_RO = flow_vol
        initial_TDS = 15 # this value will be sequentially decreased in solve_RO_initial_condition function
        initial_TDS_flow_mass_phase_comp = flow_vol_to_RO * initial_TDS

        m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq","TDS"].fix(initial_TDS_flow_mass_phase_comp)
        m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq","H2O"] = m.fs.feed.properties[0].flow_mass_comp["H2O"].value

        # Pump
        RO_main.pump.efficiency_pump.fix(0.80)
        op_pressure, recovery = calculate_operating_pressure_and_recovery(feed_state_block=m.fs.tb_pre_main.properties_out[0])
        RO_main.pump.outlet.pressure[0].fix(op_pressure)
        print(f"Operating pressure set to {op_pressure * 1e-5:.2f} bar and recovery set to {recovery * 100:.2f}%")

        # RO unit
        RO_main.RO.A_comp.fix(4.2e-12)  # membrane water permeability
        RO_main.RO.B_comp.fix(3.5e-8)  # membrane salt permeability
        RO_main.RO.feed_side.channel_height.fix(1e-3)  # channel height in membrane stage [m]
        RO_main.RO.length.fix(8)  # membrane length [m]
        RO_main.RO.feed_side.spacer_porosity.fix(0.85)  # spacer porosity in membrane stage [-]
        RO_main.RO.permeate.pressure[0].fix(101325)  # atmospheric pressure [Pa]
        RO_main.RO.feed_side.velocity[0, 0].fix(0.25)

        feed_side_area_guess = flow_vol_to_RO / RO_main.RO.feed_side.velocity[0, 0].value
        width_guess = feed_side_area_guess / RO_main.RO.feed_side.spacer_porosity.value / RO_main.RO.feed_side.channel_height.value

        RO_main.RO.feed_side.area.setub(None)
        RO_main.RO.width.setub(None)
        RO_main.RO.area.setub(None)
        RO_main.RO.width.fix(width_guess)
        RO_main.RO.width.unfix()  # Unfixed variables are width and recovery
        RO_main.RO.recovery_vol_phase[0, "Liq"].fix(recovery)
        RO_main.RO.recovery_vol_phase[0, "Liq"].unfix()

        # ERD unit
        RO_main.ERD.efficiency_pump.fix(0.8)
        RO_main.ERD.control_volume.properties_out[0].pressure.fix(101325)  # Fix ERD outlet pressure to 1 atm

        # RO_post.UV_AOP.load_parameters_from_database(use_default_removal=True)

    return

def scale_system(m, treatment_train = None):
    # initialize feed
    for j in m.fs.feed.properties[0].conc_mass_comp:
        val = value(m.fs.feed.properties[0].conc_mass_comp[j])
        m.fs.prop_zo.set_default_scaling("conc_mass_comp", 1 / val, index=j)
    for j in m.fs.feed.properties[0].flow_mass_comp:
        val = value(m.fs.feed.properties[0].flow_mass_comp[j])
        m.fs.prop_zo.set_default_scaling("flow_mass_comp", 1 / val, index=j)

    if treatment_train == "CBAT":
        non_RO = m.fs.non_RO
        for prop_blk in [non_RO.Ozone.properties_in[0], non_RO.Ozone.properties_treated[0]]:
            for j in prop_blk.flow_mass_comp:
                val = value(m.fs.feed.properties[0].flow_mass_comp[j])
                iscale.set_scaling_factor(prop_blk.flow_mass_comp[j], 1 / val)
        iscale.set_scaling_factor(non_RO.Ozone.contact_time[0], 0.1)
        iscale.set_scaling_factor(non_RO.Ozone.mass_transfer_efficiency[0], 1)
        iscale.set_scaling_factor(non_RO.Ozone.specific_energy_coeff[0], 1/5)
        iscale.set_scaling_factor(non_RO.Ozone.O3toTOC[0], 1)

        for prop_blk in [non_RO.BAF.properties_in[0], non_RO.BAF.properties_treated[0]]:
            for j in prop_blk.flow_mass_comp:
                val = value(non_RO.Ozone.properties_treated[0].flow_mass_comp[j])
                iscale.set_scaling_factor(prop_blk.flow_mass_comp[j], 1 / val)
        iscale.set_scaling_factor(non_RO.BAF.EBCT[0], 1/20)
        iscale.set_scaling_factor(non_RO.BAF.energy_electric_flow_vol_inlet, 4)
        iscale.set_scaling_factor(non_RO.BAF.removal_frac_mass_comp[0, "toc"], 3)

        for prop_blk in [non_RO.UF.properties_in[0], non_RO.UF.properties_treated[0]]:
            for j in prop_blk.flow_mass_comp:
                val = value(non_RO.BAF.properties_treated[0].flow_mass_comp[j])
                iscale.set_scaling_factor(prop_blk.flow_mass_comp[j], 1 / val)
        iscale.set_scaling_factor(non_RO.UF.energy_electric_flow_vol_inlet, 1)
        iscale.set_scaling_factor(non_RO.UF.energy_membrane_replacement_flow_vol_inlet, 3)

        for prop_blk in [non_RO.GAC.properties_in[0], non_RO.GAC.properties_treated[0]]:
            for j in prop_blk.flow_mass_comp:
                val = value(non_RO.UF.properties_treated[0].flow_mass_comp[j])
                iscale.set_scaling_factor(prop_blk.flow_mass_comp[j], 1 / val)
        iscale.set_scaling_factor(non_RO.GAC.properties_in[0].conc_mass_comp["toc"], 0.1)
        iscale.set_scaling_factor(non_RO.GAC.EBCT[0], 1/15)
        iscale.set_scaling_factor(non_RO.GAC.energy_electric_flow_vol_inlet, 4)
        iscale.set_scaling_factor(non_RO.GAC.required_BV[0], 5e-5)
        iscale.set_scaling_factor(non_RO.GAC.removal_frac_mass_comp[0, "toc"], 2)

        for prop_blk in [non_RO.UV_AOP.properties_in[0], non_RO.UV_AOP.properties_treated[0]]:
            for j in prop_blk.flow_mass_comp:
                val = value(non_RO.GAC.properties_treated[0].flow_mass_comp[j])
                iscale.set_scaling_factor(prop_blk.flow_mass_comp[j], 1 / val)
        iscale.set_scaling_factor(non_RO.UV_AOP.uv_dose[0], 2e-3)
        iscale.set_scaling_factor(non_RO.UV_AOP.hydrogen_peroxide_dose[0], 1/5)
        iscale.set_scaling_factor(non_RO.UV_AOP.energy_electric_flow_vol_inlet, 10)

        for prop_blk in [non_RO.Cl.properties_in[0], non_RO.Cl.properties_treated[0]]:
            for j in prop_blk.flow_mass_comp:
                val = value(non_RO.UV_AOP.properties_treated[0].flow_mass_comp[j])
                iscale.set_scaling_factor(prop_blk.flow_mass_comp[j], 1 / val)
        iscale.set_scaling_factor(non_RO.Cl.energy_electric_flow_vol_inlet, 2e4) #todo: check this value
        # iscale.set_scaling_factor(non_RO.Cl.initial_chlorine_demand[0], 1)
        iscale.set_scaling_factor(non_RO.Cl.contact_time[0], 1/30)
        iscale.set_scaling_factor(non_RO.Cl.chlorine_decay_rate[0], 10)
        iscale.set_scaling_factor(non_RO.Cl.chlorine_dose[0], 1)

    elif treatment_train == "RBAT":
        RO_pre = m.fs.RO_pre
        RO_main = m.fs.RO_main

        for prop_blk in [RO_pre.Ozone.properties_in[0], RO_pre.Ozone.properties_treated[0]]:
            for j in prop_blk.flow_mass_comp:
                val = value(m.fs.feed.properties[0].flow_mass_comp[j])
                iscale.set_scaling_factor(prop_blk.flow_mass_comp[j], 1 / val)
        iscale.set_scaling_factor(RO_pre.Ozone.contact_time[0], 0.1)
        iscale.set_scaling_factor(RO_pre.Ozone.mass_transfer_efficiency[0], 1)
        iscale.set_scaling_factor(RO_pre.Ozone.specific_energy_coeff[0], 1 / 5)
        iscale.set_scaling_factor(RO_pre.Ozone.O3toTOC[0], 1)

        for prop_blk in [RO_pre.BAF.properties_in[0], RO_pre.BAF.properties_treated[0]]:
            for j in prop_blk.flow_mass_comp:
                val = value(RO_pre.Ozone.properties_treated[0].flow_mass_comp[j])
                iscale.set_scaling_factor(prop_blk.flow_mass_comp[j], 1 / val)
        iscale.set_scaling_factor(RO_pre.BAF.EBCT[0], 1 / 20)
        iscale.set_scaling_factor(RO_pre.BAF.energy_electric_flow_vol_inlet, 4)
        iscale.set_scaling_factor(RO_pre.BAF.removal_frac_mass_comp[0, "toc"], 3)

        for prop_blk in [RO_pre.UF.properties_in[0], RO_pre.UF.properties_treated[0]]:
            for j in prop_blk.flow_mass_comp:
                val = value(RO_pre.BAF.properties_treated[0].flow_mass_comp[j])
                iscale.set_scaling_factor(prop_blk.flow_mass_comp[j], 1 / val)
        iscale.set_scaling_factor(RO_pre.UF.energy_electric_flow_vol_inlet, 1)
        iscale.set_scaling_factor(RO_pre.UF.energy_membrane_replacement_flow_vol_inlet, 3)

        for j in m.fs.tb_pre_main.properties_in[0].flow_mass_comp:
            val = value(RO_pre.UF.properties_treated[0].flow_mass_comp[j])
            iscale.set_scaling_factor(m.fs.tb_pre_main.properties_in[0].flow_mass_comp[j], 1 / val)

        iscale.set_scaling_factor(
            m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "H2O"],
            1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "H2O"].value
        )
        iscale.set_scaling_factor(
            m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"],
            1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
        )
        iscale.set_scaling_factor(m.fs.tb_pre_main.properties_out[0].temperature, 1e-2)
        iscale.set_scaling_factor(m.fs.tb_pre_main.properties_out[0].pressure, 1e-5)

        #### Main pump ####
        op_pressure = RO_main.pump.control_volume.properties_out[0].pressure.value
        iscale.set_scaling_factor(RO_main.pump.control_volume.work, 1 / (
                op_pressure * m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value /
                RO_main.pump.efficiency_pump[0].value))
        iscale.set_scaling_factor(RO_main.pump.work_fluid[0], 1 / (
                op_pressure * m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value))
        iscale.set_scaling_factor(RO_main.pump.control_volume.properties_out[0].pressure, 1 / op_pressure)
        iscale.set_scaling_factor(RO_main.pump.control_volume.properties_in[0].pressure, 1 / 101325)
        iscale.set_scaling_factor(
            RO_main.pump.control_volume.properties_out[0].flow_vol_phase["Liq"],
            1 / m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value
        )

        #### RO unit ####
        recovery = RO_main.RO.recovery_vol_phase[0, "Liq"].value
        # === Geometric parameters of RO units ===
        iscale.set_scaling_factor(RO_main.RO.length, 1e-1)
        iscale.set_scaling_factor(RO_main.RO.feed_side.channel_height, 1e3)
        iscale.set_variable_scaling_from_current_value(RO_main.RO.width)
        calculate_variable_from_constraint(RO_main.RO.area, RO_main.RO.eq_area)
        iscale.set_variable_scaling_from_current_value(RO_main.RO.area)
        calculate_variable_from_constraint(
            RO_main.RO.feed_side.area, RO_main.RO.feed_side.eq_area
        )
        iscale.set_variable_scaling_from_current_value(RO_main.RO.feed_side.area)
        iscale.set_scaling_factor(RO_main.RO.feed_side.spacer_porosity, 1)
        iscale.set_scaling_factor(RO_main.RO.A_comp[0, "H2O"], 1e12)
        iscale.set_scaling_factor(RO_main.RO.B_comp[0, "TDS"], 1e8)
        iscale.set_scaling_factor(RO_main.RO.mixed_permeate[0].pressure, 1e-5)
        iscale.set_scaling_factor(RO_main.RO.recovery_vol_phase[0, "Liq"], 2)

        # === Operating conditions ===
        hc = RO_main.RO.feed_side.channel_height.value  # channel height [m]
        width = RO_main.RO.width.value  # width of the module [m]
        length = RO_main.RO.length.value  # length of the module [m]
        q_feed = m.fs.tb_pre_main.properties_out[0].flow_vol_phase["Liq"].value  # feed flow rate [m³/s]
        porosity = RO_main.RO.feed_side.spacer_porosity.value  # dimensionless
        dh = 4 * porosity / (2 / hc + (1 - porosity) * 8 / hc)

        # Inlet and outlet velocities
        v_inlet = q_feed / (width * hc)  # inlet velocity [m/s]
        v_exit = v_inlet * (1 - recovery)  # exit velocity [m/s]

        # === Physical properties ===
        dens = m.fs.tb_pre_main.properties_out[0].dens_mass_phase["Liq"].value  # kg/m³
        visc = m.fs.tb_pre_main.properties_out[0].visc_d_phase["Liq"].value  # Pa·s
        diff = m.fs.tb_pre_main.properties_out[0].diffus_phase_comp["Liq", "TDS"].value  # m²/s

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
        for v in RO_main.RO.feed_side.N_Sc_comp.values():
            iscale.set_scaling_factor(v, 1 / sc_in)
        for v in RO_main.RO.feed_side.N_Re.values():
            iscale.set_scaling_factor(v, 1 / re_avg)
        for v in RO_main.RO.feed_side.friction_factor_darcy.values():
            iscale.set_scaling_factor(v, 1 / f_avg)
        for v in RO_main.RO.feed_side.N_Sh_comp.values():
            sh_avg = re_avg ** 0.36 * sc_avg ** 0.36
            iscale.set_scaling_factor(v, 1 / sh_avg)

        for v in RO_main.RO.feed_side.dP_dx.values():
            iscale.set_scaling_factor(v, length / dp_avg)

        iscale.set_scaling_factor(RO_main.RO.deltaP, 1 / dp_avg)

        # Feed side (bulk)
        iscale.set_scaling_factor(RO_main.RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "H2O"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "H2O"].value
                                  )
        iscale.set_scaling_factor(RO_main.RO.feed_side.properties_out[0].flow_mass_phase_comp["Liq", "H2O"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "H2O"].value / (
                                              1 - recovery)
                                  )
        iscale.set_scaling_factor(RO_main.RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "TDS"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
                                  )
        iscale.set_scaling_factor(RO_main.RO.feed_side.properties_out[0].flow_mass_phase_comp["Liq", "TDS"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
                                  )

        # Feed side (interface)
        iscale.set_scaling_factor(RO_main.RO.feed_side.properties_interface[0, 0].flow_mass_phase_comp["Liq", "H2O"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "H2O"].value
                                  )
        iscale.set_scaling_factor(RO_main.RO.feed_side.properties_interface[0, 1].flow_mass_phase_comp["Liq", "H2O"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "H2O"].value / (
                                              1 - recovery)
                                  )
        iscale.set_scaling_factor(RO_main.RO.feed_side.properties_interface[0, 0].flow_mass_phase_comp["Liq", "TDS"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
                                  )
        iscale.set_scaling_factor(RO_main.RO.feed_side.properties_interface[0, 1].flow_mass_phase_comp["Liq", "TDS"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
                                  )

        # Permeate side
        iscale.set_scaling_factor(RO_main.RO.permeate_side[0, 0].flow_mass_phase_comp["Liq", "H2O"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp[
                                      "Liq", "H2O"].value / recovery
                                  )
        iscale.set_scaling_factor(RO_main.RO.permeate_side[0, 1].flow_mass_phase_comp["Liq", "H2O"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp[
                                      "Liq", "H2O"].value / recovery
                                  )

        iscale.set_scaling_factor(RO_main.RO.permeate_side[0, 0].flow_mass_phase_comp["Liq", "TDS"],
                                  100 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
                                  )
        iscale.set_scaling_factor(RO_main.RO.permeate_side[0, 1].flow_mass_phase_comp["Liq", "TDS"],
                                  100 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
                                  )

        iscale.set_scaling_factor(
            RO_main.RO.mixed_permeate[0].flow_mass_phase_comp["Liq", "H2O"],
            1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "H2O"].value / recovery
        )
        iscale.set_scaling_factor(
            RO_main.RO.mixed_permeate[0].flow_mass_phase_comp["Liq", "TDS"],
            100 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
        )

        iscale.set_scaling_factor(RO_main.RO.mass_transfer_phase_comp[0, "Liq", "H2O"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp[
                                      "Liq", "H2O"].value / recovery)
        iscale.set_scaling_factor(RO_main.RO.mass_transfer_phase_comp[0, "Liq", "TDS"],
                                  100 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value)

        iscale.set_scaling_factor(RO_main.RO.feed_side.properties_in[0].flow_vol_phase["Liq"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value)
        iscale.set_scaling_factor(RO_main.RO.feed_side.properties_out[0].flow_vol_phase["Liq"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value / (1 - recovery))
        iscale.set_scaling_factor(RO_main.RO.feed_side.properties_interface[0, 0].flow_vol_phase["Liq"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value)
        iscale.set_scaling_factor(RO_main.RO.feed_side.properties_interface[0, 1].flow_vol_phase["Liq"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value / (1 - recovery))
        iscale.set_scaling_factor(RO_main.RO.permeate_side[0, 0].flow_vol_phase["Liq"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value / recovery)
        iscale.set_scaling_factor(RO_main.RO.permeate_side[0, 0].flow_vol_phase["Liq"],
                                  1 / m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value / recovery)

        iscale.calculate_scaling_factors(RO_main.RO)


        # ERD unit
        iscale.set_scaling_factor(
            RO_main.ERD.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "H2O"],
            1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "H2O"].value / (1 - recovery)
        )
        iscale.set_scaling_factor(
            RO_main.ERD.control_volume.properties_out[0].flow_mass_phase_comp["Liq", "H2O"],
            1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "H2O"].value / (1 - recovery)
        )
        iscale.set_scaling_factor(
            RO_main.ERD.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "TDS"],
            100 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
        )
        iscale.set_scaling_factor(
            RO_main.ERD.control_volume.properties_out[0].flow_mass_phase_comp["Liq", "TDS"],
            100 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
        )

        iscale.set_scaling_factor(RO_main.ERD.control_volume.work, 1 / (
                op_pressure * m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value * (1 - recovery) /
                RO_main.ERD.efficiency_pump[0].value))
        iscale.set_scaling_factor(RO_main.pump.work_fluid[0], 1 / (
                op_pressure * m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value * (1 - recovery))
                                  )
        iscale.set_scaling_factor(RO_main.ERD.control_volume.properties_in[0].pressure, 1 / op_pressure)
        iscale.set_scaling_factor(RO_main.ERD.control_volume.properties_out[0].pressure, 1 / 101325)
        iscale.set_scaling_factor(
            RO_main.ERD.control_volume.properties_out[0].flow_vol_phase["Liq"],
            1 / (m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value * (1 - recovery))
        )

def initialize_system(m, solute_list = None, treatment_train = None):
    # initialize feed
    m.fs.feed.initialize()

    if treatment_train == "CBAT":  # non-RO DPR
        non_RO = m.fs.non_RO
        propagate_state(m.fs.s_non_RO_feed)

        non_RO.Ozone.initialize()
        propagate_state(non_RO.s01)

        non_RO.BAF.initialize()
        propagate_state(non_RO.s02)

        non_RO.UF.initialize()
        propagate_state(non_RO.s03)

        non_RO.GAC.initialize()
        propagate_state(non_RO.s04)

        non_RO.UV_AOP.initialize()
        propagate_state(non_RO.s05)

        non_RO.Cl.initialize()
        propagate_state(non_RO.s06)

        m.fs.treated_nonRO.initialize()

    elif treatment_train == "RBAT":  # RO DPR
        RO_pre = m.fs.RO_pre
        RO_main = m.fs.RO_main
        # RO_post = m.fs.RO_post
        propagate_state(m.fs.s_RO_feed)

        RO_pre.Ozone.initialize()
        propagate_state(RO_pre.s01)

        RO_pre.BAF.initialize()
        propagate_state(RO_pre.s02)

        RO_pre.UF.initialize()
        propagate_state(m.fs.s_UF_tb1)

        m.fs.tb_pre_main.initialize()
        propagate_state(m.fs.s_tb1_RO)

        RO_main.pump.initialize()
        propagate_state(RO_main.s01)

        RO_main.RO.initialize()
        propagate_state(RO_main.s02)
        propagate_state(RO_main.s03)

        m.fs.treated_RO.initialize()

        RO_main.ERD.initialize()
        propagate_state(m.fs.s_disposal)

        m.fs.brine.initialize()


        # # Propagate state from RO to tb_main_post
        # propagate_state(m.fs.s_RO_tb2)
        #
        # # Translate flow from tb_main_post to RO_post for all solutes
        # m.fs.tb_main_post.properties_out[0].flow_mass_comp["H2O"] = value(
        #     m.fs.tb_main_post.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
        # )
        # m.fs.tb_main_post.properties_out[0].flow_mass_comp["tds"] = value(
        #     m.fs.tb_main_post.properties_in[0].flow_mass_phase_comp["Liq", "TDS"]
        # )
        #
        # # TODO: Should define specific solutes that cannot be perfectly removed by RO
        # specific_solutes = {"toc": 0.99, "nitrate": 0.90,
        #                     "nitrite": 0.90}  # solutes & rejection ratio (= 1-C_permeate/C_feed)
        # for solute in specific_solutes:
        #     m.fs.tb_main_post.properties_out[0].flow_mass_comp[solute].set_value(
        #         (1 - specific_solutes[solute])
        #         * RO_main.RO.recovery_vol_phase[0, "Liq"].value
        #         * m.fs.tb_pre_main.properties_in[0].flow_mass_comp[solute].value
        #     )
        # filtered_solute_list = [s for s in solute_list if s not in ["tds", "H2O"] and s not in specific_solutes]
        # for solute in filtered_solute_list:
        #     m.fs.tb_main_post.properties_out[0].flow_mass_comp[solute].set_value(0)

        # for solute in specific_solutes:
        #     m.fs.tb_main_post.properties_out[0].flow_mass_comp[solute] = (1-specific_solutes[solute]) * RO_main.RO.recovery_vol_phase[0, "Liq"] * m.fs.tb_pre_main.properties_in[0].flow_mass_comp[solute]
        #
        # filtered_solute_list = [s for s in solute_list if s not in ["tds", "H2O"] or specific_solutes]
        # for solute in filtered_solute_list:
        #     m.fs.tb_main_post.properties_out[0].flow_mass_comp[solute] = 0
        #
        # propagate_state(m.fs.s_tb2_prod)

#         # # Propagate state from tb2 to RO_post
#         # propagate_state(m.fs.s_tb2_UVAOP)
#         # RO_post.UV_AOP.initialize()

    return



def adaptive_overpressure_sweep(m, solute_list=None, target_tds=None):
    print("###### Adaptive Overpressure Sweep #######")
    print(f"\nTrying target TDS: {target_tds} kg/m3")

    try:
        # First attempt: normal pressure estimation
        solve_RO_initial_conditions(
            m,
            solute_list=solute_list,
            water_recovery=0.7,
            TDS_passage=0.01,
            overpressure_sweep=False,
        )
        print(f"Successfully solved at TDS {target_tds} kg/m3 with normal pressure")
        return  # Success, so exit

    except Exception as e:
        print(f"Normal solve failed at TDS {target_tds} kg/m3: {str(e)}")

    # Fallback: try overpressure guesses
    for guess in [80, 70, 60, 50, 40, 30, 20]:
        print(f"Trying overpressure guess: {guess}")
        try:
            solve_RO_initial_conditions(
                m,
                solute_list=solute_list,
                water_recovery=0.7,
                TDS_passage=0.01,
                overpressure_sweep=True,
                overpressure_guess=guess,
            )
            print(f"Successfully solved at TDS {target_tds} kg/m3 with overpressure guess {guess}")
            return  # Success, so exit
        except Exception as e2:
            print(f"Overpressure guess {guess} failed: {str(e2)}")

    # If all attempts fail
    raise RuntimeError(
        f"\nCould not solve at the target TDS ({target_tds} kg/m3) using any overpressure guess."
    )

def solve_RO_initial_conditions(m,
                               solute_list = None,
                               water_recovery=0.7,
                               TDS_passage=0.01,
                               overpressure_sweep=False,
                               overpressure_guess=None):
    print('\nSolve for treated wastewater conditions')
    solver = get_solver()
    # update to treated wastewater

    feed_temperature = (273.15 + 25) * pyunits.K
    feed_pressure = 101325 * pyunits.Pa
    flow_vol = m.fs.feed.flow_vol[0].value

    # fix solute concentrations
    for solute, value_with_unit in solute_list.items():
        m.fs.feed.conc_mass_comp[0, solute].fix(value_with_unit)
    m.fs.feed.initialize()
    propagate_state(m.fs.s_RO_feed)

    RO_pre = m.fs.RO_pre
    RO_main = m.fs.RO_main
    RO_pre.Ozone.initialize()
    propagate_state(RO_pre.s01)

    RO_pre.BAF.initialize()
    propagate_state(RO_pre.s02)

    RO_pre.UF.initialize()
    propagate_state(m.fs.s_UF_tb1)

    flow_vol_to_RO = flow_vol
    new_TDS = value(solute_list["tds"])
    new_TDS_flow_mass_phase_comp = flow_vol_to_RO * new_TDS
    m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].unfix
    m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].fix(new_TDS_flow_mass_phase_comp)
    iscale.set_scaling_factor(
        m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"],
        1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
    )
    m.fs.tb_pre_main.initialize()
    propagate_state(m.fs.s_tb1_RO)

    #### Pump ####
    if overpressure_sweep == False:
        osm_pressure = m.fs.tb_pre_main.properties_out[0].pressure_osm_phase['Liq'].value
        try:
            operating_pressure, recovery = calculate_operating_pressure_and_recovery(
                feed_state_block=m.fs.tb_pre_main.properties_out[0],
                water_recovery=water_recovery,
                min_recovery=0.3,
                TDS_passage=TDS_passage,
                recovery_step=0.05,
                solver=None,
            )
        except:
            print('Failed to solve for operating pressure!')
            operating_pressure = osm_pressure * 10
            recovery = 0.5
    else:
        osm_pressure = m.fs.tb_pre_main.properties_out[0].pressure_osm_phase['Liq'].value
        operating_pressure = osm_pressure * overpressure_guess
        recovery = 0.5

    print('New osmotic pressure: {} bar'.format(osm_pressure / 1e5))
    print('New operating pressure: {} bar'.format(operating_pressure / 1e5))

    RO_main.pump.outlet.pressure[0].fix(operating_pressure)
    # operating_pressure = RO_main.pump.control_volume.properties_out[0].pressure.value
    iscale.set_scaling_factor(RO_main.pump.control_volume.work, 1 / (
            operating_pressure * m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value /
            RO_main.pump.efficiency_pump[0].value))
    iscale.set_scaling_factor(RO_main.pump.work_fluid[0], 1 / (
            operating_pressure * m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value))
    iscale.set_scaling_factor(RO_main.pump.control_volume.properties_out[0].pressure, 1 / operating_pressure)

    RO_main.pump.initialize()
    propagate_state(RO_main.s01)


    #### RO unit ####
    RO_main.RO.recovery_vol_phase[0, "Liq"].fix(recovery)
    RO_main.RO.recovery_vol_phase[0, "Liq"].unfix()

    # Feed side (bulk)
    iscale.set_scaling_factor(RO_main.RO.feed_side.properties_in[0].flow_mass_phase_comp["Liq", "TDS"],
                              1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
                              )
    iscale.set_scaling_factor(RO_main.RO.feed_side.properties_out[0].flow_mass_phase_comp["Liq", "TDS"],
                              1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
                              )

    # Feed side (interface)
    iscale.set_scaling_factor(RO_main.RO.feed_side.properties_interface[0, 0].flow_mass_phase_comp["Liq", "TDS"],
                              1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
                              )
    iscale.set_scaling_factor(RO_main.RO.feed_side.properties_interface[0, 1].flow_mass_phase_comp["Liq", "TDS"],
                              1 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
                              )

    # Permeate side
    iscale.set_scaling_factor(RO_main.RO.permeate_side[0, 0].flow_mass_phase_comp["Liq", "TDS"],
                              100 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
                              )
    iscale.set_scaling_factor(RO_main.RO.permeate_side[0, 1].flow_mass_phase_comp["Liq", "TDS"],
                              100 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
                              )

    iscale.set_scaling_factor(
        RO_main.RO.mixed_permeate[0].flow_mass_phase_comp["Liq", "TDS"],
        100 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
    )

    iscale.set_scaling_factor(RO_main.RO.mass_transfer_phase_comp[0, "Liq", "TDS"],
                              100 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value)

    iscale.calculate_scaling_factors(RO_main.RO)

    RO_main.RO.initialize()
    propagate_state(RO_main.s02)
    propagate_state(RO_main.s03)

    m.fs.treated_RO.initialize()

    # ERD unit
    iscale.set_scaling_factor(
        RO_main.ERD.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "TDS"],
        100 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
    )
    iscale.set_scaling_factor(
        RO_main.ERD.control_volume.properties_out[0].flow_mass_phase_comp["Liq", "TDS"],
        100 / m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].value
    )

    iscale.set_scaling_factor(RO_main.ERD.control_volume.work, 1 / (
            operating_pressure * m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value * (1 - recovery) /
            RO_main.ERD.efficiency_pump[0].value))
    iscale.set_scaling_factor(RO_main.pump.work_fluid[0], 1 / (
            operating_pressure * m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'].value * (1 - recovery))
                              )
    iscale.set_scaling_factor(RO_main.ERD.control_volume.properties_in[0].pressure, 1 / operating_pressure)

    RO_main.ERD.initialize()
    propagate_state(m.fs.s_disposal)

    m.fs.brine.initialize()

    result = solver.solve(m, tee=False)
    assert_optimal_termination(result)


def optimize_operation(m, state = None, effluent_type = None, treatment_train = None):
    if treatment_train == "CBAT": #non-RO
        non_RO = m.fs.non_RO

        non_RO.Ozone.contact_time[0].unfix()
        non_RO.Ozone.contact_time[0].setlb(5)
        non_RO.Ozone.contact_time[0].setub(10)

        if state == "CA":

            if non_RO.Ozone.config.LRVO3_required == 1:
                non_RO.Ozone.contact_time[0].fix(5) # HRT >= 1.41889 min; max(1.41889, 5)
            elif non_RO.Ozone.config.LRVO3_required == 2:
                non_RO.Ozone.contact_time[0].fix(5)  # HRT >= 3.33244 min; max(3.33244, 5)
            elif non_RO.Ozone.config.LRVO3_required == 3:
                non_RO.Ozone.contact_time[0].fix(6.38946)  # HRT >= 6.38946 min; max(3.33244, 5)
            else:
                raise ConfigurationError(
                    f"LRVO3_required value cannot be greater than 3." # LRV 4 requires at least 14.7328 min, which exceeds 10 min
                )

        else:
            non_RO.Ozone.O3toTOC[0].unfix()
            non_RO.Ozone.O3toTOC[0].setlb(0.5)
            non_RO.Ozone.O3toTOC[0].setub(1)

        non_RO.BAF.EBCT[0].unfix()
        non_RO.BAF.EBCT[0].setlb(20)
        non_RO.BAF.EBCT[0].setub(30)

        non_RO.BAF.removal_frac_mass_comp[0, "toc"].unfix()
        non_RO.BAF.removal_frac_mass_comp[0, "toc"].setlb(0.0)
        non_RO.BAF.removal_frac_mass_comp[0, "toc"].setub(1.0)

        non_RO.GAC.EBCT[0].unfix()
        non_RO.GAC.EBCT[0].setlb(10)
        non_RO.GAC.EBCT[0].setub(20)

        non_RO.GAC.required_BV[0].unfix()
        non_RO.GAC.required_BV[0].setlb(50)
        non_RO.GAC.required_BV[0].setub(24675)

        non_RO.GAC.removal_frac_mass_comp[0, "toc"].unfix()
        non_RO.GAC.removal_frac_mass_comp[0, "toc"].setlb(0.43)
        non_RO.GAC.removal_frac_mass_comp[0, "toc"].setub(1.0)

        non_RO.UV_AOP.hydrogen_peroxide_dose[0].unfix()
        non_RO.UV_AOP.hydrogen_peroxide_dose[0].setlb(4.16974)
        non_RO.UV_AOP.hydrogen_peroxide_dose[0].setub(10)

        #todo should be classified into two differnet group: RBAT and CBAT
        non_RO.UV_AOP.uv_dose[0].unfix()
        non_RO.UV_AOP.uv_dose[0].setlb(300) # reg
        non_RO.UV_AOP.uv_dose[0].setub(3500)

        non_RO.Cl.contact_time[0].unfix()
        non_RO.Cl.contact_time[0].setlb(30)
        non_RO.Cl.contact_time[0].setub(60)
        #todo in RBAT initial_chlorine demand should be also considered

        # Constraint: removal fraction expression for toc for BAF
        if hasattr(non_RO, "link_removal_frac_toc"):
            non_RO.del_component("link_removal_frac_toc")
        non_RO.link_removal_frac_toc = Constraint(
            m.fs.time,
            rule=lambda b, t: non_RO.BAF.removal_frac_mass_comp[t, "toc"] ==
                              (21.9
                               + 5.3 * (non_RO.Ozone.O3toTOC[t] - 0.48) / 0.44
                               + 3.9 * (non_RO.BAF.EBCT[t] / pyunits.minute - 14.4) / 8.5
                               + 1.4 * (non_RO.Ozone.O3toTOC[t] - 0.48) / 0.44 * (
                                           non_RO.BAF.EBCT[t] / pyunits.minute - 14.4) / 8.5
                               ) / 100
        )

        if state == "CA":
            toc_eff = pyunits.convert(0.5 * pyunits.mg / pyunits.L, to_units=pyunits.kg / pyunits.m ** 3)
        else:
            toc_eff = pyunits.convert(2 * pyunits.mg / pyunits.L, to_units=pyunits.kg / pyunits.m ** 3)

        # Constraint: TOC removal for the entire treatment train
        if hasattr(non_RO, "total_toc_removal"):
            non_RO.del_component("total_toc_removal")
        non_RO.total_toc_removal = Constraint(
            m.fs.time,
            rule=lambda b, t: (1 - non_RO.GAC.removal_frac_mass_comp[t, "toc"]) * (
                    1 - non_RO.BAF.removal_frac_mass_comp[t, "toc"]) ==
                              toc_eff / (m.fs.feed.conc_mass_comp[t, "toc"] / (
                    non_RO.Ozone.recovery_frac_mass_H2O[t] * non_RO.BAF.recovery_frac_mass_H2O[t] *
                    non_RO.UF.recovery_frac_mass_H2O[t] * non_RO.GAC.recovery_frac_mass_H2O[t]))
        )

        # Constraint: Replacement frequency constraint
        if hasattr(non_RO, "required_BV_constraint"):
            non_RO.del_component("required_BV_constraint")
        non_RO.required_BV_constraint = Constraint(
            m.fs.time,
            rule=lambda b, t: non_RO.GAC.required_BV[t] == (
                    (
                            -233248.7 * non_RO.GAC.removal_frac_mass_comp[t, "toc"] ** 3
                            + 576888.7 * non_RO.GAC.removal_frac_mass_comp[t, "toc"] ** 2
                            - 491370.9 * non_RO.GAC.removal_frac_mass_comp[t, "toc"]
                            + 147842.7
                    )
                    * (
                            292000 * (0.2823 * pyo.exp(-0.224 * pyunits.convert(
                        non_RO.GAC.properties_in[t].conc_mass_comp["toc"] / (pyunits.mg / pyunits.L),
                        to_units=pyunits.dimensionless)
                                                       ))

                    ) / 24962.88
            )
        )


    elif treatment_train == "RBAT":


        #todo: need to unfix tds mass comp and fix with the ?....

        RO_pre = m.fs.RO_pre

        RO_pre.Ozone.contact_time[0].unfix()
        RO_pre.Ozone.contact_time[0].setlb(5)
        RO_pre.Ozone.contact_time[0].setub(10)

        if state == "CA":

            if RO_pre.Ozone.config.LRVO3_required == 1:
                RO_pre.Ozone.contact_time[0].fix(5)  # HRT >= 1.41889 min; max(1.41889, 5)
            elif RO_pre.Ozone.config.LRVO3_required == 2:
                RO_pre.Ozone.contact_time[0].fix(5)  # HRT >= 3.33244 min; max(3.33244, 5)
            elif RO_pre.Ozone.config.LRVO3_required == 3:
                RO_pre.Ozone.contact_time[0].fix(6.38946)  # HRT >= 6.38946 min; max(3.33244, 5)
            else:
                raise ConfigurationError(
                    f"LRVO3_required value cannot be greater than 3."
                    # LRV 4 requires at least 14.7328 min, which exceeds 10 min
                )

        else:
            RO_pre.Ozone.O3toTOC[0].unfix()
            RO_pre.Ozone.O3toTOC[0].setlb(0.5)
            RO_pre.Ozone.O3toTOC[0].setub(1)

        RO_pre.BAF.EBCT[0].unfix()
        RO_pre.BAF.EBCT[0].setlb(20 * pyunits.minute)
        RO_pre.BAF.EBCT[0].setub(30 * pyunits.minute)

        RO_pre.BAF.removal_frac_mass_comp[0, "toc"].unfix()
        RO_pre.BAF.removal_frac_mass_comp[0, "toc"].setlb(0.0)
        RO_pre.BAF.removal_frac_mass_comp[0, "toc"].setub(1.0)

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
        RO_main.P1.control_volume.properties_out[0].pressure.setub(8300000)  # pressure vessel burst pressure
        RO_main.P1.control_volume.properties_out[0].pressure.setlb(100000)

        # RO inlet velocity
        RO_main.RO.feed_side.velocity[0, 0].unfix()
        RO_main.RO.feed_side.velocity[0, 0].setub(0.3)
        RO_main.RO.feed_side.velocity[0, 0].setlb(0.1)

        # RO membrane area
        # RO_main.RO.area.unfix()
        # RO_main.RO.area.setub(27000) #26008는 됌 --> width = 2000 m // 9는 안됌
        RO_main.RO.area.setub(1000000)
        RO_main.RO.area.setlb(50)

        RO_main.RO.width.setub(500000)

        # RO recovery - likely limited by operating pressure
        RO_main.RO.recovery_vol_phase[0, "Liq"].unfix()
        RO_main.RO.recovery_vol_phase[0, "Liq"].setub(0.90)
        RO_main.RO.recovery_vol_phase[0, "Liq"].setlb(0.10)

        # Permeate salt concentration constraint
        m.fs.RO_main.RO.mixed_permeate[0].conc_mass_phase_comp["Liq", "TDS"].setub(0.5)
        m.fs.brine.properties[0].conc_mass_phase_comp
        # Constraint: removal fraction expression for toc for BAF

        if hasattr(RO_pre, "link_removal_frac_toc"):
            RO_pre.del_component("link_removal_frac_toc")
        RO_pre.link_removal_frac_toc = Constraint(
            m.fs.time,
            rule=lambda b, t: RO_pre.BAF.removal_frac_mass_comp[t, "toc"] ==
                              (21.9
                               + 5.3 * (RO_pre.Ozone.O3toTOC[t] - 0.48) / 0.44
                               + 3.9 * (RO_pre.BAF.EBCT[t] / pyunits.minute - 14.4) / 8.5
                               + 1.4 * (RO_pre.Ozone.O3toTOC[t] - 0.48) / 0.44 * (
                                       RO_pre.BAF.EBCT[t] / pyunits.minute - 14.4) / 8.5
                               ) / 100
        )

    # m.fs.objective = Objective(expr=m.fs.LCOT_wo_revenue)
    m.fs.objective = Objective(expr=m.fs.LCOW_wo_revenue)

    return

def calculate_operating_pressure_and_recovery(
        feed_state_block=None,
        over_pressure=0.15,
        water_recovery=0.7,
        min_recovery=0.3,
        TDS_passage=0.01,
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
            value(feed_state_block.flow_mass_phase_comp["Liq", "TDS"]) * (1 - TDS_passage)
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


def solve(blk, solver=None, checkpoint=None, tee=False, fail_flag=True):
    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=tee)
    return results

def add_costing(m, treatment_train=None):
    # Zero order costing
    source_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "nonRO_DPR_global_costing.yaml",
    )

    if treatment_train == "CBAT":  # non-RO DPR costing
        non_RO = m.fs.non_RO
        m.fs.zo_costing = ZeroOrderCosting(case_study_definition=source_file)

        # non-RO DPR: cost of each unit process
        non_RO.Ozone.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
        non_RO.BAF.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
        non_RO.UF.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
        non_RO.GAC.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
        non_RO.UV_AOP.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
        non_RO.Cl.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)

        # Aggregate unit level costs and calculate overall process costs
        m.fs.zo_costing.cost_process()

        feed_flowrate = m.fs.feed.flow_vol[0]
        m.fs.zo_costing.add_electricity_intensity(feed_flowrate)
        m.fs.specific_energy_intensity = Expression(
            expr=(m.fs.zo_costing.electricity_intensity),
            doc="Specific energy consumption of the non-RO DPR treatment train on a feed flowrate basis [kWh/m3]",
        )

        # Water recovery revenue
        # TODO: check recovered_water_cost for BAF, UF, and GAC byproduct (backwash); currently this value is based on dye_desalination yaml file
        m.fs.water_recovery_revenue = Expression(
            expr=(
                    -1 * m.fs.zo_costing.utilization_factor
                    * m.fs.zo_costing.recovered_water_cost
                    * pyunits.convert(
                m.fs.byproduct_BAF.properties[0].flow_vol
                + m.fs.byproduct_UF.properties[0].flow_vol
                + m.fs.byproduct_GAC.properties[0].flow_vol,
                    to_units=pyunits.m ** 3 / m.fs.zo_costing.base_period,
                )
            ),
            doc="Savings from water recovered (BAF, UF, and GAC backwashed water (byproduct)) back to the plant",
        )

        # Combine results from costing packages and calculate overall metrics
        @m.fs.Expression(doc="Total capital cost of the non-RO DPR treatment train")
        def total_capital_cost(b):
            return pyunits.convert(
                m.fs.zo_costing.total_capital_cost, to_units=pyunits.USD_2020
            )

        @m.fs.Expression(doc="Total operating cost of the non-RO DPR treatment train")
        def total_operating_cost(b):
            return pyunits.convert(
                m.fs.zo_costing.total_fixed_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year,
            ) + pyunits.convert(
                m.fs.zo_costing.total_variable_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year,
            )

        @m.fs.Expression(doc="Total cost of water recovery")
        def total_externalities(b): # can be either - or +
            return pyunits.convert(m.fs.water_recovery_revenue, to_units=pyunits.USD_2020 / pyunits.year)

        @m.fs.Expression(
            doc="Levelized cost of the non-RO DPR treatment with respect to volumetric feed flow"
        )
        def LCOT(b):
            return (
                b.total_capital_cost * b.zo_costing.capital_recovery_factor
                + b.total_operating_cost
                + b.total_externalities
            ) / (
                pyunits.convert(
                    b.feed.properties[0].flow_vol,
                    to_units=pyunits.m ** 3 / pyunits.year,
                )
                * b.zo_costing.utilization_factor
            )

        @m.fs.Expression(
            doc="Levelized cost of the non-RO DPR treatment with respect to volumetric feed flow, not including externalities (water recovery)"
        )
        def LCOT_wo_revenue(b):
            return (
                    b.total_capital_cost * b.zo_costing.capital_recovery_factor
                    + b.total_operating_cost
            ) / (
                    pyunits.convert(
                        b.feed.properties[0].flow_vol,
                        to_units=pyunits.m ** 3 / pyunits.year,
                    )
                    * b.zo_costing.utilization_factor
            )

        @m.fs.Expression(
            doc="Levelized cost of water (non-RO DPR) with respect to volumetric treated water flow"
        )
        def LCOW(b):
            return (
                    b.total_capital_cost * b.zo_costing.capital_recovery_factor
                    + b.total_operating_cost
                    + b.total_externalities
            ) / (
                    pyunits.convert(
                        b.treated_nonRO.properties[0].flow_vol,
                        to_units=pyunits.m ** 3 / pyunits.year,
                    )
                    * b.zo_costing.utilization_factor
            )

        @m.fs.Expression(
            doc="Levelized cost of water (non-RO DPR) with respect to volumetric treated water flow, not including externalities (water recovery)"
        )
        def LCOW_wo_revenue(b):
            return (
                b.total_capital_cost * b.zo_costing.capital_recovery_factor
                + b.total_operating_cost
            ) / (
                pyunits.convert(
                    b.treated_nonRO.properties[0].flow_vol,
                    to_units=pyunits.m ** 3 / pyunits.year,
                )
                * b.zo_costing.utilization_factor
            )

    elif treatment_train == "RBAT":  # RO DPR costing
        RO_pre = m.fs.RO_pre
    #     RO_post = m.fs.RO_post
        m.fs.zo_costing = ZeroOrderCosting(case_study_definition=source_file)
        m.fs.ro_costing = WaterTAPCosting()

        #todo: need to set up costing factors

        m.fs.ro_costing.electricity_cost = value(m.fs.zo_costing.electricity_cost)
        m.fs.ro_costing.base_currency = pyunits.USD_2020

        # m.fs.ro_costing.electricity_cost = value (0.08 * pyo.units.USD_2020 / pyo.units.kWh)
        m.fs.ro_costing.wacc = value(0.09307339771758532)
        m.fs.ro_costing.plant_lifetime = value(30 * pyunits.year)
        m.fs.ro_costing.utilization_factor = value(0.9)
        m.fs.ro_costing.land_cost_percent_FCI = value(0)
        m.fs.ro_costing.working_capital_percent_FCI = value(2)
        m.fs.ro_costing.salaries_percent_FCI = value(0)
        m.fs.ro_costing.benefit_percent_of_salary = value(0)
        m.fs.ro_costing.maintenance_costs_percent_FCI = value(0.04 / pyunits.year)
        m.fs.ro_costing.laboratory_fees_percent_FCI = value(0)
        m.fs.ro_costing.insurance_and_taxes_percent_FCI = value(0)
        m.fs.ro_costing.total_investment_factor = value(3) # RO는 TIC 자체가 고려된 값이기 때문에 원래 TIC가 고려안된 ZO의 경우에는 1.3(including installation)
                                                                   # 가 고려된 최종 capital cost 가 3인데, RO는 따로 TIC가 있으니까 이렇게함
        m.fs.ro_costing.TIC = value(2 / 1.3)#2) # this is an initial value


        # RO DPR: cost of each unit process (RO_pre)
        RO_pre.Ozone.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
        RO_pre.BAF.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
        RO_pre.UF.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)

        # Aggregate unit level costs and calculate overall process costs
        m.fs.zo_costing.cost_process()

        feed_flowrate = m.fs.feed.flow_vol[0]
        m.fs.zo_costing.add_electricity_intensity(feed_flowrate)

        # RO DPR: cost of RO (RO_main)
        # RO equipment is costed using more detailed costing package
        RO_main = m.fs.RO_main
        RO_main.P1.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.ro_costing,
            costing_method_arguments={"cost_electricity_flow": True},
        )
        RO_main.RO.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.ro_costing
        )

        RO_main.M1.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.ro_costing
        )
        RO_main.PXR.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.ro_costing
        )
        RO_main.P2.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.ro_costing,
            costing_method_arguments={"cost_electricity_flow": True},
        )

        m.fs.ro_costing.cost_process()
        m.fs.ro_costing.add_specific_energy_consumption(feed_flowrate)

        m.fs.specific_energy_intensity = Expression(
            expr=(
                    m.fs.zo_costing.electricity_intensity
                    + m.fs.ro_costing.specific_energy_consumption
            ),
            doc="Specific energy consumption of the RO DPR treatment train on a feed flowrate basis [kWh/m3]",
        )

        # # Cost of each unit process (RO_post)
        # RO_post.UV_AOP.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing)
        #
        # # Aggregate unit level costs and calculate overall process costs
        # m.fs.zo_costing.cost_process()
        #
        # feed_flowrate_post = RO_post.UV_AOP.properties_in[0].flow_vol
        # m.fs.zo_costing.add_electricity_intensity(feed_flowrate_post)

        # Water recovery revenue
        # TODO: check recovered_water_cost for UF byproduct (backwash); currently this value is based on dye_desalination yaml file
        m.fs.water_recovery_revenue = Expression(
            expr=(
                    -1 * m.fs.zo_costing.utilization_factor
                    * m.fs.zo_costing.recovered_water_cost
                    * pyunits.convert(
                m.fs.byproduct_BAF.properties[0].flow_vol
                + m.fs.byproduct_UF.properties[0].flow_vol,
                to_units=pyunits.m ** 3 / m.fs.zo_costing.base_period,
            )
            ),
            doc="Savings from water recovered (BAF, UF backwashed water (byproduct)) back to the plant",
        )

        # Combine results from costing packages and calculate overall metrics
        @m.fs.Expression(doc="Total capital cost of the RO DPR treatment train")
        def total_capital_cost(b):
            return pyunits.convert(
                m.fs.zo_costing.total_capital_cost, to_units=pyunits.USD_2020
            ) + pyunits.convert(
                m.fs.ro_costing.total_capital_cost, to_units=pyunits.USD_2020
            ) # + pyunits.convert(
            #     m.fs.zo_costing.total_capital_cost, to_units=pyunits.USD_2020


        @m.fs.Expression(doc="Total operating cost of the RO DPR treatment train")
        def total_operating_cost(b):
            return pyunits.convert(
                m.fs.zo_costing.total_fixed_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year
            ) + pyunits.convert(
                m.fs.zo_costing.total_variable_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year
            ) + pyunits.convert(
                m.fs.ro_costing.total_operating_cost,
                to_units=pyunits.USD_2020 / pyunits.year,
            ) #+ pyunits.convert(
            #     m.fs.zo_costing.total_fixed_operating_cost,
            #     to_units=pyunits.USD_2020 / pyunits.year,
            # ) + pyunits.convert(
            #     m.fs.zo_costing.total_variable_operating_cost,
            #     to_units=pyunits.USD_2020 / pyunits.year,


        @m.fs.Expression(doc="Total cost of water recovery")
        def total_externalities(b): # can be either - or +
            return pyunits.convert(m.fs.water_recovery_revenue, to_units=pyunits.USD_2020 / pyunits.year)

        @m.fs.Expression(
            doc="Levelized cost of the RO DPR treatment with respect to volumetric feed flow"
        )
        def LCOT(b):
            return (
                b.total_capital_cost * b.zo_costing.capital_recovery_factor
                + b.total_operating_cost
                + b.total_externalities
            ) / (
                pyunits.convert(
                    b.feed.properties[0].flow_vol,
                    to_units=pyunits.m ** 3 / pyunits.year,
                )
                * b.zo_costing.utilization_factor
            )

        @m.fs.Expression(
            doc="Levelized cost of the RO DPR treatment with respect to volumetric feed flow, not including externalities"
        )
        def LCOT_wo_revenue(b):
            return (
                    b.total_capital_cost * b.zo_costing.capital_recovery_factor
                    + b.total_operating_cost
            ) / (
                    pyunits.convert(
                        b.feed.properties[0].flow_vol,
                        to_units=pyunits.m ** 3 / pyunits.year,
                    )
                    * b.zo_costing.utilization_factor
            )

        @m.fs.Expression(
            doc="Levelized cost of water (RO DPR) with respect to volumetric treated water flow"
        )
        def LCOW(b):
            return (
                b.total_capital_cost * b.zo_costing.capital_recovery_factor
                + b.total_operating_cost
                + b.total_externalities
            ) / (
                 pyunits.convert(
                    b.treated_RO.properties[0].flow_vol,
                    to_units=pyunits.m ** 3 / pyunits.year,
                )
                #  pyunits.convert(
                #     b.treated_RO.properties[0].flow_vol,
                #     to_units=pyunits.m ** 3 / pyunits.year,
                # )
                * b.zo_costing.utilization_factor
            )

        @m.fs.Expression(
            doc="Levelized cost of water (RO DPR) with respect to volumetric treated water flow, not including externalities"
        )
        def LCOW_wo_revenue(b):
            return (
                b.total_capital_cost * b.zo_costing.capital_recovery_factor
                + b.total_operating_cost
            ) / (
                pyunits.convert(
                    b.RO_main.RO.mixed_permeate[0].flow_vol_phase['Liq'],
                    to_units=pyunits.m ** 3 / pyunits.year,
                )
                * b.zo_costing.utilization_factor
            )

    return


def initialize_costing(m, treatment_train=None):
    if treatment_train == "CBAT":
        m.fs.zo_costing.initialize()
    elif treatment_train == "RBAT":
        m.fs.zo_costing.initialize()
        m.fs.ro_costing.initialize()
    return

def display_results(m, treatment_train=None):
    for comp in m.fs.prop_zo.component_list:
        print(comp)
    m.fs.feed.report()
    m.fs.feed.properties.report()
    print("flow_vol:", pyo.value(m.fs.feed.properties[0].flow_vol)) # param or expression으로 계산된값
    print("flow_vol:", pyo.value(m.fs.feed.flow_vol[0])) # 우리가 지정한 값

    print("conc_mass_comp H2O:", pyo.value(m.fs.feed.properties[0].conc_mass_comp["H2O"])) # param or expression으로 계산된값
    print("conc_mass_comp toc:", pyo.value(m.fs.feed.properties[0].conc_mass_comp["toc"])) # param or expression으로 계산된값
    print("conc_mass_comp toc:", pyo.value(m.fs.feed.conc_mass_comp[0, "toc"])) # 우리가 지정한 값
    # print("conc_mass_comp h2o:", pyo.value(m.fs.feed.conc_mass_comp[0, "H2O"]))

    print("flow_mass_comp H2O:", pyo.value(m.fs.feed.properties[0].flow_mass_comp["H2O"])) # Var로 선언된 값이고 (m.fs.feed.properties stateblock에)
                                                                                           # feedZO 에서 feed.flow_vol과 conc_mass_comp로 계산되는 값
    print("flow_mass_comp H2O:", pyo.value(m.fs.feed.flow_mass_comp[0, "H2O"])) # feed.py의 reference를 통해 properties[0].flow_mass_comp가 ref된 것

    print("dens_mass:", pyo.value(m.fs.feed.properties[0].dens_mass))
    m.fs.feed.properties.display()
    m.fs.feed.properties[0].display()
    m.fs.feed.properties[0].report()
    print(hasattr(m.fs.feed.properties[0], "temperature"))

    if treatment_train == "CBAT":
        print("\n----------Unit models ----------")
        unit_models = {name: block for name, block in m.fs.non_RO.component_map(Block).items() if
                       isinstance(block, UnitModelBlockData)}
        for unit_name, unit_block in unit_models.items():
            unit_block.report()

        # Flow rate
        print("\nFeed flow rate in each unit process")
        print(
            f"Feed flow rate: {value(pyunits.convert(m.fs.feed.flow_vol[0], to_units=pyunits.Mgallons / pyunits.day)): .3f} MGD")
        for unit_name, unit_block in unit_models.items():
            unit_flow_rate = value(
                pyunits.convert(unit_block.properties_in[0].flow_vol, to_units=pyunits.Mgallons / pyunits.day))
            print(f"Feed flow rate in {unit_name}: {unit_flow_rate: .3f} MGD")

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

        print(
            f"Operating conditions (Ozone): "
            f"O3:TOC = {value(m.fs.non_RO.Ozone.O3toTOC[0]):.3f}, "
            f"Contact Time = {value(pyunits.convert(m.fs.non_RO.Ozone.contact_time[0], to_units=pyunits.min)):.3f} min, "
            f"Ozone Dose= {value(pyunits.convert(m.fs.non_RO.Ozone.ozone_consumption[0], to_units=(pyunits.mg / pyunits.liter))):.3f} mg/L, "
            f"Residual O3= {(0.704 * value(m.fs.non_RO.Ozone.O3toTOC[0]) + 1.136) * value(pyo.exp(-m.fs.non_RO.Ozone.kO3[0] * (m.fs.non_RO.Ozone.contact_time[0] - 0.5 * pyunits.min))):.3f} mg/L, "

        )

        print(
            f"Operating conditions (BAF): "
            f"EBCT = {value(pyunits.convert(m.fs.non_RO.BAF.EBCT[0], to_units=pyunits.min)):.3f} min, "
            f"TOC removal (%) = {value(m.fs.non_RO.BAF.removal_frac_mass_comp[0, 'toc']) * 100:.2f}%, "
            f"TOC level in BAF influent = {value(pyunits.convert(m.fs.non_RO.BAF.properties_in[0].conc_mass_comp['toc'], to_units=pyunits.mg / pyunits.L)):.3f} mg/L"
        )

        print(
            f"Operating conditions (UF): "
            f"TOC removal (%) = {value(m.fs.non_RO.UF.removal_frac_mass_comp[0, 'toc']) * 100:.2f}%, "
            f"TOC level in UF influent = {value(pyunits.convert(m.fs.non_RO.UF.properties_in[0].conc_mass_comp['toc'], to_units=pyunits.mg / pyunits.L)):.3f} mg/L"
        )

        print(
            f"Operating conditions (GAC): "
            f"EBCT = {value(pyunits.convert(m.fs.non_RO.GAC.EBCT[0], to_units=pyunits.min)):.3f} min, "
            f"Replacement_frequnecy = {value(pyunits.convert(m.fs.non_RO.GAC.replacement_frequency[0], to_units=pyunits.min)):.3f} min, "
            f"Required_BV = {value(m.fs.non_RO.GAC.required_BV[0]):.3f} BV, "
            f"TOC removal (%) = {value(m.fs.non_RO.GAC.removal_frac_mass_comp[0, 'toc']) * 100:.2f}%, "
            f"TOC level in GAC influent = {value(pyunits.convert(m.fs.non_RO.GAC.properties_in[0].conc_mass_comp['toc'],to_units=pyunits.mg /pyunits.L)):.3f} mg/L, "
            f"TOC level in GAC effluent = {value(pyunits.convert(m.fs.non_RO.GAC.properties_treated[0].conc_mass_comp['toc'], to_units=pyunits.mg / pyunits.L)):.3f} mg/L"
        )

        print(
            f"Operating conditions (UV-AOP): "
            f"H2O2 dose = {value(pyunits.convert(m.fs.non_RO.UV_AOP.hydrogen_peroxide_dose[0], to_units=pyunits.mg /pyunits.L)):.3f} mg/L, "
            f"UV dose = {value(pyunits.convert(m.fs.non_RO.UV_AOP.uv_dose[0], to_units=pyunits.mJ /pyunits.cm **2)):.3f} mJ/cm2"
        )

        print(
            f"Operating conditions (Cl): "
            f"Cl dose = {value(pyunits.convert(m.fs.non_RO.Cl.chlorine_dose[0], to_units=pyunits.mg / pyunits.L)):.3f} mg/L, "
            f"Contact time = {value(pyunits.convert(m.fs.non_RO.Cl.contact_time[0], to_units=pyunits.min)):.3f} min, "
            f"Residual Cl = {value(pyunits.convert(m.fs.non_RO.Cl.chlorine_dose[0] + m.fs.non_RO.Cl.initial_chlorine_demand[0], to_units=pyunits.mg / pyunits.L)) * value(pyo.exp(-m.fs.non_RO.Cl.chlorine_decay_rate[0] * value(pyunits.convert(m.fs.non_RO.Cl.contact_time[0], to_units=pyunits.hour)))) :.3f} mg/L"
        )

    if treatment_train == "RBAT":
        print("\n----------Unit models in RO DPR ----------")
        unit_models = {name: block for name, block in m.fs.RO_pre.component_map(Block).items() if
                       isinstance(block, UnitModelBlockData)}
        for unit_name, unit_block in unit_models.items():
            unit_block.report()
        m.fs.RO_main.RO.report()
        # m.fs.RO_post.UV_AOP.report()

        m.fs.tb_pre_main.report()
        m.fs.tb_pre_main.properties_in[0].display()
        m.fs.tb_pre_main.properties_out[0].display()
        m.fs.RO_main.RO.feed_side.properties[0, 0].display()

        m.fs.RO_main.RO.display()
        m.fs.RO_main.RO.feed_side.properties_in[0.0].pprint()
        m.fs.RO_main.RO.feed_side.properties_in[0.0].eq_dens_mass_phase.pprint()

        # Flow rate
        print("\nFeed flow rate in each unit process")
        print(
            f"Feed flow rate: {value(pyunits.convert(m.fs.feed.flow_vol[0], to_units=pyunits.Mgallons / pyunits.day)): .3f} MGD")
        for unit_name, unit_block in unit_models.items():
            unit_flow_rate = value(
                pyunits.convert(unit_block.properties_in[0].flow_vol, to_units=pyunits.Mgallons / pyunits.day))
            print(f"Feed flow rate in {unit_name}: {unit_flow_rate: .3f} MGD")

        print(
            f"Outlet flow rate of UF: {value(pyunits.convert(m.fs.RO_pre.UF.properties_treated[0].flow_vol, to_units=pyunits.Mgallons / pyunits.day)): .3f} MGD")
        print(
            f"Flow rate of inlet of tb1: {value(pyunits.convert(m.fs.tb_pre_main.properties_in[0].flow_vol, to_units=pyunits.Mgallons / pyunits.day)): .3f} MGD")
        print(
            f"Flow rate of outlet of tb1: {value(pyunits.convert(m.fs.tb_pre_main.properties_out[0].flow_vol_phase['Liq'], to_units=pyunits.Mgallons / pyunits.day)): .3f} MGD")
        print(
            f"Feed flow rate in RO: {value(pyunits.convert(m.fs.RO_main.RO.feed_side.properties[0, 0].flow_vol_phase['Liq'], to_units=pyunits.Mgallons / pyunits.day)): .3f} MGD")
        print(
            f"density in RO: {value(pyunits.convert(m.fs.RO_main.RO.feed_side.properties[0, 0].dens_mass_phase['Liq'], to_units=pyunits.kg / pyunits.m**3)): .3f} kg/m3")

        print(
            f"Flow rate of RO permeate: {value(pyunits.convert(m.fs.RO_main.RO.mixed_permeate[0].flow_vol_phase['Liq'], to_units=pyunits.Mgallons / pyunits.day)): .3f} MGD")


        # Overall system recovery
        print("\n----------System Recovery (RO DPR)----------\n")
        sys_water_recovery = (
                m.fs.treated_RO.properties[0].flow_mass_comp["H2O"]()
                / m.fs.feed.flow_mass_comp[0, "H2O"]()
        )
        print(f"System water recovery: {sys_water_recovery * 100 : .3f}%")


        solute_list = m.fs.prop_zo.solute_set
        for solute in solute_list:
            sys_recovery = (
                    m.fs.treated_RO.properties[0].flow_mass_comp[solute]()
                    / m.fs.feed.flow_mass_comp[0, solute]()
            )
            print(f"System {solute} recovery: {sys_recovery * 100: .9f}%")

        print(
            f"Operating conditions (Ozone): "
            f"O3:TOC = {value(m.fs.RO_pre.Ozone.O3toTOC[0]):.3f}, "
            f"Contact Time = {value(pyunits.convert(m.fs.RO_pre.Ozone.contact_time[0], to_units=pyunits.min)):.3f} min, "
            f"Ozone Dose= {value(pyunits.convert(m.fs.RO_pre.Ozone.ozone_consumption[0], to_units=(pyunits.mg / pyunits.liter))):.3f} mg/L, "
            f"Residual O3= {(0.704 * value(m.fs.RO_pre.Ozone.O3toTOC[0]) + 1.136) * value(pyo.exp(-m.fs.RO_pre.Ozone.kO3[0] * (m.fs.RO_pre.Ozone.contact_time[0] - 0.5 * pyunits.min))):.3f} mg/L, "

        )

        print(
            f"Operating conditions (BAF): "
            f"EBCT = {value(pyunits.convert(m.fs.RO_pre.BAF.EBCT[0], to_units=pyunits.min)):.3f} min, "
            f"TOC removal (%) = {value(m.fs.RO_pre.BAF.removal_frac_mass_comp[0, 'toc']) * 100:.2f}%, "
            f"TOC level in BAF influent = {value(pyunits.convert(m.fs.RO_pre.BAF.properties_in[0].conc_mass_comp['toc'], to_units=pyunits.mg / pyunits.L)):.3f} mg/L"
        )

        print(
            f"Operating conditions (UF): "
            f"TOC removal (%) = {value(m.fs.RO_pre.UF.removal_frac_mass_comp[0, 'toc']) * 100:.2f}%, "
            f"TOC level in UF influent = {value(pyunits.convert(m.fs.RO_pre.UF.properties_in[0].conc_mass_comp['toc'], to_units=pyunits.mg / pyunits.L)):.3f} mg/L"
        )

        print(
            f"Operating conditions (RO): "
            f"Operating pressure_pump1 (bar) = {value(pyunits.convert(m.fs.RO_main.P1.control_volume.properties_out[0].pressure,to_units=pyunits.bar)):.2f} bar, "
            f"Operating pressure (bar) = {value(pyunits.convert(m.fs.RO_main.RO.feed_side.properties[0, 0].pressure, to_units=pyunits.bar)):.2f} bar, "
            f"Membrane feed side velocity (m/2) = {value(pyunits.convert(m.fs.RO_main.RO.feed_side.velocity[0, 0], to_units=pyunits.m / pyunits.s)):.3f} m/s, "
            f"Membrane area (m2)= {value(pyunits.convert(m.fs.RO_main.RO.area, to_units=pyunits.m**2)):.3f} m2, "
            f"Recovery (%) = {value(pyunits.convert(m.fs.RO_main.RO.recovery_vol_phase[0, 'Liq'], to_units=pyunits.dimensionless)*100):.3f}% "

            f"\nGeometry of RO membrane (RO): "
            f"Membrane width (m)= {value(pyunits.convert(m.fs.RO_main.RO.width, to_units=pyunits.m)):.3f} m, "
            f"Membrane length (m)= {value(pyunits.convert(m.fs.RO_main.RO.length, to_units=pyunits.m)):.3f} m, "
            # f"Membrane channel height (m)= {value(pyunits.convert(m.fs.RO_main.RO.channel_height, to_units=pyunits.m)):.3f} m, "
            # f"Membrane width= {value(pyunits.convert(m.fs.RO_main.RO.spacer_porosity, to_units=pyunits.dimensionless)):.3f} "
        )


        # print(
        #     f"Operating conditions (UV-AOP): "
        #     f"H2O2 dose = {value(pyunits.convert(m.fs.non_RO.UV_AOP.hydrogen_peroxide_dose[0], to_units=pyunits.mg /pyunits.L)):.3f} mg/L, "
        #     f"UV dose = {value(pyunits.convert(m.fs.non_RO.UV_AOP.uv_dose[0], to_units=pyunits.mJ /pyunits.cm **2)):.3f} mJ/cm2"
        # )
        #
        # print(
        #     f"Operating conditions (Cl): "
        #     f"Cl dose = {value(pyunits.convert(m.fs.non_RO.Cl.chlorine_dose[0], to_units=pyunits.mg / pyunits.L)):.3f} mg/L, "
        #     f"Contact time = {value(pyunits.convert(m.fs.non_RO.Cl.contact_time[0], to_units=pyunits.min)):.3f} min, "
        #     f"Residual Cl = {value(pyunits.convert(m.fs.non_RO.Cl.chlorine_dose[0] + m.fs.non_RO.Cl.initial_chlorine_demand[0], to_units=pyunits.mg / pyunits.L)) * value(pyo.exp(-m.fs.non_RO.Cl.chlorine_decay_rate[0] * value(pyunits.convert(m.fs.non_RO.Cl.contact_time[0], to_units=pyunits.hour)))) :.3f} mg/L"
        # )

def display_cost(m, treatment_train=None):
    if treatment_train == "CBAT":
        # Capex
        print("\n----------System costing metrics----------\n")
        capex = value(pyunits.convert(m.fs.total_capital_cost, to_units=pyunits.MUSD_2020))
        print(f"Total Capital Cost: {capex:.4f} M$")
        print(f"Base currency: {m.fs.zo_costing.base_currency}")
        print(f"Base currency factor: {value(pyunits.convert(1 * pyunits.USD_2014, to_units=pyunits.USD_2020))}")

        print("\n----------Unit Capital Costs----------")
        total_unit_capex_cal = 0
        for u in m.fs.zo_costing._registered_unit_costing:
            unit_name = u.parent_block().local_name
            print(
                f"{unit_name} capital cost: {value(pyunits.convert(u.capital_cost*m.fs.zo_costing.total_investment_factor, to_units=pyunits.MUSD_2020)):.4f} M$")
            total_unit_capex_cal += value(pyunits.convert(u.capital_cost*m.fs.zo_costing.total_investment_factor, to_units=pyunits.MUSD_2020))
        print(f"(Calculated) Sum of capital costs of unit processes: {total_unit_capex_cal:.4f} M$")
        print("---------------------------")

        # Opex
        opex = value(pyunits.convert(m.fs.total_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))
        total_fixed_operating_cost = value(pyunits.convert(m.fs.zo_costing.total_fixed_operating_cost,
                                                           to_units=pyunits.MUSD_2020 / pyunits.year))
        total_variable_operating_cost = value(pyunits.convert(m.fs.zo_costing.total_variable_operating_cost,
                                                              to_units=pyunits.MUSD_2020 / pyunits.year))

        print(f"\nTotal Operating Cost (Cop,tot): {opex:.4f} M$/year")
        print(f"Total fixed operating cost (Cop,fix): {total_fixed_operating_cost:.4f} M$/year")
        print(f"Total variable operating cost (Cop,fix): {total_variable_operating_cost:.4f} M$/year\n")

        # Comparison with calculated values
        total_fixed_operating_cost_cal = value(pyunits.convert(m.fs.zo_costing.aggregate_fixed_operating_cost + m.fs.zo_costing.maintenance_labor_chemical_operating_cost,
                                                               to_units=pyunits.MUSD_2020 / pyunits.year))
        total_variable_operating_cost_vop_cal = value(
            pyunits.convert(m.fs.zo_costing.aggregate_variable_operating_cost,
                            to_units=pyunits.MUSD_2020 / pyunits.year))
        print("Used flows:")
        for flow in m.fs.zo_costing.used_flows:
           print(flow)
        total_flow_cost_cal = value(
            pyunits.convert(
                sum(m.fs.zo_costing.aggregate_flow_costs[flow] for flow in m.fs.zo_costing.used_flows)
                * m.fs.zo_costing.utilization_factor,
                to_units=pyunits.MUSD_2020 / pyunits.year
            )
        )
        total_operating_cost_cal = total_fixed_operating_cost_cal + total_variable_operating_cost_vop_cal + total_flow_cost_cal
        total_variable_operating_cost_cal = total_variable_operating_cost_vop_cal + total_flow_cost_cal

        print(f"(Calculated) Total Operating Cost (Cop,tot): {total_operating_cost_cal:.4f} M$")
        print(f"(Calcualted) Total fixed operating cost (Cop,fix): {total_fixed_operating_cost_cal:.4f} M$/year")
        print(f"(Calculated) Total variable operating cost (Cop,fix): {total_variable_operating_cost_cal:.4f} M$/year")
        print(
            f"(Calculated) Total variable operating cost from unit models (Cvop,u): {total_variable_operating_cost_vop_cal:.4f} M$/year")
        print(f"(Calculated) Total flow cost (futil*Cflow,tot): {total_flow_cost_cal:.4f} M$/year")

        print("\n----------Unit Operating Costs (only flow cost)----------")

        flow_types = m.fs.zo_costing.used_flows
        util = m.fs.zo_costing.utilization_factor
        flow_cost_params = {
            ft: getattr(m.fs.zo_costing, f"{ft}_cost") for ft in flow_types
        }

        unit_blocks = {name: block for name, block in m.fs.non_RO.component_map(Block).items() if
                       isinstance(block, UnitModelBlockData)}

        total_flow_cost = 0.0

        for unit_name, unit_block in unit_blocks.items():
            unit_total = 0.0
            print(f"{unit_name}:")

            for flow_type in flow_types:
                if hasattr(unit_block, flow_type):
                    var = getattr(unit_block, flow_type)
                    if var.is_indexed():
                        v = var[0]
                    else:
                        v = var
                    cost = value(pyunits.convert(v * flow_cost_params[flow_type] * util,
                                                 to_units=pyunits.USD_2020 / pyunits.year))
                    print(f"  {flow_type} cost: {cost:,.2f} USD/year")
                    unit_total += cost

            print(f"  Total flow cost: {unit_total:,.2f} USD/year\n")
            total_flow_cost += unit_total

        print(f"Total flow cost across all units: {total_flow_cost:,.2f} USD/year")


        #
        #
        # Ozone_opex = value(
        #     pyunits.convert(m.fs.non_RO.Ozone.electricity[0] * m.fs.zo_costing.electricity_cost * m.fs.zo_costing.utilization_factor,
        #                     to_units=pyunits.USD_2020 / pyunits.year,
        #                     )
        # )
        #
        # BAF_opex = value(pyunits.convert(
        #     (
        #             pyunits.convert(m.fs.non_RO.BAF.electricity[0] * m.fs.zo_costing.electricity_cost,
        #                             to_units=pyunits.USD_2020 / pyunits.hour) +
        #             pyunits.convert(
        #                 m.fs.non_RO.BAF.activated_carbon[0] * m.fs.zo_costing.activated_carbon_cost,
        #                 to_units=pyunits.USD_2020 / pyunits.hour)
        #     ) * m.fs.zo_costing.utilization_factor,
        #     to_units=pyunits.USD_2020 / pyunits.year)
        # )
        #
        # # UF_opex = value(
        # #     pyunits.convert(
        # #         m.fs.non_RO.UF.electricity[
        # #             0] * m.fs.zo_costing.electricity_cost * m.fs.zo_costing.utilization_factor,
        # #         to_units=pyunits.USD_2020 / pyunits.year,
        # #     )
        # # )
        # #
        # # GAC_opex = value(pyunits.convert(
        # #     (
        # #             pyunits.convert(m.fs.non_RO.GAC.electricity[0] * m.fs.zo_costing.electricity_cost,
        # #                             to_units=pyunits.USD_2020 / pyunits.hour) +
        # #             pyunits.convert(
        # #                 m.fs.non_RO.GAC.activated_carbon_demand[0] * m.fs.zo_costing.activated_carbon_cost,
        # #                 to_units=pyunits.USD_2020 / pyunits.hour)
        # #     ) * m.fs.zo_costing.utilization_factor,
        # #     to_units=pyunits.USD_2020 / pyunits.year)
        # # )
        # #
        # # UV_AOP_opex = value(pyunits.convert(
        # #     (
        # #             pyunits.convert(m.fs.non_RO.UV_AOP.electricity[0] * m.fs.zo_costing.electricity_cost,
        # #                             to_units=pyunits.USD_2020 / pyunits.hour) +
        # #             pyunits.convert(
        # #                 m.fs.non_RO.UV_AOP.chemical_flow_mass[0] * m.fs.zo_costing.hydrogen_peroxide_cost,
        # #                 to_units=pyunits.USD_2020 / pyunits.hour)
        # #     ) * m.fs.zo_costing.utilization_factor,
        # #     to_units=pyunits.USD_2020 / pyunits.year)
        # # )
        # #
        # # chem_flow_mass = (
        # #         m.fs.non_RO.Cl.chlorine_dose[0] * m.fs.non_RO.Cl.properties_in[0].flow_vol
        # # )
        # # Cl_opex = value(pyunits.convert(
        # #     (
        # #             pyunits.convert(m.fs.non_RO.Cl.electricity[0] * m.fs.zo_costing.electricity_cost,
        # #                             to_units=pyunits.USD_2020 / pyunits.hour) +
        # #             pyunits.convert(chem_flow_mass * m.fs.zo_costing.chlorine_cost,
        # #                             to_units=pyunits.USD_2020 / pyunits.hour)
        # #     ) * m.fs.zo_costing.utilization_factor,
        # #     to_units=pyunits.USD_2020 / pyunits.year)
        # # )
        # #
        # Opex_dict = {"Ozone": Ozone_opex, "BAF": BAF_opex} #, "UF": UF_opex, "GAC": GAC_opex, "UV_AOP": UV_AOP_opex,
        # #              "Cl": Cl_opex}
        # for unit, unit_opex in Opex_dict.items():
        #     print(f"{unit} Opex: {unit_opex:.4f} USD/year")
        print("---------------------------")

        externalities = value(pyunits.convert(m.fs.total_externalities, to_units=pyunits.MUSD_2020 / pyunits.year))
        wrr = value(pyunits.convert(m.fs.water_recovery_revenue, to_units=pyunits.USD_2020 / pyunits.year))

        # normalized costs
        feed_flowrate = value(
            pyunits.convert(
                m.fs.feed.properties[0].flow_vol, to_units=pyunits.m ** 3 / pyunits.hr
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
        lcot_wo_revenue = value(pyunits.convert(m.fs.LCOT_wo_revenue, to_units=pyunits.USD_2020 / pyunits.m ** 3))
        lcow = value(pyunits.convert(m.fs.LCOW, to_units=pyunits.USD_2020 / pyunits.m ** 3))
        lcow_wo_revenue = value(pyunits.convert(m.fs.LCOW_wo_revenue, to_units=pyunits.USD_2020 / pyunits.m ** 3))
        sec = m.fs.specific_energy_intensity()
        #
        print(f"\nTotal Capital Cost: {capex:.4f} M$")
        print(f"\nTotal Operating Cost: {opex:.4f} M$/year")
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

        # print(f"LCOT_wo: {value(m.fs.LCOT_wo_revenue): .4f} ")
        # m.fs.pprint()

    elif treatment_train == "RBAT":
        # Capex
        print("\n----------System costing metrics (RO DPR)----------\n")
        capex = value(pyunits.convert(m.fs.total_capital_cost, to_units=pyunits.MUSD_2020))
        print(f"Total Capital Cost: {capex:.4f} M$")
        print(f"Base currency: {m.fs.zo_costing.base_currency}")
        print(f"Base currency factor: {value(pyunits.convert(1 * pyunits.USD_2014, to_units=pyunits.USD_2020))}")

        print(f"Total ZO Capital Cost: {value(pyunits.convert(m.fs.zo_costing.total_capital_cost, to_units=pyunits.MUSD_2020)):.4f} M$")
        print(f"Total investment factor of ZO: {value(m.fs.zo_costing.total_investment_factor):.4f}")
        print(f"Total RO Capital Cost: {value(pyunits.convert(m.fs.ro_costing.total_capital_cost, to_units=pyunits.MUSD_2020)):.4f} M$")
        print(f"Total investment factor of RO: {value(m.fs.ro_costing.total_investment_factor):.4f}")
        print(f"Base currency factor (RO capital): {value(pyunits.convert(1 * pyunits.USD_2018, to_units=pyunits.USD_2020))}")

        print("\n----------Unit Capital Costs----------")
        total_unit_capex_cal = 0
        for u in m.fs.zo_costing._registered_unit_costing:
            unit_name = u.parent_block().local_name
            print(
                f"{unit_name} capital cost: {value(pyunits.convert(u.capital_cost*m.fs.zo_costing.total_investment_factor, to_units=pyunits.MUSD_2020)):.4f} M$")
            total_unit_capex_cal += value(pyunits.convert(u.capital_cost*m.fs.zo_costing.total_investment_factor, to_units=pyunits.MUSD_2020))

        for u in m.fs.ro_costing._registered_unit_costing:
            unit_name = u.parent_block().local_name
            print(
                f"{unit_name} capital cost: {value(pyunits.convert(u.capital_cost*m.fs.ro_costing.total_investment_factor, to_units=pyunits.MUSD_2020)):.4f} M$")
            total_unit_capex_cal += value(pyunits.convert(u.capital_cost*m.fs.ro_costing.total_investment_factor, to_units=pyunits.MUSD_2020))

        print(f"(Calculated) Sum of capital costs of unit processes: {total_unit_capex_cal:.4f} M$")
        print("---------------------------")

        # Opex
        opex = value(pyunits.convert(m.fs.total_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))
        total_fixed_operating_cost = (value(pyunits.convert(m.fs.zo_costing.total_fixed_operating_cost,
                                                            to_units=pyunits.MUSD_2020 / pyunits.year)) )
                                      # + value(pyunits.convert(m.fs.zo_costing.total_fixed_operating_cost,
                                      #                         to_units=pyunits.MUSD_2020 / pyunits.year)))  # excluding RO
        total_variable_operating_cost = (value(pyunits.convert(m.fs.zo_costing.total_variable_operating_cost,
                                                               to_units=pyunits.MUSD_2020 / pyunits.year)) )
                                         # + value(pyunits.convert(m.fs.zo_costing.total_variable_operating_cost,
                                         #                         to_units=pyunits.MUSD_2020 / pyunits.year)))  # excluding RO

        ro_opex = value(
            pyunits.convert(m.fs.ro_costing.total_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))

        print(f"\nTotal Operating Cost (Cop,tot): {opex:.4f} M$/year")
        print(f"\nOperating Cost of RO: {ro_opex:.4f} M$/year")

        print(f"Total fixed operating cost, excluding RO  (Cop,fix): {total_fixed_operating_cost:.4f} M$/year")
        print(f"Total variable operating cost, excluding RO (Cop,fix): {total_variable_operating_cost:.4f} M$/year\n")

        # Comparison with calculated values
        total_fixed_operating_cost_cal = (value(pyunits.convert(m.fs.zo_costing.aggregate_fixed_operating_cost + m.fs.zo_costing.maintenance_labor_chemical_operating_cost,
                                                                to_units=pyunits.MUSD_2020 / pyunits.year))  )
                    #                       + value(
                    # pyunits.convert(m.fs.zo_costing.aggregate_fixed_operating_cost + m.fs.zo_costing.maintenance_labor_chemical_operating_cost,
                    #                 to_units=pyunits.MUSD_2020 / pyunits.year)))  # excluding RO
        total_variable_operating_cost_vop_cal = (value(
            pyunits.convert(m.fs.zo_costing.aggregate_variable_operating_cost,
                            to_units=pyunits.MUSD_2020 / pyunits.year))   )
                    #                              + value(
                    # pyunits.convert(m.fs.zo_costing.aggregate_variable_operating_cost,
                    #                 to_units=pyunits.MUSD_2020 / pyunits.year)))  # excluding RO

        print("Used flows:")
        for flow in m.fs.zo_costing.used_flows:
            print(flow)
        total_flow_cost_cal = value(  # excluding RO
            pyunits.convert(
                (sum(m.fs.zo_costing.aggregate_flow_costs[flow] for flow in m.fs.zo_costing.used_flows)   )
                 # + sum(m.fs.zo_costing.aggregate_flow_costs[flow] for flow in
                 #       m.fs.zo_costing.used_flows))
                * m.fs.zo_costing.utilization_factor,
                to_units=pyunits.MUSD_2020 / pyunits.year
            )
        )
        total_operating_cost_cal = total_fixed_operating_cost_cal + total_variable_operating_cost_vop_cal + total_flow_cost_cal  # excluding RO
        total_variable_operating_cost_cal = total_variable_operating_cost_vop_cal + total_flow_cost_cal  # excluding flow cost used in RO

        print(f"(Calculated) Total Operating Cost (Cop,tot): {total_operating_cost_cal:.4f} M$")
        print(f"(Calculted) Total fixed operating cost (Cop,fix): {total_fixed_operating_cost_cal:.4f} M$/year")
        print(f"(Calculated) Total variable operating cost (Cop,fix): {total_variable_operating_cost_cal:.4f} M$/year")
        print(
            f"(Calculated) Total variable operating cost from unit models (Cvop,u): {total_variable_operating_cost_vop_cal:.4f} M$/year")
        print(f"(Calculated) Total flow cost (futil*Cflow,tot): {total_flow_cost_cal:.4f} M$/year")

        print("\n----------Unit Operating Costs (only flow cost)----------")

        flow_types = m.fs.zo_costing.used_flows
        util = m.fs.zo_costing.utilization_factor
        flow_cost_params = {
            ft: getattr(m.fs.zo_costing, f"{ft}_cost") for ft in flow_types
        }

        unit_blocks = {name: block for name, block in m.fs.zo_costing.component_map(Block).items() if
                       isinstance(block, UnitModelBlockData)}

        total_flow_cost = 0.0

        for unit_name, unit_block in unit_blocks.items():
            unit_total = 0.0
            print(f"{unit_name}:")

            for flow_type in flow_types:
                if hasattr(unit_block, flow_type):
                    var = getattr(unit_block, flow_type)
                    if var.is_indexed():
                        v = var[0]
                    else:
                        v = var
                    cost = value(pyunits.convert(v * flow_cost_params[flow_type] * util,
                                                 to_units=pyunits.USD_2020 / pyunits.year))
                    print(f"  {flow_type} cost: {cost:,.2f} USD/year")
                    unit_total += cost

            print(f"  Total flow cost: {unit_total:,.2f} USD/year\n")
            total_flow_cost += unit_total

        print(f"Total flow cost across all units: {total_flow_cost:,.2f} USD/year")

        print("---------------------------")
        externalities = value(pyunits.convert(m.fs.total_externalities, to_units=pyunits.MUSD_2020 / pyunits.year))
        wrr = value(pyunits.convert(m.fs.water_recovery_revenue, to_units=pyunits.USD_2020 / pyunits.year))

        # normalized costs
        feed_flowrate = value(
            pyunits.convert(
                m.fs.feed.properties[0].flow_vol, to_units=pyunits.m ** 3 / pyunits.hr
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
        lcot_wo_revenue = value(pyunits.convert(m.fs.LCOT_wo_revenue, to_units=pyunits.USD_2020 / pyunits.m ** 3))
        lcow = value(pyunits.convert(m.fs.LCOW, to_units=pyunits.USD_2020 / pyunits.m ** 3))
        lcow_wo_revenue = value(pyunits.convert(m.fs.LCOW_wo_revenue, to_units=pyunits.USD_2020 / pyunits.m ** 3))
        sec = m.fs.specific_energy_intensity()
        #
        print(f"\nTotal Capital Cost: {capex:.4f} M$")
        print(f"\nTotal Operating Cost: {opex:.4f} M$/year")
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
    m, results = main()