"""
Non-RO DPR (O3/BAF/UF/Carbon_Adsorption/UV_AOP/Chlorination) and RO DPR (Cl/UF/RO/UV_AOP)
This module contains a zero-order representation of each unit except RO
"""

import os, math
import idaes.logger as idaeslog
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
    Feed,
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

from watertap.core.zero_order_properties import WaterParameterBlock
from watertap.core.wt_database import Database
from watertap.unit_models.zero_order import (
    FeedZO,
    OzoneDPRZOv0,
    BioActiveFiltrationDPRZO,
    UltraFiltrationDPRZO,
    GACDPRZO,
    UVAOPDPRZO,
    ChlorinationZO,
)

from watertap.costing.zero_order_costing import ZeroOrderCosting
from watertap.costing import WaterTAPCosting

import pprint

# Some more information about this module
__author__ = "Inhyeong Jeon"

# Set up logger
_log = idaeslog.getLogger(__name__)

def main(working_directory=None):
    # Define initial setting
    LRV_req, details, state, treatment_train, effluent_type, solute_list, system_capacity = DPR_initial_setting(
        state="FL", treatment_train="CBAT", effluent_type="TERTIARY", system_capacity=(10 * pyunits.Mgallons / pyunits.day)
    )

    # display_initial_setting(LRV_req=LRV_req, process_details=details, solute_list=solute_list)

    # build, set, and initialize
    if treatment_train == "CBAT":
        m = build_nonRO(working_directory=working_directory, solute_list=solute_list,
                        state=state,
                        effluent_type=effluent_type,
                        LRVO3_req=LRV_req["ozone_crypto_lrv_required"],
                        LRVClvirus_req=LRV_req["cl2_virus_lrv_required"],
                        LRVClgiardia_req=LRV_req["cl2_giardia_lrv_required"],)
    # elif treatment_train == "RBAT":
    #     m = build_RO(working_directory=working_directory, solute_list=solute_list,
    #                     state=state,
    #                     effluent_type=effluent_type,
    #                     LRVO3_req=LRV_req["ozone_crypto_lrv_required"],
    #                     LRVClvirus_req=LRV_req["cl2_virus_lrv_required"],
    #                     LRVClgiardia_req=LRV_req["cl2_giardia_lrv_required"], )

    set_operating_conditions(m, solute_list=solute_list, treatment_train=treatment_train, system_capacity=system_capacity)
    assert_units_consistent(m)

    initialize_system(m, solute_list=solute_list, treatment_train=treatment_train)
    assert_degrees_of_freedom(m, 0)

    results = solve(m, checkpoint="solve flowsheet after initializing system", tee=True)
    assert_optimal_termination(results)

    add_costing(m, treatment_train=treatment_train)
    initialize_costing(m, treatment_train=treatment_train)
    assert_degrees_of_freedom(m, 0)  # ensures problem is square

    optimize_operation(m, state=state, effluent_type=effluent_type, treatment_train=treatment_train)

    results = solve(m, checkpoint="solve flowsheet after initializing system", tee=True)
    assert_optimal_termination(results)

    # display_initial_setting(LRV_req=LRV_req, process_details=details, solute_list=solute_list)
    display_results(m, treatment_train=treatment_train)
    display_cost(m, treatment_train=treatment_train)

    return m, results

def DPR_initial_setting(state=None, treatment_train=None, effluent_type=None, system_capacity=None):  #In the future, we might need to consider another type of tratment_train for Secondary effluent (for nutrient removal)
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

    return LRV_req, process_details, state, treatment_train, effluent_type, selected_solutes, system_capacity

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

def build_nonRO(working_directory=None, state=None, solute_list=None, effluent_type=None, LRVO3_req=None, LRVClvirus_req=None, LRVClgiardia_req=None):
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
    m.fs.prop_zo = WaterParameterBlock(solute_list=solute_list)

    # define blocks
    non_RO = m.fs.non_RO = Block()

    # define flowsheet inlets and outlets
    m.fs.feed = FeedZO(property_package=m.fs.prop_zo)

    # non-RO DPR components
    non_RO.Ozone = OzoneDPRZOv0(property_package=m.fs.prop_zo, database=m.db, state=state, effluent_type=effluent_type,
                              LRVO3_required=LRVO3_req)
    non_RO.BAF = BioActiveFiltrationDPRZO(property_package=m.fs.prop_zo, database=m.db, state=state)
    non_RO.UF = UltraFiltrationDPRZO(property_package=m.fs.prop_zo, database=m.db)
    non_RO.GAC = GACDPRZO(property_package=m.fs.prop_zo, database=m.db)
    non_RO.UV_AOP = UVAOPDPRZO(property_package=m.fs.prop_zo, database=m.db)
    # non_RO.Cl = ChlorinationZO(property_package=m.fs.prop_zo, database=m.db)

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
    non_RO.s05 = Arc(source=non_RO.UV_AOP.treated, destination=m.fs.treated_nonRO.inlet)
    # non_RO.s06 = Arc(source=non_RO.Cl.treated, destination=m.fs.treated_nonRO.inlet)

    m.fs.s01 = Arc(source=non_RO.BAF.byproduct, destination=m.fs.byproduct_BAF.inlet)
    m.fs.s02 = Arc(source=non_RO.UF.byproduct, destination=m.fs.byproduct_UF.inlet)
    m.fs.s03 = Arc(source=non_RO.GAC.byproduct, destination=m.fs.byproduct_GAC.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)
    return m

def set_operating_conditions(model, solute_list=None, treatment_train=None, system_capacity=None):
    # feed
    feed_temperature = (273.15 + 25) * pyunits.K
    feed_pressure = 101325 * pyunits.Pa
    feed_flow_vol = system_capacity

    model.fs.feed.flow_vol[0].fix(feed_flow_vol)

    # fix solute concentrations
    for solute, value_with_unit in solute_list.items():
        model.fs.feed.conc_mass_comp[0, solute].fix(value_with_unit)

    iscale.set_variable_scaling_from_current_value(model.fs.feed, descend_into=False)
    solve(model.fs.feed, checkpoint="solve feed block")

    if treatment_train == "CBAT":  # non-RO DPR
        non_RO = model.fs.non_RO

        # Unit processes in non-RO DPR
        non_RO.Ozone.load_parameters_from_database(use_default_removal=True)
        non_RO.Ozone.O3toTOC[0].fix(0.5)

        # if non_RO.Ozone.config.LRVO3_required == 1:
        #     non_RO.Ozone.O3toTOC[0].fix(0.5)
        # elif non_RO.Ozone.config.LRVO3_required == 2:
        #     non_RO.Ozone.O3toTOC[0].fix(0.66)
        # elif non_RO.Ozone.config.LRVO3_required == 3:
        #     non_RO.Ozone.O3toTOC[0].fix(0.9)
        # elif non_RO.Ozone.config.LRVO3_required == 4:
        #     non_RO.Ozone.O3toTOC[0].fix(1)



        non_RO.BAF.load_parameters_from_database(use_default_removal=True)
        # non_RO.BAF.removal_frac_mass_comp[0, "toc"].unfix()
        non_RO.BAF.EBCT[0].fix(20 * pyunits.minute)
        non_RO.BAF.O3toTOC[0].fix(0.5)
        # non_RO.BAF.O3toTOC[0] = value(non_RO.Ozone.O3toTOC[0])

        non_RO.UF.load_parameters_from_database(use_default_removal=True)

        non_RO.GAC.load_parameters_from_database(use_default_removal=True)
        non_RO.GAC.EBCT[0].fix(15 * pyunits.minute)
        non_RO.GAC.required_BV[0].fix(20000)

        non_RO.UV_AOP.load_parameters_from_database(use_default_removal=True)
        non_RO.UV_AOP.hydrogen_peroxide_dose[0].fix(5 * pyunits.mg / pyunits.L)
        # non_RO.UV_AOP.uv_dose[0].fix(2000 * pyunits.mg / pyunits.L)

        # non_RO.Cl.load_parameters_from_database(use_default_removal=True)

    # elif treatment_train == "RBAT":  # RO DPR
    #     RO_pre = m.fs.RO_pre
        # RO_main = m.fs.RO_main
        # RO_post = m.fs.RO_post

        # RO_pre.Ozone.load_parameters_from_database(use_default_removal=True)
        # RO_pre.UF.load_parameters_from_database(use_default_removal=True)
        #
        # RO_main.P1.efficiency_pump.fix(0.80)
        # operating_pressure = 70e5 * pyunits.Pa
        # RO_main.P1.control_volume.properties_out[0].pressure.fix(operating_pressure)
        # RO_main.RO.A_comp.fix(4.2e-12)  # membrane water permeability
        # RO_main.RO.B_comp.fix(3.5e-8)  # membrane salt permeability
        # RO_main.RO.feed_side.channel_height.fix(1e-3)  # channel height in membrane stage [m]
        # RO_main.RO.feed_side.spacer_porosity.fix(0.97)  # spacer porosity in membrane stage [-]
        # RO_main.RO.permeate.pressure[0].fix(feed_pressure)  # atmospheric pressure [Pa]
        # RO_main.RO.feed_side.velocity[0, 0].fix(0.25)
        # RO_main.RO.recovery_vol_phase[0, "Liq"].fix(0.4)
        # m.fs.tb_pre_main.properties_out[0].temperature.fix(feed_temperature)
        # m.fs.tb_pre_main.properties_out[0].pressure.fix(feed_pressure)
        #
        # # pressure exchanger
        # RO_main.PXR.efficiency_pressure_exchanger.fix(0.95)
        # # booster pump
        # RO_main.P2.efficiency_pump.fix(0.80)
        #
        # RO_post.UV_AOP.load_parameters_from_database(use_default_removal=True)

    return

def initialize_system(model, solute_list = None, treatment_train = None):
    # initialize feed
    solve(model.fs.feed, checkpoint="solve flowsheet after initializing feed")

    seq = SequentialDecomposition()
    seq.options.tear_set = []
    seq.options.iterLim = 1

    if treatment_train == "CBAT":  # non-RO DPR
        non_RO = model.fs.non_RO
        propagate_state(model.fs.s_non_RO_feed)

        # Initialize non-RO DPR
        seq.run(non_RO, lambda u: u.initialize())
        # seq.run(non_RO, lambda u: u.initialize(solver_options={"symbolic_solver_labels": True}))
#
#     elif treatment_train == "RBAT":  # RO DPR
#         RO_pre = m.fs.RO_pre
#         # RO_main = m.fs.RO_main
#         # RO_post = m.fs.RO_post
#
#         # Propagate state from feed to Ozonation
#         propagate_state(m.fs.s_RO_feed)
#         seq.run(RO_pre, lambda u: u.initialize())
#
#         # Transfer state from UF to tb1
#         # propagate_state(m.fs.s_UF_tb1)
#         #
#         # m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "H2O"] = value(
#         #     m.fs.tb_pre_main.properties_in[0].flow_mass_comp["H2O"]
#         # )
#         # m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "TDS"] = value(
#         #     m.fs.tb_pre_main.properties_in[0].flow_mass_comp["tds"]
#         # )
#         #
#         # # Propagate state from tb_pre_main to RO
#         # propagate_state(m.fs.s_tb1_RO)
#         #
#         # # Initialize RO_main
#         # RO_main.RO.inlet.flow_mass_phase_comp[0, "Liq", "H2O"] = value(
#         #     m.fs.tb_pre_main.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]
#         # )
#         # RO_main.RO.inlet.temperature[0] = value(
#         #     m.fs.tb_pre_main.properties_out[0].temperature
#         # )
#         # RO_main.RO.inlet.pressure[0] = value(
#         #     RO_main.P1.control_volume.properties_out[0].pressure
#         # )
#         #
#         # RO_main.RO.initialize()
#         # solve(RO_main, checkpoint="solve flowsheet after initializing RO_main")
#         #
#         # # Propagate state from RO to tb_main_post
#         # propagate_state(m.fs.s_RO_tb2)
#         #
#         # # Translate flow from tb_main_post to RO_post for all solutes
#         # m.fs.tb_main_post.properties_out[0].flow_mass_comp["H2O"] = value(
#         #     m.fs.tb_main_post.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
#         # )
#         # m.fs.tb_main_post.properties_out[0].flow_mass_comp["tds"] = value(
#         #     m.fs.tb_main_post.properties_in[0].flow_mass_phase_comp["Liq", "TDS"]
#         # )
#         #
#         # filtered_solute_list = [s for s in solute_list if s not in ["tds", "H2O"]]
#         # for solute in filtered_solute_list:
#         #     m.fs.tb_main_post.properties_out[0].flow_mass_comp[solute] = 0
#         #
#         # # Propagate state from tb2 to RO_post
#         # propagate_state(m.fs.s_tb2_UVAOP)
#         # RO_post.UV_AOP.initialize()

    return

def optimize_operation(m, state = None, effluent_type = None, treatment_train = None):
    if treatment_train == "CBAT": #non-RO
        non_RO = m.fs.non_RO
        non_RO.Ozone.contact_time[0].unfix()
        if state == "CA":
            non_RO.Ozone.O3toTOC[0].fix(1)
        else:
            if effluent_type == "TERTIARY":
                if non_RO.Ozone.config.LRVO3_required == 2:
                    non_RO.Ozone.O3toTOC[0].unfix()
                    non_RO.Ozone.O3toTOC[0].setlb(0.606942)
                    non_RO.Ozone.O3toTOC[0].setub(0.701277)
                elif non_RO.Ozone.config.LRVO3_required == 3:
                    non_RO.Ozone.O3toTOC[0].unfix()
                    non_RO.Ozone.O3toTOC[0].setlb(0.7994)
                    non_RO.Ozone.O3toTOC[0].setub(1.0)
                elif non_RO.Ozone.config.LRVO3_required == 4:
                    non_RO.Ozone.O3toTOC[0].unfix()
                    non_RO.Ozone.O3toTOC[0].setlb(0.99279)
                    non_RO.Ozone.O3toTOC[0].setub(1.0)
            elif effluent_type == "SECONDARY":
                if non_RO.Ozone.config.LRVO3_required == 2:
                    non_RO.Ozone.O3toTOC[0].unfix()
                    non_RO.Ozone.O3toTOC[0].setlb(0.804291)
                    non_RO.Ozone.O3toTOC[0].setub(0.897623)

        non_RO.BAF.EBCT[0].unfix()
        non_RO.BAF.EBCT[0].setlb(20 * pyunits.minute)
        non_RO.BAF.EBCT[0].setub(30 * pyunits.minute)

        non_RO.BAF.removal_frac_mass_comp[0, "toc"].unfix()
        non_RO.BAF.removal_frac_mass_comp[0, "toc"].setlb(0.0)
        non_RO.BAF.removal_frac_mass_comp[0, "toc"].setub(1.0)

        non_RO.BAF.O3toTOC[0].unfix()
        non_RO.BAF.O3toTOC[0].setlb(0.5)
        non_RO.BAF.O3toTOC[0].setub(1)

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
        non_RO.UV_AOP.hydrogen_peroxide_dose[0].setlb(4.1)
        non_RO.UV_AOP.hydrogen_peroxide_dose[0].setub(10)

        non_RO.UV_AOP.uv_dose[0].unfix()
        non_RO.UV_AOP.uv_dose[0].setlb(300) # reg
        non_RO.UV_AOP.uv_dose[0].setub(4000)


        # Constraint: O3toTOC linkage
        if hasattr(non_RO, "link_O3toTOC"):
            non_RO.del_component("link_O3toTOC")
        non_RO.link_O3toTOC = Constraint(
            m.fs.time,
            rule=lambda b, t: non_RO.BAF.O3toTOC[t] == non_RO.Ozone.O3toTOC[t]
        )

        # Constraint: removal fraction expression for TOC for BAF
        if hasattr(non_RO.BAF, "link_removal_frac_toc"):
            non_RO.BAF.del_component("link_removal_frac_toc")
        non_RO.BAF.link_removal_frac_toc = Constraint(
            m.fs.time,
            rule=lambda b, t: non_RO.BAF.removal_frac_mass_comp[t, "toc"] ==
                              (21.9
                               + 5.3 * (non_RO.BAF.O3toTOC[t] - 0.48) / 0.44
                               + 3.9 * (non_RO.BAF.EBCT[t] / pyunits.minute - 14.4) / 8.5
                               + 1.4 * (non_RO.BAF.O3toTOC[t] - 0.48) / 0.44 * (
                                           non_RO.BAF.EBCT[t] / pyunits.minute - 14.4) / 8.5
                               ) / 100
        )

        if state == "CA":
            toc_eff = pyunits.convert(0.5 * pyunits.mg / pyunits.L, to_units=pyunits.kg / pyunits.m ** 3)
        else:
            toc_eff = pyunits.convert(2 * pyunits.mg / pyunits.L, to_units=pyunits.kg / pyunits.m ** 3)

            # Constraint: TOC removal for the entire treatment train
            if hasattr(non_RO.GAC, "total_toc_removal"):
                non_RO.GAC.del_component("total_toc_removal")
            non_RO.GAC.total_toc_removal = Constraint(
                m.fs.time,
                rule=lambda b, t: (1 - non_RO.GAC.removal_frac_mass_comp[t, "toc"]) * (
                            1 - non_RO.BAF.removal_frac_mass_comp[t, "toc"]) ==
                                  toc_eff / (m.fs.feed.conc_mass_comp[t, "toc"] / (
                            non_RO.Ozone.recovery_frac_mass_H2O[t] * non_RO.BAF.recovery_frac_mass_H2O[t] *
                            non_RO.UF.recovery_frac_mass_H2O[t] * non_RO.GAC.recovery_frac_mass_H2O[t]))
            )

        # Constraint: Replacement frequency constraint
        if hasattr(non_RO.GAC, "required_BV_constraint"):
            non_RO.GAC.del_component("required_BV_constraint")
        non_RO.GAC.required_BV_constraint = Constraint(
            m.fs.time,
            rule=lambda b, t: non_RO.GAC.required_BV[t] == (
                    (
                            -233248.7 * non_RO.GAC.removal_frac_mass_comp[t, "toc"] ** 3
                            + 576888.7 * non_RO.GAC.removal_frac_mass_comp[t, "toc"] ** 2
                            - 491370.9 * non_RO.GAC.removal_frac_mass_comp[t, "toc"]
                            + 147842.7
                    )
                    * (
                            292000 * (pyunits.convert(non_RO.GAC.properties_in[t].conc_mass_comp["toc"],to_units=pyunits.mg /pyunits.L)/ (1 * pyunits.mg / pyunits.L)) ** (-1.49)
                    )
                    / 24962.88
            )
        )






    # elif treatment_train == "RBAT":
    #
    #     """
    #     Unfixes RO operating conditions and sets solver objective
    #         - Operating pressure: 1 - 83 bar
    #         - Crossflow velocity: 10 - 30 cm/s
    #         - Membrane area: 50 - 5000 m2
    #         - Volumetric recovery: 10 - 75 %
    #     """
    #     RO_main = m.fs.RO_main
    #
    #     # RO operating pressure
    #     RO_main.P1.control_volume.properties_out[0].pressure.unfix()
    #     RO_main.P1.control_volume.properties_out[0].pressure.setub(
    #         8300000
    #     )  # pressure vessel burst pressure
    #     RO_main.P1.control_volume.properties_out[0].pressure.setlb(100000)
    #
    #     # RO inlet velocity
    #     RO_main.RO.feed_side.velocity[0, 0].unfix()
    #     RO_main.RO.feed_side.velocity[0, 0].setub(0.3)
    #     RO_main.RO.feed_side.velocity[0, 0].setlb(0.1)
    #
    #     # RO membrane area
    #     RO_main.RO.area.unfix()
    #     RO_main.RO.area.setub(5000)
    #     RO_main.RO.area.setlb(50)
    #
    #     # RO recovery - likely limited by operating pressure
    #     RO_main.RO.recovery_vol_phase[0, "Liq"].unfix()
    #     RO_main.RO.recovery_vol_phase[0, "Liq"].setub(0.75)
    #     RO_main.RO.recovery_vol_phase[0, "Liq"].setlb(0.10)
    #
    #     # Permeate salt concentration constraint
    #     m.fs.RO_main.RO.mixed_permeate[0].conc_mass_phase_comp["Liq", "TDS"].setub(0.5)
    #     m.fs.brine.properties[0].conc_mass_phase_comp

    m.fs.objective = Objective(expr=m.fs.LCOT_wo_revenue)

    return

def solve(blk, solver=None, checkpoint=None, tee=False, fail_flag=True):
    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=tee)
    return results

def add_costing(model, treatment_train=None):
    # Zero order costing
    source_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "nonRO_DPR_global_costing.yaml",
    )

    if treatment_train == "CBAT":  # non-RO DPR costing
        non_RO = model.fs.non_RO
        model.fs.zo_costing_nonRO = ZeroOrderCosting(case_study_definition=source_file)

        # non-RO DPR: cost of each unit process
        non_RO.Ozone.costing = UnitModelCostingBlock(flowsheet_costing_block=model.fs.zo_costing_nonRO)
        non_RO.BAF.costing = UnitModelCostingBlock(flowsheet_costing_block=model.fs.zo_costing_nonRO)
        non_RO.UF.costing = UnitModelCostingBlock(flowsheet_costing_block=model.fs.zo_costing_nonRO)
        non_RO.GAC.costing = UnitModelCostingBlock(flowsheet_costing_block=model.fs.zo_costing_nonRO)
        non_RO.UV_AOP.costing = UnitModelCostingBlock(flowsheet_costing_block=model.fs.zo_costing_nonRO)
        # non_RO.Cl.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing_nonRO)

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

    # elif treatment_train == "RBAT":  # RO DPR costing
    #     RO_pre = m.fs.RO_pre
    #     RO_post = m.fs.RO_post
    #     m.fs.zo_costing_RO_pre = ZeroOrderCosting(case_study_definition=source_file)
    #     m.fs.ro_costing = WaterTAPCosting()
    #     m.fs.zo_costing_RO_post = ZeroOrderCosting(case_study_definition=source_file)
    #
    #     # RO DPR: cost of each unit process (RO_pre)
    #     RO_pre.Cl.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing_RO_pre)
    #     RO_pre.UF.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing_RO_pre)
    #
    #     # Aggregate unit level costs and calculate overall process costs
    #     m.fs.zo_costing_RO_pre.cost_process()
    #
    #     feed_flowrate = m.fs.feed.flow_vol[0]
    #     m.fs.zo_costing_RO_pre.add_electricity_intensity(feed_flowrate)
    #
    #     # RO DPR: cost of RO (RO_main)
    #     # RO equipment is costed using more detailed costing package
    #     RO_main = m.fs.RO_main
    #     RO_main.P1.costing = UnitModelCostingBlock(
    #         flowsheet_costing_block=m.fs.ro_costing,
    #         costing_method_arguments={"cost_electricity_flow": True},
    #     )
    #     RO_main.RO.costing = UnitModelCostingBlock(
    #         flowsheet_costing_block=m.fs.ro_costing
    #     )
    #
    #     RO_main.M1.costing = UnitModelCostingBlock(
    #         flowsheet_costing_block=m.fs.ro_costing
    #     )
    #     RO_main.PXR.costing = UnitModelCostingBlock(
    #         flowsheet_costing_block=m.fs.ro_costing
    #     )
    #     RO_main.P2.costing = UnitModelCostingBlock(
    #         flowsheet_costing_block=m.fs.ro_costing,
    #         costing_method_arguments={"cost_electricity_flow": True},
    #     )
    #
    #     m.fs.ro_costing.electricity_cost = value(m.fs.zo_costing_RO_pre.electricity_cost)
    #     m.fs.ro_costing.base_currency = pyunits.USD_2020
    #
    #     m.fs.ro_costing.cost_process()
    #
    #     m.fs.ro_costing.add_specific_energy_consumption(feed_flowrate)
    #
    #     m.fs.specific_energy_intensity = Expression(
    #         expr=(
    #                 m.fs.zo_costing_RO_pre.electricity_intensity
    #                 + m.fs.ro_costing.specific_energy_consumption
    #         ),
    #         doc="Specific energy consumption of the RO DPR treatment train on a feed flowrate basis [kWh/m3]",
    #     )
    #
    #     # Cost of each unit process (RO_post)
    #     RO_post.UV_AOP.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.zo_costing_RO_post)
    #
    #     # Aggregate unit level costs and calculate overall process costs
    #     m.fs.zo_costing_RO_post.cost_process()
    #
    #     feed_flowrate_post = RO_post.UV_AOP.properties_in[0].flow_vol
    #     m.fs.zo_costing_RO_post.add_electricity_intensity(feed_flowrate_post)
    #
    #     # Water recovery revenue
    #     # TODO: check recovered_water_cost for UF byproduct (backwash); currently this value is based on dye_desalination yaml file
    #     m.fs.water_recovery_revenue = Expression(
    #         expr=(
    #                 -1 * m.fs.zo_costing_RO_pre.utilization_factor
    #                 * m.fs.zo_costing_RO_pre.recovered_water_cost
    #                 * pyunits.convert(m.fs.byproduct_UF.properties[0].flow_vol,
    #             to_units=pyunits.m ** 3 / m.fs.zo_costing_RO_pre.base_period,
    #         )
    #         ),
    #         doc="Savings from water recovered (UF backwashed water (byproduct)) back to the plant",
    #     )
    #
    #     # Combine results from costing packages and calculate overall metrics
    #     @m.fs.Expression(doc="Total capital cost of the RO DPR treatment train")
    #     def total_capital_cost(b):
    #         return pyunits.convert(
    #             m.fs.zo_costing_RO_pre.total_capital_cost, to_units=pyunits.USD_2020
    #         ) + pyunits.convert(
    #             m.fs.ro_costing.total_capital_cost, to_units=pyunits.USD_2020
    #         ) + pyunits.convert(
    #             m.fs.zo_costing_RO_post.total_capital_cost, to_units=pyunits.USD_2020
    #         )
    #
    #     @m.fs.Expression(doc="Total operating cost of the RO DPR treatment train")
    #     def total_operating_cost(b):
    #         return pyunits.convert(
    #             m.fs.zo_costing_RO_pre.total_fixed_operating_cost,
    #             to_units=pyunits.USD_2020 / pyunits.year,
    #         ) + pyunits.convert(
    #             m.fs.zo_costing_RO_pre.total_variable_operating_cost,
    #             to_units=pyunits.USD_2020 / pyunits.year,
    #         ) + pyunits.convert(
    #             m.fs.ro_costing.total_operating_cost,
    #             to_units=pyunits.USD_2020 / pyunits.year,
    #         ) + pyunits.convert(
    #             m.fs.zo_costing_RO_post.total_fixed_operating_cost,
    #             to_units=pyunits.USD_2020 / pyunits.year,
    #         ) + pyunits.convert(
    #             m.fs.zo_costing_RO_post.total_variable_operating_cost,
    #             to_units=pyunits.USD_2020 / pyunits.year,
    #         )
    #
    #     @m.fs.Expression(doc="Total cost of water recovery")
    #     def total_externalities(b): # can be either - or +
    #         return pyunits.convert(m.fs.water_recovery_revenue, to_units=pyunits.USD_2020 / pyunits.year)
    #
    #     @m.fs.Expression(
    #         doc="Levelized cost of the RO DPR treatment with respect to volumetric feed flow"
    #     )
    #     def LCOT(b):
    #         return (
    #             b.total_capital_cost * b.zo_costing_RO_pre.capital_recovery_factor
    #             + b.total_operating_cost
    #             + b.total_externalities
    #         ) / (
    #             pyunits.convert(
    #                 b.feed.properties[0].flow_vol,
    #                 to_units=pyunits.m ** 3 / pyunits.year,
    #             )
    #             * b.zo_costing_RO_pre.utilization_factor
    #         )
    #
    #     @m.fs.Expression(
    #         doc="Levelized cost of the RO DPR treatment with respect to volumetric feed flow, not including externalities"
    #     )
    #     def LCOT_wo_revenue(b):
    #         return (
    #                 b.total_capital_cost * b.zo_costing_RO_pre.capital_recovery_factor
    #                 + b.total_operating_cost
    #         ) / (
    #                 pyunits.convert(
    #                     b.feed.properties[0].flow_vol,
    #                     to_units=pyunits.m ** 3 / pyunits.year,
    #                 )
    #                 * b.zo_costing_RO_pre.utilization_factor
    #         )
    #
    #     @m.fs.Expression(
    #         doc="Levelized cost of water (RO DPR) with respect to volumetric treated water flow"
    #     )
    #     def LCOW(b):
    #         return (
    #             b.total_capital_cost * b.zo_costing_RO_pre.capital_recovery_factor
    #             + b.total_operating_cost
    #             + b.total_externalities
    #         ) / (
    #             pyunits.convert(
    #                 b.treated_RO.properties[0].flow_vol,
    #                 to_units=pyunits.m ** 3 / pyunits.year,
    #             )
    #             * b.zo_costing_RO_pre.utilization_factor
    #         )
    #
    #     @m.fs.Expression(
    #         doc="Levelized cost of water (RO DPR) with respect to volumetric treated water flow, not including externalities"
    #     )
    #     def LCOW_wo_revenue(b):
    #         return (
    #             b.total_capital_cost * b.zo_costing_RO_pre.capital_recovery_factor
    #             + b.total_operating_cost
    #         ) / (
    #             pyunits.convert(
    #                 b.treated_RO.properties[0].flow_vol,
    #                 to_units=pyunits.m ** 3 / pyunits.year,
    #             )
    #             * b.zo_costing_RO_pre.utilization_factor
    #         )

    return

def initialize_costing(model, treatment_train=None):
    if treatment_train == "CBAT":
        model.fs.zo_costing_nonRO.initialize()
    elif treatment_train == "RBAT":
        model.fs.zo_costing_RO_pre.initialize()
        model.fs.ro_costing.initialize()
        model.fs.zo_costing_RO_post.initialize()
    return

def display_results(m, treatment_train=None):
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
            f"Contact Time = {value(pyunits.convert(m.fs.non_RO.Ozone.contact_time[0], to_units=pyunits.min)):.3f} min,"
            f"Ozone Dose= {value(pyunits.convert(m.fs.non_RO.Ozone.ozone_consumption[0], to_units=(pyunits.mg / pyunits.liter))):.3f} mg/L"
        )

        print(
            f"Operating conditions (BAF): "
            f"O3:TOC = {value(m.fs.non_RO.BAF.O3toTOC[0]):.3f}, "
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

def display_results_nonRO_DPR(m):
    print("\n----------Unit models in non-RO DPR ----------")
    unit_models = {name: block for name, block in m.fs.non_RO.component_map(Block).items() if
                   isinstance(block, UnitModelBlockData)}
    for unit_name, unit_block in unit_models.items():
        unit_block.report()

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

def display_cost(m, treatment_train=None):
    if treatment_train == "CBAT":
        # Capex
        print("\n----------System costing metrics----------\n")
        capex = value(pyunits.convert(m.fs.total_capital_cost, to_units=pyunits.MUSD_2020))
        print(f"Total Capital Cost: {capex:.4f} M$")
        print(f"Base currency: {m.fs.zo_costing_nonRO.base_currency}")
        print(f"Base currency factor: {value(pyunits.convert(1 * pyunits.USD_2014, to_units=pyunits.USD_2020))}")

        print("\n----------Unit Capital Costs----------")
        total_unit_capex_cal = 0
        for u in m.fs.zo_costing_nonRO._registered_unit_costing:
            unit_name = u.parent_block().local_name
            print(
                f"{unit_name} capital cost: {value(pyunits.convert(u.capital_cost*m.fs.zo_costing_nonRO.total_investment_factor, to_units=pyunits.MUSD_2020)):.4f} M$")
            total_unit_capex_cal += value(pyunits.convert(u.capital_cost*m.fs.zo_costing_nonRO.total_investment_factor, to_units=pyunits.MUSD_2020))
        print(f"(Calculated) Sum of capital costs of unit processes: {total_unit_capex_cal:.4f} M$")
        print("---------------------------")

        # Opex
        opex = value(pyunits.convert(m.fs.total_operating_cost, to_units=pyunits.MUSD_2020 / pyunits.year))
        total_fixed_operating_cost = value(pyunits.convert(m.fs.zo_costing_nonRO.total_fixed_operating_cost,
                                                           to_units=pyunits.MUSD_2020 / pyunits.year))
        total_variable_operating_cost = value(pyunits.convert(m.fs.zo_costing_nonRO.total_variable_operating_cost,
                                                              to_units=pyunits.MUSD_2020 / pyunits.year))

        print(f"\nTotal Operating Cost (Cop,tot): {opex:.4f} M$/year")
        print(f"Total fixed operating cost (Cop,fix): {total_fixed_operating_cost:.4f} M$/year")
        print(f"Total variable operating cost (Cop,fix): {total_variable_operating_cost:.4f} M$/year\n")

        # Comparison with calculated values
        total_fixed_operating_cost_cal = value(pyunits.convert(m.fs.zo_costing_nonRO.aggregate_fixed_operating_cost + m.fs.zo_costing_nonRO.maintenance_labor_chemical_operating_cost,
                                                               to_units=pyunits.MUSD_2020 / pyunits.year))
        total_variable_operating_cost_vop_cal = value(
            pyunits.convert(m.fs.zo_costing_nonRO.aggregate_variable_operating_cost,
                            to_units=pyunits.MUSD_2020 / pyunits.year))
        print("Used flows:")
        for flow in m.fs.zo_costing_nonRO.used_flows:
           print(flow)
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
        print(
            f"(Calculated) Total variable operating cost from unit models (Cvop,u): {total_variable_operating_cost_vop_cal:.4f} M$/year")
        print(f"(Calculated) Total flow cost (futil*Cflow,tot): {total_flow_cost_cal:.4f} M$/year")

        print("\n----------Unit Operating Costs (only flow cost)----------")

        flow_types = m.fs.zo_costing_nonRO.used_flows
        util = m.fs.zo_costing_nonRO.utilization_factor
        flow_cost_params = {
            ft: getattr(m.fs.zo_costing_nonRO, f"{ft}_cost") for ft in flow_types
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
        #     pyunits.convert(m.fs.non_RO.Ozone.electricity[0] * m.fs.zo_costing_nonRO.electricity_cost * m.fs.zo_costing_nonRO.utilization_factor,
        #                     to_units=pyunits.USD_2020 / pyunits.year,
        #                     )
        # )
        #
        # BAF_opex = value(pyunits.convert(
        #     (
        #             pyunits.convert(m.fs.non_RO.BAF.electricity[0] * m.fs.zo_costing_nonRO.electricity_cost,
        #                             to_units=pyunits.USD_2020 / pyunits.hour) +
        #             pyunits.convert(
        #                 m.fs.non_RO.BAF.activated_carbon[0] * m.fs.zo_costing_nonRO.activated_carbon_cost,
        #                 to_units=pyunits.USD_2020 / pyunits.hour)
        #     ) * m.fs.zo_costing_nonRO.utilization_factor,
        #     to_units=pyunits.USD_2020 / pyunits.year)
        # )
        #
        # # UF_opex = value(
        # #     pyunits.convert(
        # #         m.fs.non_RO.UF.electricity[
        # #             0] * m.fs.zo_costing_nonRO.electricity_cost * m.fs.zo_costing_nonRO.utilization_factor,
        # #         to_units=pyunits.USD_2020 / pyunits.year,
        # #     )
        # # )
        # #
        # # GAC_opex = value(pyunits.convert(
        # #     (
        # #             pyunits.convert(m.fs.non_RO.GAC.electricity[0] * m.fs.zo_costing_nonRO.electricity_cost,
        # #                             to_units=pyunits.USD_2020 / pyunits.hour) +
        # #             pyunits.convert(
        # #                 m.fs.non_RO.GAC.activated_carbon_demand[0] * m.fs.zo_costing_nonRO.activated_carbon_cost,
        # #                 to_units=pyunits.USD_2020 / pyunits.hour)
        # #     ) * m.fs.zo_costing_nonRO.utilization_factor,
        # #     to_units=pyunits.USD_2020 / pyunits.year)
        # # )
        # #
        # # UV_AOP_opex = value(pyunits.convert(
        # #     (
        # #             pyunits.convert(m.fs.non_RO.UV_AOP.electricity[0] * m.fs.zo_costing_nonRO.electricity_cost,
        # #                             to_units=pyunits.USD_2020 / pyunits.hour) +
        # #             pyunits.convert(
        # #                 m.fs.non_RO.UV_AOP.chemical_flow_mass[0] * m.fs.zo_costing_nonRO.hydrogen_peroxide_cost,
        # #                 to_units=pyunits.USD_2020 / pyunits.hour)
        # #     ) * m.fs.zo_costing_nonRO.utilization_factor,
        # #     to_units=pyunits.USD_2020 / pyunits.year)
        # # )
        # #
        # # chem_flow_mass = (
        # #         m.fs.non_RO.Cl.chlorine_dose[0] * m.fs.non_RO.Cl.properties_in[0].flow_vol
        # # )
        # # Cl_opex = value(pyunits.convert(
        # #     (
        # #             pyunits.convert(m.fs.non_RO.Cl.electricity[0] * m.fs.zo_costing_nonRO.electricity_cost,
        # #                             to_units=pyunits.USD_2020 / pyunits.hour) +
        # #             pyunits.convert(chem_flow_mass * m.fs.zo_costing_nonRO.chlorine_cost,
        # #                             to_units=pyunits.USD_2020 / pyunits.hour)
        # #     ) * m.fs.zo_costing_nonRO.utilization_factor,
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
    model, results = main(working_directory="local")