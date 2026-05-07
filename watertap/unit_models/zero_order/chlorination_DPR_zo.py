#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
"""
This module contains a zero-order representation of a Chlorination unit.
"""

import pyomo.environ as pyo
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.environ import units as pyunits, Var
from idaes.core import declare_process_block_class
from watertap.core import build_siso, constant_intensity, ZeroOrderBaseData

# Some more information about this module
__author__ = "Inhyeong Jeon"


@declare_process_block_class("ChlorinationDPRZO")
class ChlorinationDPRZOData(ZeroOrderBaseData):
    """
    Zero-Order m for a Chlorination unit operation.
    """

    CONFIG = ZeroOrderBaseData.CONFIG()

    CONFIG.declare("LRVCl_required_for_virus", ConfigValue(
        default=None,
        domain=None,
        description="Required LRV for virus"
    ))

    CONFIG.declare("LRVCl_required_for_giardia", ConfigValue(
        default=None,
        domain=None,
        description="Required LRV for Giardia"
    ))

    def build(self):
        super().build()
        self._tech_type = "chlorination_DPR"
        build_siso(self)

        self.energy_electric_flow_vol_inlet = Var(
            units=pyunits.kWh / pyunits.m ** 3,
            doc="Electricity intensity with respect to inlet flowrate of unit",
        )

        if self.config.LRVCl_required_for_virus == 0 and self.config.LRVCl_required_for_giardia == 0:
            self.energy_electric_flow_vol_inlet = 0 #* pyunits.kWh / pyunits.m ** 3,

        self._fixed_perf_vars.append(self.energy_electric_flow_vol_inlet)

        #todo need to consider this for RBAT
        self.initial_chlorine_demand = Var(
            self.flowsheet().time,
            units=pyunits.mg / pyunits.liter,
            doc="Initial chlorine demand",
        )

        self.contact_time = Var(
            self.flowsheet().time, units=pyunits.min, doc="Chlorine contact time"
        )

        self.chlorine_decay_rate = Var(
            self.flowsheet().time,
            units=1 / pyunits.hour,
            doc="Chlorine decay rate",
        )

        self._fixed_perf_vars.append(self.initial_chlorine_demand)
        # self._fixed_perf_vars.append(self.contact_time)
        self._fixed_perf_vars.append(self.chlorine_decay_rate)

        self.chlorine_dose = Var(
            self.flowsheet().time, units=pyunits.mg / pyunits.L, doc="Chlorine dose"
        )

        self.chlorine = Var(
            self.flowsheet().time,
            units=pyunits.kg / pyunits.s,
            bounds=(0, None),
            doc="Mass flow rate of chlorine",
        )

        self.electricity = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(0, None),
            units=pyunits.kW,
            doc="Power demand",
        )

        @self.Constraint(self.flowsheet().time, doc="Chlorine dose constraint")
        def chlorine_dose_constraint(b, t):
            if self.config.LRVCl_required_for_virus != 0 and self.config.LRVCl_required_for_giardia == 0:
                CTreq = 0.5 * self.config.LRVCl_required_for_virus * (pyunits.mg * pyunits.min / pyunits.liter)
            elif self.config.LRVCl_required_for_virus == 0 and self.config.LRVCl_required_for_giardia != 0:
                CTreq = 15.257 * self.config.LRVCl_required_for_giardia * (pyunits.mg * pyunits.min / pyunits.liter) + (
                            0.1333 * pyunits.mg * pyunits.min / pyunits.liter)
            elif self.config.LRVCl_required_for_virus != 0 and self.config.LRVCl_required_for_giardia != 0:
                CTreq = max(0.5 * self.config.LRVCl_required_for_virus,
                            15.257 * self.config.LRVCl_required_for_giardia * (
                                        pyunits.mg * pyunits.min / pyunits.liter) + (
                                        0.1333 * pyunits.mg * pyunits.min / pyunits.liter))

            #todo what if initial Cl is enough high to meet CT value? + the final chloride concentration should be higher than 0.2 mg/L (check the value)
            #todo maybe should be considered in the optimization process (main code)

            if self.config.LRVCl_required_for_virus == 0 and self.config.LRVCl_required_for_giardia == 0:
                return b.chlorine_dose[t] == 0 * pyunits.mg / pyunits.L
            else:
                decay_rate_min = pyunits.convert(b.chlorine_decay_rate[t], to_units=1 / pyunits.min)
                exp_term = pyo.exp(pyunits.convert(
                    -b.chlorine_decay_rate[t] * b.contact_time[t], to_units=pyunits.dimensionless
                ))

                expr1 = (decay_rate_min * CTreq) / (1 - exp_term) - b.initial_chlorine_demand[t]
                expr2 = (0.2 * pyunits.mg / pyunits.L) / exp_term - b.initial_chlorine_demand[t]
                max_expr = 0.5 * (expr1 + expr2 + abs(expr1 - expr2))

                return b.chlorine_dose[t] == max_expr

                #
                #
                # return b.chlorine_dose[t] == max((
                #         pyunits.convert(b.chlorine_decay_rate[t], to_units=1 / pyunits.min) * CTreq / (1 - pyo.exp(
                #     pyunits.convert(-b.chlorine_decay_rate[t] * b.contact_time[t], to_units=pyunits.dimensionless))) -
                #         b.initial_chlorine_demand[t]), (0.2 * (pyunits.mg / pyunits.L) / pyo.exp(
                #     pyunits.convert(-b.chlorine_decay_rate[t] * b.contact_time[t], to_units=pyunits.dimensionless)) -
                #                                         b.initial_chlorine_demand[t]))

        @self.Constraint(self.flowsheet().time, doc="Cl power constraint")
        def electricity_constraint(b, t):  # unit=kWh/h=kW
            return b.electricity[t] == (
                    b.energy_electric_flow_vol_inlet *
                    pyunits.convert(
                        b.properties_in[t].flow_vol,
                        to_units=pyunits.m ** 3 / pyunits.hour
                    )
            )

        @self.Constraint(self.flowsheet().time, doc="Chlorine mass flow rate constraint")
        def chlorine_flow_mass_constraint(b, t):
            return b.chlorine[t] == pyunits.convert(
                b.chlorine_dose[t] * b.properties_in[t].flow_vol,
                to_units=pyunits.kg / pyunits.s,
            )

    @property
    def default_costing_method(self):
        return self.cost_chlorination

    @staticmethod
    def cost_chlorination(blk):
        t0 = blk.flowsheet().time.first()

        # 1) LRV가 모두 0이면 자본비용 0으로 고정하고 log 계산 자체를 건너뜀
        if (blk.unit_model.config.LRVCl_required_for_virus == 0
                and blk.unit_model.config.LRVCl_required_for_giardia == 0):
            blk.capital_cost = pyo.Var(
                initialize=0,
                units=blk.config.flowsheet_costing_block.base_currency,
                bounds=(0, None),
                doc="Capital cost of unit operation",
            )

            expr= 0 * blk.config.flowsheet_costing_block.base_currency

            blk.costing_package.add_cost_factor(blk, "TIC")
            blk.capital_cost_constraint = pyo.Constraint(expr=blk.capital_cost == blk.cost_factor * expr)

            blk.config.flowsheet_costing_block.cost_flow(
                blk.unit_model.electricity[t0], "electricity"
            )
            blk.config.flowsheet_costing_block.cost_flow(
                blk.unit_model.chlorine[t0], "chlorine"
            )
            return

        # ---- 아래는 기존 계산 그대로 ----
        parameter_dict = blk.unit_model.config.database.get_unit_operation_parameters(
            blk.unit_model._tech_type, subtype=blk.unit_model.config.process_subtype
        )
        A, B, C = blk.unit_model._get_tech_parameters(
            blk, parameter_dict, blk.unit_model.config.process_subtype,
            ["capital_a_parameter", "capital_b_parameter", "capital_c_parameter"],
        )

        blk.capital_cost = pyo.Var(
            initialize=1,
            units=blk.config.flowsheet_costing_block.base_currency,
            bounds=(0, None),
            doc="Capital cost of unit operation",
        )

        ln_Q = pyo.log(
            pyo.units.convert(
                blk.unit_model.properties_in[t0].flow_vol / (pyo.units.m ** 3 / pyo.units.second),
                to_units=pyo.units.dimensionless,
            )
        )
        ln_D = pyo.log(
            pyo.units.convert(
                blk.unit_model.chlorine_dose[t0] / (pyo.units.mg / pyo.units.liter),
                to_units=pyo.units.dimensionless,
            )
        )

        expr = pyo.units.convert(
            A * ln_Q + B * ln_D + C * ln_Q * ln_D,
            to_units=blk.config.flowsheet_costing_block.base_currency,
        )

        blk.costing_package.add_cost_factor(blk, "TIC")
        blk.capital_cost_constraint = pyo.Constraint(expr=blk.capital_cost == blk.cost_factor * expr)

        blk.config.flowsheet_costing_block.cost_flow(blk.unit_model.electricity[t0], "electricity")
        blk.config.flowsheet_costing_block.cost_flow(blk.unit_model.chlorine[t0], "chlorine")

    # @staticmethod
    # def cost_chlorination(blk):
    #     """
    #     Capital cost is assumed to be negligible.
    #     This method registers the chlorine flow and electricity demand as costed flows.
    #     """
    #     t0 = blk.flowsheet().time.first()
    #
    #     # expr = 0 * pyunits.MUSD_2014
    #     #
    #     # expr = pyo.units.convert(expr, to_units=blk.config.flowsheet_costing_block.base_currency)
    #
    #     # Get parameter dict from database
    #     parameter_dict = blk.unit_model.config.database.get_unit_operation_parameters(
    #         blk.unit_model._tech_type, subtype=blk.unit_model.config.process_subtype
    #     )
    #
    #     # Get costing parameter sub-block for this technology
    #     A, B, C = blk.unit_model._get_tech_parameters(
    #         blk,
    #         parameter_dict,
    #         blk.unit_model.config.process_subtype,
    #         ["capital_a_parameter", "capital_b_parameter", "capital_c_parameter"],
    #     )
    #
    #     # Add cost variable and constraint
    #     blk.capital_cost = pyo.Var(
    #         initialize=1,
    #         units=blk.config.flowsheet_costing_block.base_currency,
    #         bounds=(0, None),
    #         doc="Capital cost of unit operation",
    #     )
    #
    #     ln_Q = pyo.log(
    #         pyo.units.convert(
    #             blk.unit_model.properties_in[t0].flow_vol
    #             / (pyo.units.m**3 / pyo.units.second),
    #             to_units=pyo.units.dimensionless,
    #         )
    #     )
    #     ln_D = pyo.log(
    #         pyo.units.convert(
    #             blk.unit_model.chlorine_dose[t0] / (pyo.units.mg / pyo.units.liter),
    #             to_units=pyo.units.dimensionless,
    #         )
    #     )
    #
    #     expr = pyo.units.convert(
    #         A * ln_Q + B * ln_D + C * ln_Q * ln_D,
    #         to_units=blk.config.flowsheet_costing_block.base_currency,
    #     )
    #
    #     blk.costing_package.add_cost_factor(
    #         blk, "TIC"
    #     )
    #
    #     blk.capital_cost_constraint = pyo.Constraint(
    #         expr=blk.capital_cost == blk.cost_factor * expr
    #     )
    #
    #     # Register flows
    #     blk.config.flowsheet_costing_block.cost_flow(
    #         blk.unit_model.electricity[t0], "electricity"
    #     )
    #
    #     blk.config.flowsheet_costing_block.cost_flow(
    #         blk.unit_model.chlorine[t0], "chlorine"
    #     )
