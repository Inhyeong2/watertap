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
This module contains a zero-order representation of a UV-AOP unit
operation.
"""

import pyomo.environ as pyo
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.environ import units as pyunits, Var
from idaes.core import declare_process_block_class
from watertap.core import build_siso, ZeroOrderBaseData


# Some more information about this module
__author__ = "Inhyeong Jeon"


@declare_process_block_class("UVAOPDPRZO")
class UVAOPDPRZOData(ZeroOrderBaseData):
    """
    Zero-Order m for a UV-AOP unit operation.
    """

    CONFIG = ZeroOrderBaseData.CONFIG()

    CONFIG.declare("treatment_train", ConfigValue(
        default=None,
        domain=str,
        description="treatment_train",
    ))

    def build(self):

        super().build()
        self._tech_type = "uv_aop_DPR"
        build_siso(self)

        self.uv_dose = Var(
            self.flowsheet().time,
            units=pyunits.mJ / pyunits.cm ** 2,
            doc="UV dosage",
        )

        self.hydrogen_peroxide_dose = Var(
            self.flowsheet().time, units=pyunits.mg / pyunits.L, bounds=(0, None),
            doc="Oxidant(H2O2) dosage (mg/L)"
        )

        self.hydrogen_peroxide = Var(
            self.flowsheet().time,
            units=pyunits.kg / pyunits.s,
            bounds=(0, None),
            doc="Mass flow rate of H2O2",
        )

        self.hypochlorite = Var(                    # Plumlee et al. for CBAT
            self.flowsheet().time,
            units=pyunits.kg / pyunits.s,
            bounds=(0, None),
            doc="Mass flow rate of hypochlorite",
        )

        self.lamp_replacement = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(0, None),
            units=pyunits.Mgallons / pyunits.day,
            doc="Lamp (replacement) demand",
        )

        self.energy_electric_flow_vol_inlet = Var(
            units=pyunits.kWh / pyunits.m ** 3,
            doc="Electricity intensity with respect to inlet flowrate of unit",
        )

        self._fixed_perf_vars.append(self.energy_electric_flow_vol_inlet)

        self.electricity = Var(
            self.flowsheet().time,
            units=pyunits.kW,
            bounds=(0, None),
            doc="Electricity consumption of unit",
        )

        if self.config.treatment_train == "RBAT":
            self.chloramine_dose = Var(
                self.flowsheet().time,
                units=pyunits.mg / pyunits.L,
                doc="Chloramine dose (mg/L as Cl2)",
            )

        @self.Constraint(self.flowsheet().time, doc="UV-AOP power constraint")
        def electricity_constraint(b, t):  # unit=kWh/h=kW
            return b.electricity[t] == (
                    b.energy_electric_flow_vol_inlet *
                    pyunits.convert(
                        b.properties_in[t].flow_vol,
                        to_units=pyunits.m ** 3 / pyunits.hour
                    )
            )

        @self.Constraint(self.flowsheet().time, doc="Required UV dose for 0.5 log-reduction of 1,4-D")
        def required_UV_dose_constraint(b, t):
            h2o2 = b.hydrogen_peroxide_dose[t] / (1 * pyunits.mg / pyunits.L)
            if b.config.treatment_train == "CBAT":
                return b.uv_dose[t] == (
                        -31.185 * h2o2 ** 3 +
                        778.54 * h2o2 ** 2 -
                        6560.7 * h2o2 +
                        19581
                ) * (pyunits.mJ / pyunits.cm ** 2)
            elif b.config.treatment_train == "RBAT":
                nh2cl = b.chloramine_dose[t] / (1 * pyunits.mg / pyunits.L)
                return b.uv_dose[t] == (
                        2336.25 * nh2cl / h2o2 + 125
                ) * (pyunits.mJ / pyunits.cm ** 2)


        # @self.Constraint(self.flowsheet().time, doc="Required UV dose for 0.5 log-reduction of 1,4-D")
        # def required_UV_dose_constraint(b, t):
        #     return b.uv_dose[t] == (38041 * (b.hydrogen_peroxide_dose[t] /(1 * pyunits.mg / pyunits.L)) ** -1.762) * (pyunits.mJ / pyunits.cm ** 2)

        @self.Constraint(self.flowsheet().time, doc="Hydrogen peroxide mass flow rate constraint")
        def hydrogen_peroxide_flow_mass_constraint(b, t):
            return b.hydrogen_peroxide[t] == pyunits.convert(
                b.hydrogen_peroxide_dose[t] * b.properties_in[t].flow_vol,
                to_units=pyunits.kg / pyunits.s,
            )

        @self.Constraint(self.flowsheet().time, doc="Hypochlorite mass flow rate constraint")
        def hypochlorite_flow_mass_constraint(b, t):
            return b.hypochlorite[t] == 1.65 * b.hydrogen_peroxide[t] # Plumlee et al.

        @self.Constraint(self.flowsheet().time, doc="Lamp replacement constraint")
        def lamp_replacement_constraint(b, t):
            return b.lamp_replacement[t] == pyunits.convert(b.properties_in[t].flow_vol, to_units=pyunits.Mgallons / pyunits.day)


    @property
    def default_costing_method(self):
        return self.cost_uv_aop

    @staticmethod
    def cost_uv_aop(blk):
        t0 = blk.flowsheet().time.first()
        """
                General method for costing UV-AOP addition. Capital cost is
                based on the inlet flowrate.
                """

        # Add cost variable
        blk.capital_cost = pyo.Var(
            initialize=1,
            units=blk.config.flowsheet_costing_block.base_currency,
            bounds=(0, None),
            doc="Capital cost of unit operation",
        )


        # Expression from Plumlee et al.
        # Convert flow to MGD
        # flow_mgd = pyo.units.convert(
        #     blk.unit_model.properties_in[t0].flow_vol,
        #     to_units=pyunits.Mgallons / pyunits.day,
        # )

        # expr = (
        #         0.1133 * (pyunits.MUSD_2014 / (pyunits.Mgallons / pyunits.day)) * flow_mgd
        #         + 0.0068 * pyunits.MUSD_2014
        # )

        # expr = pyo.units.convert(expr, to_units=blk.config.flowsheet_costing_block.base_currency)

        # Get parameter dict from database
        parameter_dict = blk.unit_model.config.database.get_unit_operation_parameters(
            blk.unit_model._tech_type, subtype=blk.unit_model.config.process_subtype
        )

        # Get costing parameter sub-block for this technology
        A, B, C, D = blk.unit_model._get_tech_parameters(
            blk,
            parameter_dict,
            blk.unit_model.config.process_subtype,
            [
                "reactor_cost",
                "lamp_cost",
                "aop_capital_a_parameter",
                "aop_capital_b_parameter",
            ],
        )

        expr = blk.unit_model._get_uv_capital_cost(blk, A, B)
        expr += blk.unit_model._get_aop_capital_cost(blk, C, D)

        # Determine if a costing factor is required
        blk.costing_package.add_cost_factor(
            blk, "TIC"
        )

        blk.capital_cost_constraint = pyo.Constraint(
            expr=blk.capital_cost == blk.cost_factor * expr
        )

        # Register flows
        blk.config.flowsheet_costing_block.cost_flow(
            blk.unit_model.electricity[t0], "electricity"
        )

        blk.config.flowsheet_costing_block.cost_flow(
            blk.unit_model.hydrogen_peroxide[t0], "hydrogen_peroxide"
        )

        blk.config.flowsheet_costing_block.cost_flow(
            blk.unit_model.hypochlorite[t0], "hypochlorite"
        )

        blk.config.flowsheet_costing_block.cost_flow(
            blk.unit_model.lamp_replacement[t0], "lamp_replacement"
        )

    @staticmethod
    def _get_uv_capital_cost(blk, A, B):
        """
        Generate expression for capital cost of UV reactor.
        """
        t0 = blk.flowsheet().time.first()

        Q = pyo.units.convert(
            blk.unit_model.properties_in[t0].flow_vol,
            to_units=pyo.units.m ** 3 / pyo.units.hr,
        )

        E = pyo.units.convert(blk.unit_model.electricity[t0], to_units=pyo.units.kW)

        expr = pyo.units.convert(
            A * Q + B * E,
            to_units=blk.config.flowsheet_costing_block.base_currency,
        )

        return expr

    @staticmethod
    def _get_aop_capital_cost(blk, A, B):
        """
        Generate expression for capital cost due to AOP addition.
        """
        t0 = blk.flowsheet().time.first()

        hydrogen_peroxide = pyo.units.convert(
            blk.unit_model.hydrogen_peroxide[t0], to_units=pyo.units.lb / pyo.units.day
        )

        expr = pyo.units.convert(
            A
            * pyo.units.convert(
                hydrogen_peroxide / (pyo.units.lb / pyo.units.day),
                to_units=pyo.units.dimensionless,
            )
            ** B,
            to_units=blk.config.flowsheet_costing_block.base_currency,
        )

        return expr