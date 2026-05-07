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
This module contains a zero-order representation of an ultra filtration unit
operation.
"""
import pyomo.environ as pyo
from pyomo.environ import units as pyunits, Var
from idaes.core import declare_process_block_class
from watertap.core import build_sido, ZeroOrderBaseData

# Some more information about this module
__author__ = "Inhyeong Jeon"


@declare_process_block_class("UltraFiltrationDPRZO")
class UltraFiltrationDPRZOData(ZeroOrderBaseData):
    """
    Zero-Order m for an Ultra Filtration unit operation.
    """

    CONFIG = ZeroOrderBaseData.CONFIG()

    def build(self):
        super().build()
        self._tech_type = "ultra_filtration_DPR"
        build_sido(self)

        self.energy_electric_flow_vol_inlet = Var(
            units=pyunits.kWh / pyunits.m ** 3,
            doc="Electricity intensity with respect to inlet flowrate of unit",
        )

        self._fixed_perf_vars.append(self.energy_electric_flow_vol_inlet)

        self.energy_membrane_replacement_flow_vol_inlet = Var(
            units=pyunits.kWh / pyunits.m ** 3,
            doc="Electricity intensity (for membrane replacement) with respect to inlet flowrate of unit",
        )

        self._fixed_perf_vars.append(self.energy_membrane_replacement_flow_vol_inlet)

        self.electricity = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(0, None),
            units=pyunits.kW,
            doc="UF power demand",
        )

        self.membrane_replacement = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(0, None),
            units=pyunits.kW,
            doc="Membrane (replacement) demand",
        )

        @self.Constraint(self.flowsheet().time, doc="UF power constraint")
        def electricity_constraint(b, t):  # unit=kWh/h=kW
            return b.electricity[t] == (
                    b.energy_electric_flow_vol_inlet *
                    pyunits.convert(
                        b.properties_in[t].flow_vol,
                        to_units=pyunits.m ** 3 / pyunits.hour
                    )
            )

        @self.Constraint(self.flowsheet().time, doc="Membrane replacement cost [kW]") # from Plumlee et al.
        def membrane_replacement_constraint(b, t):
            return b.membrane_replacement[t] == (
                    b.energy_membrane_replacement_flow_vol_inlet *
                    pyunits.convert(
                        b.properties_in[t].flow_vol,
                        to_units=pyunits.m ** 3 / pyunits.hour
                    )
            )

    @property
    def default_costing_method(self):
        return self.cost_UF

    @staticmethod
    def cost_UF(blk):
        """
        General method for costing UF addition. Capital cost is
        based on the inlet flowrate.
        """
        t0 = blk.flowsheet().time.first()

        # Convert flow to MGD
        flow_mgd = pyo.units.convert(
            blk.unit_model.properties_in[t0].flow_vol,
            to_units=pyunits.Mgallons / pyunits.day,
        )

        expr = (
                -0.0043 * (pyunits.MUSD_2014 / ((pyunits.Mgallons / pyunits.day) ** 2)) * (flow_mgd ** 2)
                + 0.5356 * (pyunits.MUSD_2014 / (pyunits.Mgallons / pyunits.day)) * flow_mgd
                + 0.734 * pyunits.MUSD_2014
        )

        expr = pyo.units.convert(expr, to_units=blk.config.flowsheet_costing_block.base_currency)

        # Get parameter dict from database
        parameter_dict = blk.unit_model.config.database.get_unit_operation_parameters(
            blk.unit_model._tech_type, subtype=blk.unit_model.config.process_subtype
        )

        # Add cost variable
        blk.capital_cost = pyo.Var(
            initialize=1,
            units=blk.config.flowsheet_costing_block.base_currency,
            bounds=(0, None),
            doc="Capital cost of unit operation",
        )

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
            blk.unit_model.membrane_replacement[t0], "membrane_replacement"
        )