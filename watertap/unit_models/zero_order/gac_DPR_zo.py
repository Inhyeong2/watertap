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
This module contains a zero-order representation of a granular activated carbon unit
operation. The whole flowsheet must contain ozonation_DPR and BAF_DPR process in prior to this m.
"""

import pyomo.environ as pyo
from pyomo.environ import units as pyunits, Var
from idaes.core import declare_process_block_class
from idaes.core.util.math import smooth_min, smooth_max
from watertap.core import build_sido, ZeroOrderBaseData

# Some more information about this module
__author__ = "Inhyeong Jeon"


@declare_process_block_class("GACDPRZO")
class GACDPRZOData(ZeroOrderBaseData):
    """
    Zero-Order m for a granular activated carbon unit operation.
    """

    CONFIG = ZeroOrderBaseData.CONFIG()

    def build(self):
        super().build()
        self._tech_type = "gac_DPR"
        build_sido(self)

        # Empty Bed Contact Time
        self.EBCT = Var(
            self.flowsheet().time,
            units=pyunits.minute,
            bounds=(0, None),
            doc="Empty bed contact time of unit"
        )

        self.energy_electric_flow_vol_inlet = Var(
            units=pyunits.kWh / pyunits.m**3,
            doc="Electricity intensity with respect to inlet flowrate of unit",
        )

        self._fixed_perf_vars.append(self.energy_electric_flow_vol_inlet)

        self.required_BV = Var(
            self.flowsheet().time,
            bounds=(0, None),
            units=pyunits.dimensionless,
            doc="Total BV treated before breakthrough point",
        )


        self.replacement_frequency = pyo.Expression(
            self.flowsheet().time,
            rule=lambda b, t: pyunits.convert(((b.required_BV[t] * b.EBCT[t] * pyunits.day) / (
                1440 * pyunits.minute)), to_units=pyunits.year),
            doc="replacement_frequency",
        )

        self.electricity = Var(
            self.flowsheet().time,
            units=pyunits.kW,
            bounds=(0, None),
            doc="Electricity consumption of unit",
        )

        self.activated_carbon = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(0, None),
            units=pyunits.kg / pyunits.year,
            doc="Activated carbon demand",
        )


        @self.Constraint(self.flowsheet().time, doc="GAC power constraint")
        def electricity_constraint(b, t):  # unit=kWh/h=kW
            return b.electricity[t] == (
                    b.energy_electric_flow_vol_inlet *
                    pyunits.convert(
                        b.properties_in[t].flow_vol,
                        to_units=pyunits.m ** 3 / pyunits.hour
                    )
            )

        @self.Constraint(self.flowsheet().time, doc="Activated carbon consumption [kg/year]")  # from Plumlee et al.
        def activated_carbon_constraint(b, t):
            flow_mgd = pyo.units.convert(
                b.properties_in[t].flow_vol,
                to_units=pyunits.Mgallons / pyunits.day
            )

            # EBCT normalization
            ebct_ratio = b.EBCT[t] / (10 * pyunits.minute)

            # Calculated assuming 100% of the design flow capacity
            average_capa = 1

            # Volume of carbon per MGD
            carbon_volume = pyunits.convert(
                928 * pyunits.ft ** 3 / (pyunits.Mgallons / pyunits.day),
                to_units=pyunits.m ** 3 / (pyunits.Mgallons / pyunits.day)
            )
            # Carbon density
            carbon_density = pyunits.convert(
                0.45 * pyunits.g / pyunits.cm ** 3,
                to_units=pyunits.kg / pyunits.m ** 3
            )


            # Final expression in kg/year
            return b.activated_carbon[t] == (
                    ebct_ratio
                    * average_capa
                    * flow_mgd
                    * carbon_volume
                    * carbon_density
                    / b.replacement_frequency[t]
            )


    @property
    def default_costing_method(self):
        return self.cost_gac

    @staticmethod
    def cost_gac(blk):
        """
                General method for costing GAC addition. Capital cost is
                based on the inlet flowrate and EBCT.
                """
        t0 = blk.flowsheet().time.first()

        # Convert flow to MGD
        flow_mgd = pyo.units.convert(
            blk.unit_model.properties_in[t0].flow_vol,
            to_units=pyunits.Mgallons / pyunits.day,
        )

        expr = (
                (
                        0.0043 * (
                        pyunits.MUSD_2014 / ((pyunits.Mgallons / pyunits.day) * pyunits.minute)) * flow_mgd *
                        blk.unit_model.EBCT[t0]
                        + 0.2087 * (pyunits.MUSD_2014 / (pyunits.Mgallons / pyunits.day)) * flow_mgd
                        + 1.1643 * pyunits.MUSD_2014
                )
                / 1.3 # the above equation is equipment cost + 30% installation, therefore, here we divide it by 1.3 for the consistency
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
            blk.unit_model.activated_carbon[t0], "activated_carbon"
        )

