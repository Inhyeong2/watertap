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
This module contains a zero-order representation of a biologically active
filtration unit operation. The whole flowsheet must contain ozonation_DPR process in prior to this m.
"""

import pyomo.environ as pyo
from pyomo.common.config import ConfigBlock, ConfigValue
from idaes.core.util.exceptions import ConfigurationError
from pyomo.environ import units as pyunits, Var
from pyomo.environ import Reference
from idaes.core import declare_process_block_class
from watertap.core import build_sido, constant_intensity, ZeroOrderBaseData

# Some more information about this module
__author__ = "Inhyeong Jeon"


@declare_process_block_class("BioActiveFiltrationDPRZO")
class BioActiveFiltrationDPRZOData(ZeroOrderBaseData):
    """
    Zero-Order m for a Biologically Active Filtration unit operation.
    """

    CONFIG = ZeroOrderBaseData.CONFIG()

    CONFIG.declare("state", ConfigValue(
        default=None,
        domain=str,
        description="State",
    ))

    def build(self):
        super().build()
        self._tech_type = "bio_active_filtration_DPR"
        build_sido(self)

        if "toc" not in self.config.property_package.config.solute_list:
            raise ConfigurationError(
                "toc must be in solute list"
            )

        self.EBCT = Var(
            self.flowsheet().time, units=pyunits.minute, doc="Empty bed contact time"
        )

        self.energy_electric_flow_vol_inlet = Var(
            units=pyunits.kWh / pyunits.m ** 3,
            doc="Electricity intensity with respect to inlet flowrate of unit",
        )

        self._fixed_perf_vars.append(self.energy_electric_flow_vol_inlet)

        # self.O3toTOC = Var(self.flowsheet().time, units=pyunits.dimensionless, doc="O3:TOC passed from ozone")

        self.electricity = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(0, None),
            units=pyunits.kW,
            doc="BAF power demand",
        )

        self.activated_carbon = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(0, None),
            units=pyunits.kg / pyunits.year,
            doc="Activated carbon demand",
        )

        @self.Constraint(self.flowsheet().time, doc="BAF power constraint")
        def electricity_constraint(b, t):  # unit=kWh/h=kW
            return b.electricity[t] == (
                    b.energy_electric_flow_vol_inlet *
                    pyunits.convert(
                        b.properties_in[t].flow_vol,
                        to_units=pyunits.m ** 3 / pyunits.hour
                    )
            )

        @self.Constraint(self.flowsheet().time, doc="Activated carbon consumption [kg/year]")   # from Plumlee et al.
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

            # Time period for replacement
            replacement_interval = 8 * pyunits.year

            # Final expression in kg/year
            return b.activated_carbon[t] == (
                    ebct_ratio
                    * average_capa
                    * flow_mgd
                    * carbon_volume
                    * carbon_density
                    / replacement_interval
            )

        # # self._perf_var_dict["Electricity Intensity"] = self.energy_electric_flow_vol_inlet
        # constant_intensity(self)

    @property
    def default_costing_method(self):
        return self.cost_BAF

    @staticmethod
    def cost_BAF(blk):
        """
        General method for costing BAF addition. Capital cost is
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
