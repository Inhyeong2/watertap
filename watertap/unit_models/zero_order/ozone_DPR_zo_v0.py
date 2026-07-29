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
This module contains a zero-order representation of a Ozone reactor unit.
"""

import pyomo.environ as pyo
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.environ import units as pyunits, Var
from pyomo.environ import Reference
from idaes.core.util.exceptions import ConfigurationError
from idaes.core import declare_process_block_class
from watertap.core import build_siso, ZeroOrderBaseData

# Some more information about this module
__author__ = "Inhyeong Jeon"


@declare_process_block_class("OzoneDPRZOv0")
class OzoneDPRZOv0Data(ZeroOrderBaseData):
    """
    Zero-Order m for an Ozone unit operation.
    """

    CONFIG = ZeroOrderBaseData.CONFIG()

    CONFIG.declare("state", ConfigValue(
        default=None,
        domain=str,
        description="State",
    ))

    CONFIG.declare("effluent_type", ConfigValue(
        default="TERTIARY",
        domain=str,
        description="Type of wastewater effluent",
    ))

    CONFIG.declare("LRVO3_required", ConfigValue(
        default=None,
        domain=None,
        description="Required LRV"
    ))

    def build(self):
        super().build()
        self._tech_type = "ozonation_DPR"
        build_siso(self)

        if "toc" not in self.config.property_package.config.solute_list:
            raise ConfigurationError(
                "toc must be in solute list"
            )

        self.contact_time = Var(
            self.flowsheet().time,
            units=pyunits.minute,
            bounds=(0, None),
            doc="Ozone contact time"
        )

        self.mass_transfer_efficiency = Var(
            self.flowsheet().time,
            units=pyunits.dimensionless,
            doc="Ozone mass transfer efficiency",
        )

        self.specific_energy_coeff = Var(
            self.flowsheet().time,
            units=pyunits.kWh / pyunits.lb,
            bounds=(0, None),
            doc="Specific energy consumption for ozone generation",
        )

        self.O3toTOC = Var(
            self.flowsheet().time,
            units=pyunits.dimensionless,
            bounds=(0, None),
            doc="O3:TOC",
        )

        # self._fixed_perf_vars.append(self.contact_time)
        self._fixed_perf_vars.append(self.mass_transfer_efficiency)
        self._fixed_perf_vars.append(self.specific_energy_coeff)
        # self._fixed_perf_vars.append(self.O3toTOC)

        self.ozone_flow_mass = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(0, None),
            units=pyunits.lb / pyunits.hr,
            doc="Mass flow rate of ozone",
        )

        self.ozone_consumption = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(0, None),
            units=pyunits.mg / pyunits.liter,
            doc="Ozone consumption",
        )

        self.electricity = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(0, None),
            units=pyunits.kW,
            doc="Ozone generation power demand",
        )

        if self.config.effluent_type.upper() == "SECONDARY":
            self.kO3 = pyo.Expression(
                self.flowsheet().time,
                rule=lambda b, t: (-0.2637 * b.O3toTOC[t] + 0.5251) * (1 / pyunits.min),
                doc="Reaction rate constant for secondary effluent",
            )

        elif self.config.effluent_type.upper() == "TERTIARY":
            self.kO3 = pyo.Expression(
                self.flowsheet().time,
                rule=lambda b, t: (-0.6422 * b.O3toTOC[t] + 0.8342) * (1 / pyunits.min),
                doc="Reaction rate constant for tertiary effluent",
            )

        # @self.Constraint(self.flowsheet().time, doc="kO3 constraint")
        # def kO3_constraint(b, t):
        #     if b.config.effluent_type.upper() == "SECONDARY":
        #         return b.kO3[t] == 1.824 * pyo.exp(-1.863 * b.O3toTOC[t]) * (1 / pyunits.min)
        #
        #     elif b.config.effluent_type.upper() == "TERTIARY":
        #         return b.kO3[t] == 0.158 * (b.O3toTOC[t] ** -1.672) * (1 / pyunits.min)
        #
        #     else:
        #         raise ConfigurationError(f"Unsupported effluent: {b.config.effluent_type.upper()}")


        self.nitrite = pyo.Expression(
            self.flowsheet().time,
            rule=lambda b, t: b.properties_in[t].conc_mass_comp["nitrite"] /(46.0055/14.0067), # Nitrite --> Nitrogen
            doc="Nitrite concentration at each time step",
        )
        # self.nitrite = Reference(self.properties_in[self.flowsheet().time.first()].conc_mass_comp["nitrite"])

        @self.Constraint(self.flowsheet().time, doc="O3toTOC ratio constraint")
        def O3toTOC_ratio_constraint(b, t):
            state = b.config.state
            LRV = b.config.LRVO3_required
            CTreq = 2.47 * LRV * pyunits.mg * pyunits.min / pyunits.L
            if state == "CA":
                return b.O3toTOC[t] == 1
            elif state == "CO":
                # exp_term = pyo.exp(pyunits.convert(
                #     -b.kO3[t] * (b.contact_time[t] - 0.5 * pyunits.min), to_units=pyunits.dimensionless
                # ))
                # expr2 = (0.2 * (pyunits.mg / pyunits.L) / exp_term - 1.136 * pyunits.mg / pyunits.L) / (
                #         0.704 * pyunits.mg / pyunits.L) # O3:TOC ratio required to meet residual O3 (0.2 mg/L) requirement
                expr2 = (
                            - 0.0091 * (b.contact_time[t] / pyunits.min) ** 2
                            + 0.2052 * (b.contact_time[t] / pyunits.min)
                            - 0.2039
                    )
                if LRV == 1: # O3:TOC 0.5 & HRT 5 is already enough to achieve more than 1 LRV.
                    expr1 = 0.5
                elif LRV == 2:
                    expr1 = (
                            -0.0009 * (b.contact_time[t] / pyunits.min) ** 3
                            + 0.0247 * (b.contact_time[t] / pyunits.min) ** 2
                            - 0.2375 * (b.contact_time[t] / pyunits.min)
                            + 1.4855
                    )
                elif LRV == 3:
                    expr1 = (
                            + 0.0045 * (b.contact_time[t] / pyunits.min) ** 2
                            - 0.0977 * (b.contact_time[t] / pyunits.min)
                            + 1.4382
                    )
                else:
                    raise ConfigurationError(
                        f"LRVO3_required value cannot be greater than 3."
                    )

                max_expr = 0.5 * (expr1 + expr2 + abs(expr1 - expr2))
                real_max = 0.5 * (max_expr + 0.5 + abs(max_expr - 0.5))
                return b.O3toTOC[t] == real_max

            else:
                if LRV == 1:  # O3:TOC 0.5 & HRT 5 is already enough to achieve more than 1 LRV.
                    expr1 = 0.5
                elif LRV == 2:
                    expr1 = (
                            -0.0009 * (b.contact_time[t] / pyunits.min) ** 3
                            + 0.0247 * (b.contact_time[t] / pyunits.min) ** 2
                            - 0.2375 * (b.contact_time[t] / pyunits.min)
                            + 1.4855
                    )
                elif LRV == 3:
                    expr1 = (
                            + 0.0045 * (b.contact_time[t] / pyunits.min) ** 2
                            - 0.0977 * (b.contact_time[t] / pyunits.min)
                            + 1.4382
                    )
                else:
                    raise ConfigurationError(
                        f"LRVO3_required value cannot be greater than 3."
                    )

                real_max = 0.5 * (expr1 + 0.5 + abs(expr1 - 0.5))
                return b.O3toTOC[t] == real_max

        #todo: it should be this in theory.

        # @self.Constraint(self.flowsheet().time, doc="O3toTOC ratio constraint")
        # def O3toTOC_ratio_constraint(b, t):
        #     state = b.config.state
        #     LRV = b.config.LRVO3_required
        #     CTreq = 2.47 * LRV * pyunits.mg * pyunits.min / pyunits.L
        #     if state == "CA":
        #         return b.O3toTOC[t] == 1
        #     elif state == "CO":
        #         exp_term = pyo.exp(pyunits.convert(
        #             -b.kO3[t] * (b.contact_time[t] - 0.5 * pyunits.min), to_units=pyunits.dimensionless
        #         ))
        #
        #         expr1 = (CTreq * b.kO3[t] / (pyunits.convert((0.5 * pyunits.min * b.kO3[t] + (1 - exp_term)),
        #                                                      to_units=pyunits.dimensionless)) - 1.136 * pyunits.mg / pyunits.L) / (
        #                         0.704 * pyunits.mg / pyunits.L)
        #
        #         expr2 = (0.2 * (pyunits.mg / pyunits.L) / exp_term - 1.136 * pyunits.mg / pyunits.L) / (
        #                 0.704 * pyunits.mg / pyunits.L)
        #
        #         max_expr = 0.5 * (expr1 + expr2 + abs(expr1 - expr2))
        #         real_max = 0.5 * (max_expr + 0.5 + abs(max_expr - 0.5))
        #         return b.O3toTOC[t] == real_max
        #     else:
        #         exp_term = pyo.exp(pyunits.convert(
        #             -b.kO3[t] * (b.contact_time[t] - 0.5 * pyunits.min), to_units=pyunits.dimensionless
        #         ))
        #         expr1 = (CTreq * b.kO3[t] / (pyunits.convert((0.5 * pyunits.min * b.kO3[t] + (1 - exp_term)),
        #                                                      to_units=pyunits.dimensionless)) - 1.136 * pyunits.mg / pyunits.L) / (
        #                         0.704 * pyunits.mg / pyunits.L)
        #         expr2 = 0.5
        #
        #         real_max = 0.5 * (expr1 + expr2 + abs(expr1 - expr2))
        #         return b.O3toTOC[t] == real_max

        @self.Constraint(self.flowsheet().time, doc="Ozone consumption constraint")
        def ozone_consumption_constraint(b, t):
            return b.ozone_consumption[t] == (
                    (
                            pyunits.convert(
                                b.properties_in[t].conc_mass_comp["toc"],
                                to_units=pyunits.mg / pyunits.liter,
                            ) * b.O3toTOC[t]
                    ) / b.mass_transfer_efficiency[t]
                    + pyunits.convert(
                b.nitrite[t],
                to_units=pyunits.mg / pyunits.liter
            )
            )


        @self.Constraint(self.flowsheet().time, doc="Ozone mass flow constraint")
        def ozone_flow_mass_constraint(b, t):
            return b.ozone_flow_mass[t] == pyunits.convert(
                b.properties_in[t].flow_vol * b.ozone_consumption[t],
                to_units=pyunits.lb / pyunits.hr,
            )

        @self.Constraint(self.flowsheet().time, doc="Ozone power constraint")
        def electricity_constraint(b, t):  # unit=kWh/h=kW
            return b.electricity[t] == (
                b.specific_energy_coeff[t] * b.ozone_flow_mass[t]
            )

    @property
    def default_costing_method(self):
        return self.cost_ozonation

    @staticmethod
    def cost_ozonation(blk):
        """
        General method for costing ozone addition. Capital cost is
        based on the inlet flowrate and ozone dose.
        """
        t0 = blk.flowsheet().time.first()

        # Convert flow to MGD
        flow_mgd = pyo.units.convert(
            blk.unit_model.properties_in[t0].flow_vol,
            to_units=pyunits.Mgallons / pyunits.day,
        )

        contact_time = blk.unit_model.contact_time[t0]
        ozone_conc = blk.unit_model.ozone_consumption[t0]

        expr = (
                pyo.units.convert(blk.unit_model.contact_time[t0] / (5 * pyunits.min), to_units=pyunits.dimensionless)
                * (
                        0.0237 * (pyunits.MUSD_2014 / (pyunits.Mgallons / pyunits.day)) * flow_mgd
                        + 2.1926 * pyunits.MUSD_2014
                        + 0.0156 * (pyunits.MUSD_2014 / (pyunits.Mgallons / pyunits.day)) * flow_mgd * (
                                pyo.units.convert(
                                    blk.unit_model.ozone_consumption[t0] / (3 * pyunits.mg / pyunits.liter),
                                    to_units=pyunits.dimensionless
                                ) - 1
                        )
                )
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
