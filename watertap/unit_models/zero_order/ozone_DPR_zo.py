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
Zero-order representation of an Ozone reactor unit for DPR (theoretical CT variant).

This is the theoretical-CT replacement for ``OzoneDPRZOv0`` (ozone_DPR_zo_v0.py). Instead of the
per-LRV polynomial surrogates (fitted to a tertiary decay constant only), the
required O3:TOC ratio is determined here directly from the underlying, physically
derived CT relationship:

    CTreq   = 2.47 * LRV                                   (Cryptosporidium)
    k       = kO3(O3:TOC, effluent_type)                   (auto-handled)
    exp_t   = exp(-k * (HRT - 0.5 min))
    C_30s   = CTreq * k / (0.5*k + (1 - exp_t))            residual after 30 s
    expr1   = (C_30s - 1.136) / 0.704                      O3:TOC to meet CT
    O3:TOC  = max(expr1, floor)

Because ``kO3`` is a function of the O3:TOC ratio and of the effluent type, both
secondary and tertiary effluent are handled by the same equation (no per-effluent
polynomial refit). Both the contact time (HRT) and O3:TOC are left as free
variables linked by this single equality, so the flowsheet optimizer trades HRT
against O3:TOC along the CT curve. Regulatory floors: California pins O3:TOC at 1.0
(most stringent) and lets the HRT carry the CT; all other states use a 0.5 floor.

A Colorado 0.2 mg/L ozone-residual term (expr2) is present but commented out in
O3toTOC_ratio_constraint: that residual requirement does not apply to ozone under
the CO regulation. Re-enable it there to restore the residual-based floor for CO.
"""

import pyomo.environ as pyo
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.environ import units as pyunits, Var
from pyomo.environ import Reference
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.math import smooth_max
from idaes.core import declare_process_block_class
from watertap.core import build_siso, ZeroOrderBaseData

# Some more information about this module
__author__ = "Inhyeong Jeon"


@declare_process_block_class("OzoneDPRZO")
class OzoneDPRZOData(ZeroOrderBaseData):
    """
    Zero-Order model for an Ozone unit operation (theoretical CT variant).
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
        # Reuse the same techno-economic database entry as OzoneDPRZO so no new
        # yaml is required (mass_transfer_efficiency, specific_energy_coeff, ...).
        self._tech_type = "ozonation_DPR"
        build_siso(self)

        if "toc" not in self.config.property_package.config.solute_list:
            raise ConfigurationError(
                "toc must be in solute list"
            )

        self.contact_time = Var(
            self.flowsheet().time,
            initialize=5,
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

        # contact_time and O3toTOC are NOT fixed from the database; they are free
        # variables linked by O3toTOC_ratio_constraint (HRT <-> O3:TOC trade-off).
        self._fixed_perf_vars.append(self.mass_transfer_efficiency)
        self._fixed_perf_vars.append(self.specific_energy_coeff)

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

        # First-order ozone decay constant as a (linear) function of the O3:TOC
        # ratio; distinct calibration for secondary vs tertiary effluent. This is
        # now actually used by the O3:TOC constraint below.
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
        else:
            raise ConfigurationError(
                f"Unsupported effluent_type: {self.config.effluent_type}"
            )

        self.nitrite = pyo.Expression(
            self.flowsheet().time,
            rule=lambda b, t: b.properties_in[t].conc_mass_comp["nitrite"] / (46.0055 / 14.0067),  # Nitrite --> Nitrogen
            doc="Nitrite concentration at each time step",
        )

        # Achieved Cryptosporidium CT: 30-s residual plateau (C_30s from the Pang
        # relationship, C_30s = 0.704*O3:TOC + 1.136) followed by first-order decay
        # over the remaining HRT. Intrinsic unit quantity; the required-LRV target
        # (CT >= 2.47*LRV) is applied by the flowsheet, except for CA where O3:TOC is
        # pinned at 1.0 and the HRT is set here (see ca_hrt_ct_constraint).
        self.CT_achieved = pyo.Expression(
            self.flowsheet().time,
            rule=lambda b, t: (0.704 * b.O3toTOC[t] + 1.136) * (pyunits.mg / pyunits.L) * (
                0.5 * pyunits.min + pyunits.convert(
                    (1 - pyo.exp(pyunits.convert(
                        -b.kO3[t] * (b.contact_time[t] - 0.5 * pyunits.min),
                        to_units=pyunits.dimensionless))) / b.kO3[t],
                    to_units=pyunits.min)),
            doc="Achieved Cryptosporidium CT (mg.min/L)",
        )

        @self.Constraint(self.flowsheet().time, doc="O3:TOC determination (theoretical CT-based)")
        def O3toTOC_ratio_constraint(b, t):
            state = b.config.state
            LRV = b.config.LRVO3_required
            # No explicit LRV cap: a required LRV that cannot be met within the
            # O3:TOC and HRT bounds simply makes the problem infeasible (the O3:TOC
            # equality would need a value above its upper bound, or the CA CT
            # inequality cannot be satisfied), which surfaces as a solve failure.
            CTreq = 2.47 * LRV * pyunits.mg * pyunits.min / pyunits.L
            k = b.kO3[t]
            exp_term = pyo.exp(pyunits.convert(
                -k * (b.contact_time[t] - 0.5 * pyunits.min), to_units=pyunits.dimensionless
            ))
            denom = pyunits.convert(
                0.5 * pyunits.min * k + (1 - exp_term), to_units=pyunits.dimensionless
            )
            # O3:TOC required to meet the Cryptosporidium CT target.
            expr1 = (CTreq * k / denom - 1.136 * pyunits.mg / pyunits.L) / (
                0.704 * pyunits.mg / pyunits.L
            )

            if state == "CA":
                # California mandates O3:TOC = 1.0 exactly (most stringent design).
                # With the dose pinned, the Cryptosporidium CT is carried by the HRT,
                # which is determined in-model by ca_hrt_ct_constraint (see below).
                return b.O3toTOC[t] == 1.0
            elif state == "CO":
                # NOTE (2026-07): the 0.2 mg/L ozone-residual requirement does NOT apply
                # to ozone under the Colorado regulation, so the residual term (expr2) is
                # disabled and CO is treated like the other non-CA states (CT + 0.5 floor).
                # Re-enable the two commented lines below to restore the residual term.
                # expr2 = (0.2 * (pyunits.mg / pyunits.L) / exp_term - 1.136 * pyunits.mg / pyunits.L) / (
                #     0.704 * pyunits.mg / pyunits.L
                # )
                # return b.O3toTOC[t] == smooth_max(smooth_max(expr1, expr2), 0.5)
                return b.O3toTOC[t] == smooth_max(expr1, 0.5)
            else:
                return b.O3toTOC[t] == smooth_max(expr1, 0.5)

        if self.config.state == "CA":
            # CA pins O3:TOC = 1.0, so the HRT (not the dose) must carry the CT.
            # This equality sets contact_time to the minimum meeting CTreq = 2.47*LRV,
            # floored at the 5-min HRT design minimum: if 5 min already exceeds CTreq,
            # CT_achieved is held at its 5-min value (=> contact_time = 5); otherwise
            # CT_achieved is driven to CTreq (=> contact_time = required HRT). This
            # fully determines contact_time for CA without any flowsheet-side CT
            # constraint or optimization objective.
            @self.Constraint(self.flowsheet().time,
                             doc="CA: HRT set so achieved CT meets 2.47*LRV (HRT floored at 5 min)")
            def ca_hrt_ct_constraint(b, t):
                LRV = b.config.LRVO3_required
                CTU = pyunits.mg * pyunits.min / pyunits.L
                CTreq = 2.47 * LRV
                exp_ref = pyo.exp(pyunits.convert(
                    -b.kO3[t] * (5 * pyunits.min - 0.5 * pyunits.min),
                    to_units=pyunits.dimensionless))
                CT_ref = (0.704 * b.O3toTOC[t] + 1.136) * (pyunits.mg / pyunits.L) * (
                    0.5 * pyunits.min + pyunits.convert(
                        (1 - exp_ref) / b.kO3[t], to_units=pyunits.min))
                return b.CT_achieved[t] / CTU == smooth_max(CTreq, CT_ref / CTU)

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
