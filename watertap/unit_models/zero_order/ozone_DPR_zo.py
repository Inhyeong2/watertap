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


# --------------------------------------------------------------------------- #
# Correlation coefficients -- single source of truth                            #
# --------------------------------------------------------------------------- #
# First-order ozone decay constant k [1/min] as a linear function of the O3:TOC
# ratio, calibrated separately for secondary and tertiary effluent (Gerrity data
# over r = 0.5-1.0). Secondary effluent carries more TOC, so ozone decays faster
# (larger k) and less CT is attainable from the same dose.
_K_COEFFS = {"TERTIARY": (-0.6422, 0.8342), "SECONDARY": (-0.2637, 0.5251)}
# Ozone residual 30 s after application, C_30s = _C30_SLOPE*r + _C30_INTERCEPT
# [mg/L] (Pang et al., 2023).
_C30_SLOPE, _C30_INTERCEPT = 0.704, 1.136
# Required Cryptosporidium exposure per log reduction [mg.min/L].
_CT_PER_LRV = 2.47
# Duration of the instantaneous-ozone-demand window [min]; the residual sits at
# C_30s over it and decays first-order afterwards.
_IOD_MIN = 0.5
# Design minimum contactor HRT [min]. CA's ca_hrt_ct_constraint floors the solved
# contact time here: if 5 min already exceeds CTreq the HRT stays at 5 rather than
# shrinking below the design minimum.
_HRT_FLOOR_MIN = 5.0


def decay_constant(o3_to_toc, effluent_type):
    """First-order ozone decay constant k [1/min]. Plain floats, no Pyomo."""
    try:
        slope, intercept = _K_COEFFS[effluent_type.upper()]
    except KeyError:
        raise ConfigurationError(f"Unsupported effluent_type: {effluent_type}")
    return slope * o3_to_toc + intercept


def achieved_CT(contact_time, o3_to_toc, effluent_type):
    """CT [mg.min/L] achieved over ``contact_time`` [min]. Plain floats.

    Mirrors the ``CT_achieved`` Pyomo Expression built below; kept in sync by
    sharing the coefficients above. ``contact_time=None`` returns the asymptotic
    ceiling as contact_time -> infinity.
    """
    import math

    k = decay_constant(o3_to_toc, effluent_type)
    c30 = _C30_SLOPE * o3_to_toc + _C30_INTERCEPT
    if contact_time is None:  # tau -> inf, the exponential term vanishes
        return c30 * (_IOD_MIN + 1.0 / k)
    return c30 * (_IOD_MIN + (1.0 - math.exp(-k * (contact_time - _IOD_MIN))) / k)


def check_LRV_attainable(effluent_type, LRV, o3_to_toc_max=1.0, hrt_max=None):
    """Raise ConfigurationError if the required Cryptosporidium LRV cannot be met.

    Ozone decays first order, so the exposure integral converges: CT saturates at
    ``C_30s * (0.5 + 1/k)`` no matter how long the contactor is. A required LRV
    above that ceiling is unreachable at *any* contact time and any tank size --
    without this check it surfaces only as an IPOPT "converged to a locally
    infeasible point" after a full build and initialization, with nothing pointing
    at the LRV as the cause.

    Two separate failures are reported:
      * above the asymptotic ceiling -> impossible in principle;
      * reachable in principle but needing more contact time than ``hrt_max``
        -> a bound the caller could raise. Pass ``hrt_max=None`` to skip it.

    ``o3_to_toc_max`` is the largest dose the caller allows (CA pins it at 1.0;
    the other states cap it there too). CT is increasing in the dose -- a larger
    ratio raises C_30s and lowers k -- so evaluating at the maximum is the correct
    best case.
    """
    if LRV is None:
        return
    ct_req = _CT_PER_LRV * LRV
    ct_ceiling = achieved_CT(None, o3_to_toc_max, effluent_type)
    k = decay_constant(o3_to_toc_max, effluent_type)
    if ct_req > ct_ceiling:
        raise ConfigurationError(
            f"LRVO3_required={LRV} needs CT = {ct_req:.2f} mg.min/L, but "
            f"{effluent_type.upper()} effluent at O3:TOC={o3_to_toc_max:g} caps the "
            f"attainable CT at {ct_ceiling:.2f} mg.min/L (first-order decay "
            f"k={k:.4f} 1/min; CT saturates as contact time grows). This LRV is not "
            f"reachable at any contact time. Reduce the ozone LRV requirement or "
            f"credit the shortfall to another barrier."
        )
    if hrt_max is not None and achieved_CT(hrt_max, o3_to_toc_max, effluent_type) < ct_req:
        import math

        # Invert CT(tau) = ct_req for the contact time actually needed.
        c30 = _C30_SLOPE * o3_to_toc_max + _C30_INTERCEPT
        tau_req = _IOD_MIN - math.log(1.0 - k * (ct_req / c30 - _IOD_MIN)) / k
        raise ConfigurationError(
            f"LRVO3_required={LRV} needs CT = {ct_req:.2f} mg.min/L, which "
            f"{effluent_type.upper()} effluent at O3:TOC={o3_to_toc_max:g} reaches only "
            f"at a contact time of {tau_req:.2f} min -- above the {hrt_max:g} min upper "
            f"bound. Raise the contact-time bound to at least {tau_req:.2f} min, or "
            f"reduce the ozone LRV requirement."
        )


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
        if self.config.effluent_type.upper() not in _K_COEFFS:
            raise ConfigurationError(
                f"Unsupported effluent_type: {self.config.effluent_type}"
            )
        _k_slope, _k_int = _K_COEFFS[self.config.effluent_type.upper()]
        self.kO3 = pyo.Expression(
            self.flowsheet().time,
            rule=lambda b, t: (_k_slope * b.O3toTOC[t] + _k_int) * (1 / pyunits.min),
            doc=f"Reaction rate constant for {self.config.effluent_type.lower()} effluent",
        )

        # Fail fast on a required LRV the CT relationship cannot deliver, instead of
        # letting it surface as an opaque IPOPT infeasibility after a full build.
        # Bound-independent check only (asymptotic ceiling); the flowsheet applies the
        # contact-time-bound check, since it owns those bounds.
        check_LRV_attainable(
            self.config.effluent_type, self.config.LRVO3_required
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
            rule=lambda b, t: (_C30_SLOPE * b.O3toTOC[t] + _C30_INTERCEPT) * (pyunits.mg / pyunits.L) * (
                _IOD_MIN * pyunits.min + pyunits.convert(
                    (1 - pyo.exp(pyunits.convert(
                        -b.kO3[t] * (b.contact_time[t] - _IOD_MIN * pyunits.min),
                        to_units=pyunits.dimensionless))) / b.kO3[t],
                    to_units=pyunits.min)),
            doc="Achieved Cryptosporidium CT (mg.min/L)",
        )

        @self.Constraint(self.flowsheet().time, doc="O3:TOC determination (theoretical CT-based)")
        def O3toTOC_ratio_constraint(b, t):
            state = b.config.state
            LRV = b.config.LRVO3_required
            # An LRV above the asymptotic CT ceiling is rejected at build time by
            # check_LRV_attainable; an LRV that merely needs more contact time than the
            # flowsheet's bound allows is rejected by the flowsheet, which owns that
            # bound. Anything surviving both is reachable here.
            CTreq = _CT_PER_LRV * LRV * pyunits.mg * pyunits.min / pyunits.L
            k = b.kO3[t]
            exp_term = pyo.exp(pyunits.convert(
                -k * (b.contact_time[t] - _IOD_MIN * pyunits.min), to_units=pyunits.dimensionless
            ))
            denom = pyunits.convert(
                _IOD_MIN * pyunits.min * k + (1 - exp_term), to_units=pyunits.dimensionless
            )
            # O3:TOC required to meet the Cryptosporidium CT target.
            expr1 = (CTreq * k / denom - _C30_INTERCEPT * pyunits.mg / pyunits.L) / (
                _C30_SLOPE * pyunits.mg / pyunits.L
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
                CTreq = _CT_PER_LRV * LRV
                exp_ref = pyo.exp(pyunits.convert(
                    -b.kO3[t] * (_HRT_FLOOR_MIN * pyunits.min - _IOD_MIN * pyunits.min),
                    to_units=pyunits.dimensionless))
                CT_ref = (_C30_SLOPE * b.O3toTOC[t] + _C30_INTERCEPT) * (pyunits.mg / pyunits.L) * (
                    _IOD_MIN * pyunits.min + pyunits.convert(
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
