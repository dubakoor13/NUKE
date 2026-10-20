"""NERVA hydrogen chemistry and 1-D thermodynamic nozzle reference solver.

NASA-polynomial-compatible, equilibrium/frozen H2 <-> 2H ideal-gas model.
Species coefficients are the *GRI-Mech 3.0 NASA-7* set, NOT NASA Glenn's
NASA-9 database or a complete NASA CEA implementation.

Validity: 200-3500 K, positive pressures. Single-phase ideal gas only.
It cannot represent liquid-H2 pump inlets, radiation/conduction/viscous
nozzle fields, ionization, or real-gas compressibility. This is a
reference-verifiable upgrade from constant cp/gamma, not a certified
NERVA flight-engine performance model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

RU = 8.31446261815324  # J/(mol K)
M_H2 = 0.00201588       # kg/mol, conserved molecular-hydrogen-equivalent mass
P_REF = 100000.0        # Pa, NASA-7 standard pressure convention used here
T_MIN = 200.0
T_MAX = 3500.0
G0 = 9.80665

# GRI-Mech 3.0 (NASA-7), 200 <= T < 1000, 1000 <= T <= 3500.
# https://combustion.berkeley.edu/gri-mech/version30/text30.html
# https://cantera.org/stable/reference/thermo/species-thermo.html
_H2_LOW = (2.34433112, 7.98052075e-3, -1.94781510e-5,
           2.01572094e-8, -7.37611761e-12, -917.935173, 0.683010238)
_H2_HIGH = (3.33727920, -4.94024731e-5, 4.99456778e-7,
            -1.79566394e-10, 2.00255376e-14, -950.158922, -3.20502331)
_H_LOW = (2.5, 7.05332819e-13, -1.99591964e-15,
          2.30081632e-18, -9.27732332e-22, 25473.6599, -0.446682853)
_H_HIGH = (2.50000001, -2.30842973e-11, 1.61561948e-14,
           -4.73515235e-18, 4.98197357e-22, 25473.6599, -0.446682914)


@dataclass(frozen=True)
class HydrogenState:
    temperature_k: float
    pressure_pa: float
    dissociation_fraction: float  # fraction of original H2 molecules dissociated
    density_kg_m3: float
    enthalpy_j_kg: float
    entropy_j_kg_k: float
    frozen_cp_j_kg_k: float


@dataclass(frozen=True)
class HydrogenNozzle:
    chemistry_mode: str
    chamber: HydrogenState
    throat: HydrogenState
    exit: HydrogenState
    throat_area_m2: float
    exit_area_m2: float
    mass_flux_kg_m2_s: float
    mass_flow_kg_s: float
    exit_velocity_m_s: float
    thrust_n: float
    isp_s: float
    enthalpy_to_kinetic_residual_j_kg: float


@dataclass(frozen=True)
class PowerCoupledHydrogenNozzle:
    reactor_thermal_power_w: float
    hydrogen_heating_power_w: float
    inlet_temperature_k: float
    chamber_temperature_k: float
    nozzle: HydrogenNozzle
    power_residual_w: float


def _species(T: float, low: tuple, high: tuple) -> tuple[float, float, float]:
    """Return molar cp, h, s for one species using NASA-7 polynomials."""
    if not T_MIN <= T <= T_MAX:
        raise ValueError("NASA-7 hydrogen valid only from 200 to 3500 K")
    a1, a2, a3, a4, a5, a6, a7 = low if T < 1000.0 else high
    cp = RU * (a1 + T*(a2 + T*(a3 + T*(a4 + T*a5))))
    h = RU * (T*(a1 + T*(a2/2 + T*(a3/3 + T*(a4/4 + T*a5/5)))) + a6)
    s = RU * (a1*math.log(T) + T*(a2 + T*(a3/2 + T*(a4/3 + T*a5/4))) + a7)
    return cp, h, s


def _dissociation_fraction(T: float, P: float) -> float:
    """Mass-action equilibrium for H2(g) <-> 2 H(g), ideal gas."""
    _, h2_h, h2_s = _species(T, _H2_LOW, _H2_HIGH)
    _, h_h, h_s = _species(T, _H_LOW, _H_HIGH)
    delta_g_over_rt = (2*(h_h - T*h_s) - (h2_h - T*h2_s))/(RU*T)
    # Alpha^2 = Kp/(Kp + 4P/Pref), numerically stable in log space.
    x = delta_g_over_rt + math.log(4.0*P/P_REF)
    if x > 700.0:
        return 0.0
    if x < -700.0:
        return 1.0
    return math.sqrt(1.0/(1.0 + math.exp(x)))


def hydrogen_state(
    temperature_k: float, pressure_pa: float,
    mode: str = "equilibrium", frozen_dissociation: float | None = None
) -> HydrogenState:
    """Evaluate mass-specific hydrogen mixture state with consistent mixing entropy.

    mode='equilibrium' evaluates the dissociation reaction at (T, P).
    mode='frozen' holds the supplied chamber dissociation, defaulting to pure H2.
    """
    T, P = float(temperature_k), float(pressure_pa)
    if not math.isfinite(T) or not T_MIN <= T <= T_MAX:
        raise ValueError("temperature outside NASA-7 valid range 200-3500 K")
    if not math.isfinite(P) or P <= 0.0:
        raise ValueError("pressure must be positive and finite")
    if mode not in ("frozen", "equilibrium"):
        raise ValueError("mode must be frozen or equilibrium")
    if mode == "equilibrium":
        if frozen_dissociation is not None:
            raise ValueError("frozen_dissociation only applies to frozen mode")
        a = _dissociation_fraction(T, P)
    else:
        a = 0.0 if frozen_dissociation is None else float(frozen_dissociation)
        if not math.isfinite(a) or not 0.0 <= a <= 1.0:
            raise ValueError("frozen dissociation must be between 0 and 1")
    cp2, h2, s2 = _species(T, _H2_LOW, _H2_HIGH)
    cph, hh, sh = _species(T, _H_LOW, _H_HIGH)
    n2, nh = 1.0 - a, 2.0*a
    ntotal = 1.0 + a
    # Ideal-mixing partial pressures are n_i/(1+a) * P.
    mixing = 0.0
    if n2 > 0.0:
        mixing += n2*math.log((n2/ntotal)*(P/P_REF))
    if nh > 0.0:
        mixing += nh*math.log((nh/ntotal)*(P/P_REF))
    entropy = (n2*s2 + nh*sh - RU*mixing)/M_H2
    enthalpy = (n2*h2 + nh*hh)/M_H2
    density = P*M_H2/(ntotal*RU*T)
    cp_frozen = (n2*cp2 + nh*cph)/M_H2
    return HydrogenState(T, P, a, density, enthalpy, entropy, cp_frozen)


def _isentrope(
    P: float, chamber: HydrogenState,
    mode: str, frozen_dissociation: float | None
) -> HydrogenState:
    """Return expansion state at pressure P with chamber entropy conserved."""
    if P > chamber.pressure_pa or P <= 0.0:
        raise ValueError("isentropic expansion pressure outside range")
    lower, upper = T_MIN, chamber.temperature_k
    s_low = hydrogen_state(lower, P, mode, frozen_dissociation).entropy_j_kg_k
    if s_low > chamber.entropy_j_kg_k + 1.0e-6:
        raise ValueError("expansion requires T below NASA-7 range")
    for _ in range(46):
        m = 0.5*(lower + upper)
        sm = hydrogen_state(m, P, mode, frozen_dissociation).entropy_j_kg_k
        if sm < chamber.entropy_j_kg_k:
            lower = m
        else:
            upper = m
    return hydrogen_state(0.5*(lower+upper), P, mode, frozen_dissociation)


def _minimum_pressure(
    chamber: HydrogenState, mode: str, frozen_dissociation: float | None
) -> float:
    """Minimum isentropic pressure consistent with T>=200 K."""
    slo = math.log(chamber.pressure_pa) - 40.0
    shi = math.log(chamber.pressure_pa)
    for _ in range(62):
        sm = 0.5*(slo + shi)
        state = hydrogen_state(T_MIN, math.exp(sm), mode, frozen_dissociation)
        if state.entropy_j_kg_k > chamber.entropy_j_kg_k:
            slo = sm
        else:
            shi = sm
    return math.exp(shi) * (1 + 2e-12)


def hydrogen_nozzle(
    chamber_pressure_pa: float, chamber_temperature_k: float,
    throat_area_m2: float, expansion_ratio: float,
    mode: str = "equilibrium", nozzle_efficiency: float = 1.0,
    ambient_pressure_pa: float = 0.0
) -> HydrogenNozzle:
    """Isentropic reacting/frozen nozzle, sonic throat from max mass flux.

    Flow is an ideal-gas thermochemical model; nozzle_efficiency is an
    explicitly approximate scalar kinetic-energy loss. No axial chemistry
    rate equation, shock/viscous losses, heat transfer, or two-phase effects.
    """
    if not math.isfinite(throat_area_m2) or throat_area_m2 <= 0.0:
        raise ValueError("throat area must be positive")
    if not math.isfinite(expansion_ratio) or expansion_ratio <= 1.0:
        raise ValueError("expansion ratio must exceed one")
    if not math.isfinite(nozzle_efficiency) or not 0 < nozzle_efficiency <= 1:
        raise ValueError("nozzle efficiency must be in (0, 1]")
    if not math.isfinite(ambient_pressure_pa) or ambient_pressure_pa < 0:
        raise ValueError("ambient pressure cannot be negative")
    chamber = hydrogen_state(chamber_temperature_k, chamber_pressure_pa, mode)
    frozen_alpha = chamber.dissociation_fraction if mode == "frozen" else None
    pmin = _minimum_pressure(chamber, mode, frozen_alpha)
    if pmin >= chamber.pressure_pa:
        raise ValueError("insufficient thermodynamic temperature span")

    def state_flow(lp: float) -> tuple[float, HydrogenState]:
        state = _isentrope(math.exp(lp), chamber, mode, frozen_alpha)
        kinetic = max(0.0, 2.0*(chamber.enthalpy_j_kg - state.enthalpy_j_kg))
        return state.density_kg_m3*math.sqrt(kinetic), state

    # Maximum mass-flux criterion (sonic throat for stable equilibrium/frozen
    # isentropes), instead of constant-gamma choking approximations.
    left, right = math.log(pmin), math.log(chamber.pressure_pa)
    golden = (math.sqrt(5.0)-1.0)/2.0
    a, b = right-golden*(right-left), left+golden*(right-left)
    fa, _ = state_flow(a)
    fb, _ = state_flow(b)
    for _ in range(52):
        if fa < fb:
            left, a, fa = a, b, fb
            b = left+golden*(right-left)
            fb, _ = state_flow(b)
        else:
            right, b, fb = b, a, fa
            a = right-golden*(right-left)
            fa, _ = state_flow(a)
    throat_log = 0.5*(left+right)
    gstar, throat = state_flow(throat_log)
    target = gstar/expansion_ratio
    gmin, _ = state_flow(math.log(pmin))
    if gmin >= target:
        raise ValueError("exit temperature below 200 K for requested area ratio")

    # Supersonic branch of the area-Mach relation, generalized through
    # conservation of entropy/enthalpy and flow continuity.
    low, high = math.log(pmin), throat_log
    for _ in range(49):
        mid = 0.5*(low+high)
        flow, _ = state_flow(mid)
        if flow < target:
            low = mid
        else:
            high = mid
    flow_exit, exit_state = state_flow(0.5*(low+high))
    exit_velocity_ideal = math.sqrt(
        max(0.0, 2.0*(chamber.enthalpy_j_kg - exit_state.enthalpy_j_kg))
    )
    exit_velocity = math.sqrt(nozzle_efficiency)*exit_velocity_ideal
    mdot = gstar*throat_area_m2
    area_exit = throat_area_m2*expansion_ratio
    thrust = mdot*exit_velocity + (exit_state.pressure_pa-ambient_pressure_pa)*area_exit
    residual = (chamber.enthalpy_j_kg - exit_state.enthalpy_j_kg) - 0.5*exit_velocity_ideal**2
    if thrust <= 0:
        raise ValueError("nozzle has nonpositive net thrust for this ambient pressure")
    return HydrogenNozzle(
        chemistry_mode=mode, chamber=chamber, throat=throat, exit=exit_state,
        throat_area_m2=throat_area_m2, exit_area_m2=area_exit,
        mass_flux_kg_m2_s=gstar, mass_flow_kg_s=mdot,
        exit_velocity_m_s=exit_velocity, thrust_n=thrust,
        isp_s=thrust/(mdot*G0), enthalpy_to_kinetic_residual_j_kg=residual
    )


def power_coupled_hydrogen_nozzle(
    reactor_thermal_power_w: float, chamber_pressure_pa: float,
    inlet_temperature_k: float, throat_area_m2: float, expansion_ratio: float,
    heating_efficiency: float = 0.95, mode: str = "equilibrium",
    nozzle_efficiency: float = 1.0
) -> PowerCoupledHydrogenNozzle:
    """Solve Q = mdot [h(Tc,Pc)-h(Tin,Pc)] at fixed Pc and throat area.

    Assumes gaseous H2 already at Pc at the inlet; excludes cryogenic feed,
    turbopump, tie tubes, reflector heating, and reactor neutronic feedback.
    """
    if not math.isfinite(reactor_thermal_power_w) or reactor_thermal_power_w <= 0:
        raise ValueError("reactor power must be positive")
    if not 0 < heating_efficiency <= 1:
        raise ValueError("heating efficiency must be in (0,1]")
    if not T_MIN <= inlet_temperature_k < T_MAX:
        raise ValueError("inlet must be gaseous hydrogen, 200-3500 K")
    if throat_area_m2 <= 0 or expansion_ratio <= 1:
        raise ValueError("invalid nozzle geometry")
    inlet = hydrogen_state(inlet_temperature_k, chamber_pressure_pa, mode)
    available = reactor_thermal_power_w*heating_efficiency

    def residual(T: float) -> float:
        result = hydrogen_nozzle(
            chamber_pressure_pa, T, throat_area_m2, expansion_ratio, mode
        )
        return (result.mass_flow_kg_s
                *(result.chamber.enthalpy_j_kg-inlet.enthalpy_j_kg) - available)

    upper = T_MAX
    if residual(upper) < 0:
        raise ValueError("reactor power requires T above 3500 K for this flow")
    lower = inlet_temperature_k
    for _ in range(37):
        mid = 0.5*(lower+upper)
        if residual(mid) < 0:
            lower = mid
        else:
            upper = mid
    Tc = 0.5*(lower+upper)
    result = hydrogen_nozzle(
        chamber_pressure_pa, Tc, throat_area_m2, expansion_ratio,
        mode=mode, nozzle_efficiency=nozzle_efficiency
    )
    closure = result.mass_flow_kg_s*(result.chamber.enthalpy_j_kg-inlet.enthalpy_j_kg)-available
    return PowerCoupledHydrogenNozzle(
        reactor_thermal_power_w=reactor_thermal_power_w,
        hydrogen_heating_power_w=available,
        inlet_temperature_k=inlet_temperature_k,
        chamber_temperature_k=Tc,
        nozzle=result, power_residual_w=closure
    )
