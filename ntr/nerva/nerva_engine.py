"""Physics-based first-order NERVA engine/nozzle model.

This module is intentionally separate from reactor criticality. It starts from
the hot-hydrogen chamber state produced by the thermal model and applies
compressible-flow/nozzle equations.

Important pressure/Isp relationship
-----------------------------------
For a calorically-perfect gas, fixed chamber temperature, fixed nozzle
expansion ratio, fixed gas properties, and vacuum ambient:
    * choked mass flow through a fixed throat is proportional to chamber pressure;
    * thrust through that fixed throat is proportional to chamber pressure;
    * specific impulse is essentially independent of chamber pressure.

Pressure affects Isp indirectly when it changes chamber temperature, chemistry,
nozzle losses, or ambient-pressure matching. The power-coupled solver below
captures the important NTR coupling: with fixed reactor power and throat area,
higher chamber pressure drives more hydrogen mass flow, which lowers the
temperature rise and therefore changes Isp.

The default effective gas properties are an engineering approximation chosen
for hot molecular hydrogen. Production work should replace them with an
equilibrium/frozen hydrogen chemistry model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


G0_M_S2 = 9.80665
HYDROGEN_R_J_KG_K = 4124.0


@dataclass(frozen=True)
class NervaGasModel:
    """Effective hot-hydrogen gas model for first-order NERVA calculations."""

    gamma: float = 1.35
    gas_constant_j_kg_k: float = HYDROGEN_R_J_KG_K
    cp_j_kg_k: float = 14300.0

    def validate(self) -> None:
        if self.gamma <= 1.0:
            raise ValueError("gamma must exceed 1")
        if self.gas_constant_j_kg_k <= 0.0:
            raise ValueError("gas constant must be positive")
        if self.cp_j_kg_k <= 0.0:
            raise ValueError("cp must be positive")


@dataclass(frozen=True)
class NervaNozzleResult:
    chamber_pressure_pa: float
    chamber_temperature_k: float
    ambient_pressure_pa: float
    throat_area_m2: float
    exit_area_m2: float
    expansion_ratio: float
    exit_mach: float
    exit_pressure_pa: float
    mass_flow_kg_s: float
    characteristic_velocity_m_s: float
    thrust_coefficient: float
    exit_velocity_m_s: float
    momentum_thrust_n: float
    pressure_thrust_n: float
    thrust_n: float
    effective_exhaust_velocity_m_s: float
    isp_s: float


@dataclass(frozen=True)
class PowerCoupledNervaResult:
    reactor_thermal_power_w: float
    propellant_heating_power_w: float
    propellant_heating_efficiency: float
    inlet_temperature_k: float
    chamber_temperature_k: float
    nozzle: NervaNozzleResult


def area_ratio_from_mach(mach: float, gamma: float) -> float:
    """Return A/A* for an isentropic perfect-gas nozzle."""
    if mach <= 0.0:
        raise ValueError("mach must be positive")
    if gamma <= 1.0:
        raise ValueError("gamma must exceed 1")

    return (
        1.0
        / mach
        * (
            2.0
            / (gamma + 1.0)
            * (1.0 + 0.5 * (gamma - 1.0) * mach * mach)
        )
        ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    )


def supersonic_mach_from_area_ratio(
    expansion_ratio: float,
    gamma: float,
) -> float:
    """Invert the isentropic area-Mach relation on the supersonic branch."""
    if expansion_ratio < 1.0:
        raise ValueError("expansion_ratio must be >= 1")
    if gamma <= 1.0:
        raise ValueError("gamma must exceed 1")
    if math.isclose(expansion_ratio, 1.0):
        return 1.0

    lower = 1.0 + 1.0e-10
    upper = 50.0

    for _ in range(160):
        middle = 0.5 * (lower + upper)
        ratio = area_ratio_from_mach(middle, gamma)
        if ratio < expansion_ratio:
            lower = middle
        else:
            upper = middle

    return 0.5 * (lower + upper)


def choked_mass_flux_kg_m2_s(
    chamber_pressure_pa: float,
    chamber_temperature_k: float,
    gas: NervaGasModel | None = None,
) -> float:
    """Return ideal choked mass flux at the nozzle throat."""
    if gas is None:
        gas = NervaGasModel()
    gas.validate()

    if chamber_pressure_pa <= 0.0:
        raise ValueError("chamber pressure must be positive")
    if chamber_temperature_k <= 0.0:
        raise ValueError("chamber temperature must be positive")

    gamma = gas.gamma
    gas_constant = gas.gas_constant_j_kg_k

    return (
        chamber_pressure_pa
        * math.sqrt(gamma / (gas_constant * chamber_temperature_k))
        * (2.0 / (gamma + 1.0))
        ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    )


def characteristic_velocity_m_s(
    chamber_temperature_k: float,
    gas: NervaGasModel | None = None,
) -> float:
    """Return ideal characteristic velocity c* for the hot hydrogen."""
    if gas is None:
        gas = NervaGasModel()
    flux_at_unit_pressure = choked_mass_flux_kg_m2_s(
        chamber_pressure_pa=1.0,
        chamber_temperature_k=chamber_temperature_k,
        gas=gas,
    )
    return 1.0 / flux_at_unit_pressure


def evaluate_nerva_nozzle(
    chamber_pressure_pa: float,
    chamber_temperature_k: float,
    throat_area_m2: float,
    expansion_ratio: float,
    ambient_pressure_pa: float = 0.0,
    nozzle_efficiency: float = 0.95,
    gas: NervaGasModel | None = None,
) -> NervaNozzleResult:
    """Evaluate a fixed-geometry NERVA hydrogen nozzle.

    The exit pressure is solved from the expansion ratio rather than supplied
    independently. This keeps chamber pressure, mass flow, exit state, pressure
    thrust, thrust coefficient, and Isp thermodynamically consistent.
    """
    if gas is None:
        gas = NervaGasModel()
    gas.validate()

    if chamber_pressure_pa <= 0.0:
        raise ValueError("chamber pressure must be positive")
    if chamber_temperature_k <= 0.0:
        raise ValueError("chamber temperature must be positive")
    if throat_area_m2 <= 0.0:
        raise ValueError("throat area must be positive")
    if expansion_ratio < 1.0:
        raise ValueError("expansion ratio must be >= 1")
    if ambient_pressure_pa < 0.0:
        raise ValueError("ambient pressure must be non-negative")
    if not (0.0 < nozzle_efficiency <= 1.0):
        raise ValueError("nozzle efficiency must lie in (0, 1]")

    gamma = gas.gamma
    gas_constant = gas.gas_constant_j_kg_k

    exit_mach = supersonic_mach_from_area_ratio(expansion_ratio, gamma)
    exit_pressure_ratio = (
        1.0 + 0.5 * (gamma - 1.0) * exit_mach**2
    ) ** (-gamma / (gamma - 1.0))
    exit_pressure_pa = chamber_pressure_pa * exit_pressure_ratio

    mass_flux = choked_mass_flux_kg_m2_s(
        chamber_pressure_pa,
        chamber_temperature_k,
        gas,
    )
    mass_flow = mass_flux * throat_area_m2

    ideal_exit_velocity = math.sqrt(
        2.0
        * gamma
        / (gamma - 1.0)
        * gas_constant
        * chamber_temperature_k
        * (
            1.0
            - exit_pressure_ratio ** ((gamma - 1.0) / gamma)
        )
    )
    exit_velocity = math.sqrt(nozzle_efficiency) * ideal_exit_velocity

    exit_area = throat_area_m2 * expansion_ratio
    momentum_thrust = mass_flow * exit_velocity
    pressure_thrust = (
        exit_pressure_pa - ambient_pressure_pa
    ) * exit_area
    thrust = momentum_thrust + pressure_thrust
    effective_velocity = thrust / mass_flow
    isp = effective_velocity / G0_M_S2

    c_star = chamber_pressure_pa * throat_area_m2 / mass_flow
    thrust_coefficient = thrust / (
        chamber_pressure_pa * throat_area_m2
    )

    return NervaNozzleResult(
        chamber_pressure_pa=float(chamber_pressure_pa),
        chamber_temperature_k=float(chamber_temperature_k),
        ambient_pressure_pa=float(ambient_pressure_pa),
        throat_area_m2=float(throat_area_m2),
        exit_area_m2=float(exit_area),
        expansion_ratio=float(expansion_ratio),
        exit_mach=float(exit_mach),
        exit_pressure_pa=float(exit_pressure_pa),
        mass_flow_kg_s=float(mass_flow),
        characteristic_velocity_m_s=float(c_star),
        thrust_coefficient=float(thrust_coefficient),
        exit_velocity_m_s=float(exit_velocity),
        momentum_thrust_n=float(momentum_thrust),
        pressure_thrust_n=float(pressure_thrust),
        thrust_n=float(thrust),
        effective_exhaust_velocity_m_s=float(effective_velocity),
        isp_s=float(isp),
    )


def throat_area_for_target_thrust(
    target_thrust_n: float,
    chamber_pressure_pa: float,
    chamber_temperature_k: float,
    expansion_ratio: float,
    ambient_pressure_pa: float = 0.0,
    nozzle_efficiency: float = 0.95,
    gas: NervaGasModel | None = None,
) -> float:
    """Size throat area for a target thrust at a specified chamber state."""
    if target_thrust_n <= 0.0:
        raise ValueError("target thrust must be positive")

    unit = evaluate_nerva_nozzle(
        chamber_pressure_pa=chamber_pressure_pa,
        chamber_temperature_k=chamber_temperature_k,
        throat_area_m2=1.0,
        expansion_ratio=expansion_ratio,
        ambient_pressure_pa=ambient_pressure_pa,
        nozzle_efficiency=nozzle_efficiency,
        gas=gas,
    )
    return target_thrust_n / unit.thrust_n


def _power_balance_residual(
    chamber_temperature_k: float,
    reactor_thermal_power_w: float,
    propellant_heating_efficiency: float,
    inlet_temperature_k: float,
    chamber_pressure_pa: float,
    throat_area_m2: float,
    gas: NervaGasModel,
) -> float:
    mass_flow = (
        choked_mass_flux_kg_m2_s(
            chamber_pressure_pa,
            chamber_temperature_k,
            gas,
        )
        * throat_area_m2
    )
    required_power = (
        mass_flow
        * gas.cp_j_kg_k
        * (chamber_temperature_k - inlet_temperature_k)
    )
    available_power = (
        reactor_thermal_power_w * propellant_heating_efficiency
    )
    return required_power - available_power


def solve_power_coupled_nerva(
    reactor_thermal_power_w: float,
    chamber_pressure_pa: float,
    throat_area_m2: float,
    expansion_ratio: float,
    inlet_temperature_k: float,
    propellant_heating_efficiency: float = 0.95,
    ambient_pressure_pa: float = 0.0,
    nozzle_efficiency: float = 0.95,
    gas: NervaGasModel | None = None,
    maximum_temperature_k: float = 5000.0,
) -> PowerCoupledNervaResult:
    """Solve chamber temperature from reactor power and choked nozzle flow.

    This is the key NTR system coupling:
        Q_to_H2 = mdot(Pc, Tc, At) * cp * (Tc - Tin)

    Pc therefore changes mass flow directly. At fixed reactor power and throat
    area, changing Pc changes Tc; Isp then changes through Tc rather than via an
    artificial direct Pc->Isp scaling.
    """
    if gas is None:
        gas = NervaGasModel()
    gas.validate()

    if reactor_thermal_power_w <= 0.0:
        raise ValueError("reactor thermal power must be positive")
    if chamber_pressure_pa <= 0.0:
        raise ValueError("chamber pressure must be positive")
    if throat_area_m2 <= 0.0:
        raise ValueError("throat area must be positive")
    if inlet_temperature_k <= 0.0:
        raise ValueError("inlet temperature must be positive")
    if not (0.0 < propellant_heating_efficiency <= 1.0):
        raise ValueError(
            "propellant_heating_efficiency must lie in (0, 1]"
        )
    if maximum_temperature_k <= inlet_temperature_k:
        raise ValueError(
            "maximum_temperature_k must exceed inlet temperature"
        )

    lower = inlet_temperature_k + 1.0e-6
    upper = maximum_temperature_k

    r_lower = _power_balance_residual(
        lower,
        reactor_thermal_power_w,
        propellant_heating_efficiency,
        inlet_temperature_k,
        chamber_pressure_pa,
        throat_area_m2,
        gas,
    )
    r_upper = _power_balance_residual(
        upper,
        reactor_thermal_power_w,
        propellant_heating_efficiency,
        inlet_temperature_k,
        chamber_pressure_pa,
        throat_area_m2,
        gas,
    )

    if r_lower > 0.0:
        raise ValueError("power-balance root is below inlet temperature")
    if r_upper < 0.0:
        raise ValueError(
            "reactor power requires a chamber temperature above "
            "maximum_temperature_k; increase throat area/pressure or "
            "raise the explicit temperature limit"
        )

    for _ in range(180):
        middle = 0.5 * (lower + upper)
        residual = _power_balance_residual(
            middle,
            reactor_thermal_power_w,
            propellant_heating_efficiency,
            inlet_temperature_k,
            chamber_pressure_pa,
            throat_area_m2,
            gas,
        )
        if residual < 0.0:
            lower = middle
        else:
            upper = middle

    chamber_temperature = 0.5 * (lower + upper)
    nozzle = evaluate_nerva_nozzle(
        chamber_pressure_pa=chamber_pressure_pa,
        chamber_temperature_k=chamber_temperature,
        throat_area_m2=throat_area_m2,
        expansion_ratio=expansion_ratio,
        ambient_pressure_pa=ambient_pressure_pa,
        nozzle_efficiency=nozzle_efficiency,
        gas=gas,
    )

    return PowerCoupledNervaResult(
        reactor_thermal_power_w=float(reactor_thermal_power_w),
        propellant_heating_power_w=float(
            reactor_thermal_power_w * propellant_heating_efficiency
        ),
        propellant_heating_efficiency=float(
            propellant_heating_efficiency
        ),
        inlet_temperature_k=float(inlet_temperature_k),
        chamber_temperature_k=float(chamber_temperature),
        nozzle=nozzle,
    )
