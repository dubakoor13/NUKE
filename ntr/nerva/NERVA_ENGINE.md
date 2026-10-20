# Physics-based NERVA engine solver

The preferred NERVA engine-performance path is now:

```text
OpenMC deposited power
        |
        v
fuel/tie H2 thermal model
        |
        v
hot-H2 chamber state
        |
        v
nerva_engine.py
        |
        +-- choked throat mass flow
        +-- fixed-expansion-ratio nozzle
        +-- exit Mach / exit pressure
        +-- c*
        +-- thrust coefficient
        +-- thrust
        +-- Isp
```

## Governing coupling

For a fixed throat, the ideal choked hydrogen mass flux is

```text
mdot / At =
Pc * sqrt(gamma / (R Tc))
   * [2 / (gamma + 1)]^[(gamma + 1)/(2(gamma - 1))]
```

The nozzle exit Mach is solved from the isentropic area-Mach relation using the
specified expansion ratio. Exit pressure therefore follows from `Pc`,
`gamma`, and the solved exit Mach; it is not an independent input.

Thrust is

```text
F = mdot * Ve + (Pe - Pa) * Ae
```

and

```text
Isp = F / (mdot * g0)
```

For fixed `Tc`, gas properties, expansion ratio and vacuum ambient, changing
`Pc` through the same throat changes `mdot` and thrust approximately
proportionally while leaving ideal `Isp` essentially unchanged.

The reactor-power-coupled solver instead solves

```text
Q_to_H2 = mdot(Pc, Tc, At) * cp * (Tc - Tin)
```

so pressure can change Isp indirectly: higher pressure through the same throat
raises mass flow; at fixed reactor power that reduces the hydrogen temperature
rise, which then reduces exhaust velocity and Isp.

## Historical regression

The regression uses the public 1972 NERVA reference:

- thrust: 75,000 lbf
- chamber pressure: 450 psia
- chamber temperature: 2350-2500 K
- nozzle expansion ratio: 100:1
- published vacuum Isp: 825-850 s

With the first-order effective hot-H2 model (`gamma=1.35`) and 95% nozzle
kinetic-energy efficiency, the solver falls inside the published Isp band.

NASA references:

- NASA NERVA engine performance summary / 1972 flight-engine reference:
  https://ntrs.nasa.gov/citations/19710017291
- NASA TM-106739, NERVA-Type Nuclear Thermal Rocket Engines:
  https://ntrs.nasa.gov/citations/19950009268
- NASA NTP overview describing high-pressure pumped hydrogen through reactor
  coolant channels and ~900 s class performance:
  https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20120003776.pdf

## CLI

```bash
python -m ntr.nerva.run_nerva_engine --preset 1972-nerva
```

or

```bash
python -m ntr.nerva.run_nerva_engine --preset graphite-derivative
```

The output JSON includes chamber state, throat/exit areas, hydrogen mass flow,
exit pressure/Mach, characteristic velocity, thrust coefficient, thrust and
specific impulse.

## Fidelity boundary

This is a first-order hot-hydrogen nozzle/system solver. The constant
effective `gamma` and `cp` should eventually be replaced by equilibrium /
frozen high-temperature hydrogen chemistry. The module does not perform
criticality or fissile-loading calculations.


## Gas-phase pressure-source / core / nozzle network (new)

The physics implementation in \`gas_feed_network.py\` joins the
**existing compressible fuel-channel H2 solver** to the **existing
choked NERVA nozzle solver** and determines **total H2 flow and
chamber pressure together**. It is a reduced-order, *single-phase
gaseous-hydrogen* feed-network demonstration, NOT a cryogenic
NERVA LH2 turbopump or complete fuel/tie/periphery cycle.

For a prescribed pressure-source head curve (a USER-SUPPLIED
surrogate, not a recovered pump map), feed/exit resistance, channel
geometry, reactor heating, chamber nozzle throat and expansion
ratio, it solves the operating-point mass-flow residual:

\`\`\`text
p_discharge = p_supply + head0 - k_source * mdot^2
p_core_in = p_discharge - k_feed * mdot^2
p_chamber = p_core_out - k_exit * mdot^2
mdot - At * G_choked(p_chamber,T_chamber) = 0
\`\`\`

The gas compressor uses ideal-gas frozen molecular H2 NASA-7
enthalpy and entropy, with a prescribed isentropic efficiency:

\`\`\`text
s(T_2s,p_discharge) = s(T_supply,p_supply)
h_2 = h_supply + [h(T_2s)-h_supply]/eta
shaft_power = mdot*(h_2-h_supply)
\`\`\`

The channel solver includes local gas acceleration and friction
pressure losses, and conserves total energy \`h + v²/2\`.
Adiabatic acceleration from the compressor discharge plenum into
the channel conserves stagnation enthalpy. The channel-exit kinetic
energy is recovered as chamber stagnation enthalpy, with an
externally assumed exit pressure loss. The nozzle is evaluated with
its existing constant-\`gamma\`, choked ideal-gas model using
the **chamber-local** frozen H2 \`cp(T)\` and corresponding
\`gamma = cp/(cp-R)\`. This remains an effective-gamma nozzle
approximation, not a full variable-chemistry nozzle solution.

### Run example

\`\`\`bash
python -m ntr.nerva.validate_gas_feed_network \
  --write-example build/gas_feed_case.json

python -m ntr.nerva.run_gas_feed_network \
  --config build/gas_feed_case.json \
  --output build/gas_feed_result.json
\`\`\`

Inputs are explicitly hypothetical engineering test values, not
NERVA engine performance measurements. The numerical validation
checks pump-compressor isentropic efficiency, channel stagnation
energy and momentum, total mass/pressure/energy closure,
source-head sensitivity, heat-input sensitivity, source-exhaustion
and thermo-range failure handling.

### Physical fidelity boundary

- Supports **only single-phase gaseous pure molecular H2 in the
  200–3500 K NASA-7 domain**. A gas compressor is **NOT a physical
  substitute for the NERVA liquid hydrogen turbopump**. The model
  MUST NOT be used to predict cryogenic pump work, two-phase
  startup behavior, cavitation, NPSH, chilldown or pump cavitation.
- No tie-tube, reflector, cooling-jacket, valve or bypass circuit
  is included in this one-family reduced-order operating point.
  The supplied reactor heating is only that deposited in modeled
  fuel channels; the total reactor thermal power can be larger.
- Pressure-source head vs mass flow, pipeline loss constants,
  channel diameter and reactor heat distribution are prescribed.
  The solver does not infer pump geometry, RPM, shaft torque,
  turbine power extraction or material stress.
- Chamber/nozzle is a quasi-steady constant-gamma effective model;
  the expansion remains approximate for high-temperature
  dissociation or changing thermodynamic properties.
- The operating point is NOT linked to actual OpenMC statepoint
  deposition without a separately extracted matching input array.
  Reactor kinetics, fuel temperatures and experiment-based
  NERVA calibration remain future work.

This milestone is designed to prove that nozzle flow, chamber
pressure, compressor work and fuel-channel hydrogen heating do
not vary independently, while keeping unimplemented cycle physics
explicit instead of inventing LH2 turbopump behavior.
