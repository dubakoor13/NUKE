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


## Cryogenic LH2 pump and vaporizing preheater physics

\`lh2_feed_cycle.py\` and \`run_lh2_feed_cycle.py\` extend the prior
**gas-only** network with a distinct, thermodynamically consistent
**cryogenic liquid H2 pump / lumped preheater / real-fluid warm core /
choked-nozzle operating-point model**.

The actual physical NERVA coolant sequence is more involved (tank,
pump, nozzle jacket, reflector, shield, core and turbine bleed) and
requires distributed phase change, losses, turbine-drive dynamics and
real pump curves. This implementation does **NOT** claim those
subsystems exist. It does not replace the legacy NASA-7 hot-hydrogen
engine code, and is not valid for historical hot-core temperatures.

### Equations implemented

Tank liquid H2 (temperature \`T1\`, pressure \`P1\`), followed by
an externally prescribed pump pressure head characteristic:

\`\`\`text
P2 = P1 + pump_shutoff_deltaP - Kpump * mdot^2

h2s = H(P2,s1)    // isentropic liquid pump
h2  = h1 + (h2s - h1) / eta_pump
Wpump = mdot*(h2-h1)

P3 = P2 - Kfeed * mdot^2
h3 = h2 + Qpreheater/mdot

h0_core_exit = h3 + (Qwall + Qdirect_H2) / mdot
Pchamber = Pcore_exit - Kchamber * mdot^2

mdot = At * choked_gas_mass_flux(Pchamber,Tchamber)
Wpump + Qpreheater + Qcore =
      mdot * (h0_chamber - h_tank)
\`\`\`

The code uses a **single CoolProp enthalpy/entropy reference** from
subcooled LH2 to gaseous core exit; pump isentropic efficiency is
calculated from actual Helmholtz hydrogen EOS values. The preheater
is permitted to cross the liquid/vapor phase boundary in a **lumped
isenthalpic-state flash**; however the core refuses liquid/two-phase
states and requires a gas inlet above 200 K. This is not a resolved
two-phase heat-exchanger/boiling model.

For gaseous core bins, mass conservation uses \`G=mdot/A\`; total
enthalpy adds kinetic energy \`h+v²/2\` and the momentum equation
includes both gas acceleration and Darcy friction, with CoolProp
density, viscosity and sonic speed. A chamber diffuser recovers
core-exit kinetic energy into stagnation enthalpy (with a separate
prescribed chamber pressure loss). The nozzle uses the existing
perfect-gas formula with **local** chamber \`gamma=cp/cv\` and
\`R_eff=P/(rho*T)\` obtained from CoolProp. This nozzle approximation
is **not** a real-fluid isentropic expansion solver.

### Execute numerical example

Requires the optional \`CoolProp\` package:

\`\`\`bash
python -m pip install CoolProp

python -m ntr.nerva.validate_lh2_feed_cycle \
  --write-example build/lh2_cycle_case.json

python -m ntr.nerva.run_lh2_feed_cycle \
  --config build/lh2_cycle_case.json \
  --output build/lh2_cycle_result.json
\`\`\`

The generated parameters and reactor heating are **hypothetical
software-physics test values**, not NERVA engine test settings. The
report contains pump discharge properties, shaft work, NPSH available,
warm gas channel pressure and temperature, nozzle pressure, flow,
thrust and Isp, with explicit pressure, enthalpy and mass residuals.

### Temperature and physical limitations

- CoolProp's Hydrogen Helmholtz EOS is limited to **Tmax 1000 K**.
  This code imposes **T <= 900 K** throughout the warm gas cycle.
  It CANNOT predict ~2500 K historical NERVA reactor heating and MUST
  NOT extrapolate CoolProp into that regime.
- Liquid pump is represented through *isentropic efficiency and a
  user-specified pressure-head curve*. No speed/torque turbine-map
  matching, vapor cavitation dynamics, NPSH required curve, turbine
  bleed, bearings, regenerative nozzle jacket or tie tube circuit.
- NPSH available is computed from liquid density and vapor pressure,
  but without a manufacturer NPSH-required curve it is NOT a pump
  cavitation qualification.
- Warm core gas uses single-phase H2 and bounded subsonic pressure
  solution. Preheater latent heating is an imposed lumped heat budget.
  No realistic two-phase startup/chilldown transient.
- Nuclear energy is externally assigned to preheater and channel
  bins; OpenMC cross-section data are not loaded here.
- Historical NERVA flowrates, pump pressure, thermal power and
  exhaust chemistry remain to be calibrated to documents and
  independently checked with engine data.

Background:
- CoolProp Hydrogen fluid limits: https://coolprop.org/fluid_properties/fluids/Hydrogen.html
- Historical NERVA engine flow path: NASA NTRS 19730062635
  https://ntrs.nasa.gov/citations/19730062635
- NERVA startup includes liquid / two-phase / gas zones:
  https://www.nasa.gov/wp-content/uploads/2025/06/b1-nerva-start-up-dynamics-and-control-1966.pdf



## Cryogenic LH2 to >1000 K hot reactor: matched enthalpy bridge

\`hot_lh2_cycle.py\` and \`run_hot_lh2_cycle.py\` link two
previously isolated physical regimes in NUKE:

1. **CoolProp HEOS::Hydrogen**: tank subcooled LH2, isentropic
   pump efficiency and shaft work, and lumped vaporization/heating
   up to a *single-phase warm gaseous* header, above 200 K and below
   900 K.
2. **NASA-7 frozen molecular H2**: subsonic compressible heated fuel
   channel (mass, stagnation enthalpy and momentum with wall friction
   and acceleration) through the 200–3500 K thermochemistry window.
3. **Frozen pure H2 nozzle**: determine choked mass flux from the
   full temperature-dependent NASA-7 enthalpy and entropy, then solve
   the expansion; no longer force a constant-gamma hot nozzle.

### Energy reference matching

CoolProp and NASA-7 enthalpy zeros are *not the same*. The two
numbers must **never** be subtracted as if they were:

\`\`\`text
h_reference_shift =
    h_CoolProp(T_bridge,P_bridge) - h_NASA7(T_bridge)

h_engine(T) = h_NASA7(T) + h_reference_shift

h_engine(T_bridge) = h_CoolProp(T_bridge,P_bridge)

pump_shaft_power + Q_preheater + Q_core
    = mdot * (h_engine(T_chamber) - h_CoolProp(T_tank,P_tank))
\`\`\`

The bridge uses a common *enthalpy reference* and thus maintains
the first law, but it does **not** make the two EOS identical.
The output explicitly reports the relative jump in \`cp\` and
density at the warm-gas reference temperature; neither is hidden.

### Coupled nozzle / pump / reactor channel mass flow

The pump pressure-source and hydraulic loss equations from the
preceding cryogenic model are retained. Fuel channel entry
acceleration is treated as adiabatic (stagnation enthalpy conserved).
The wall and directly deposited hydrogen nuclear power are counted
once each. The chamber recovers modeled channel exit kinetic energy
into stagnation enthalpy, and nozzle mass flow closes the system:

\`\`\`text
mdot = throat_area *
       choked_H2_NASA7_flux(P_chamber,T_chamber)
\`\`\`

The frozen-H2 nozzle is solved separately at the converged
operating point. A bounded flow root and explicit channel-Mach,
200–3500 K, pump-state and nozzle-domain checks cause hard
failure if no physically admissible solution exists.

Run an assumed design-case demonstration:

\`\`\`bash
python -m pip install CoolProp

python -m ntr.nerva.validate_hot_lh2_cycle \
  --write-example build/hot_lh2_case.json

python -m ntr.nerva.run_hot_lh2_cycle \
  --config build/hot_lh2_case.json \
  --output build/hot_lh2_result.json
\`\`\`

### Not yet a complete NERVA physical engine

- Nuclear energy deposition, pump curve, line losses, geometry and
  flow distribution are assumed, NOT computed from OpenMC or NERVA
  operating data in this example.
- The NASA-7 hot model assumes *pure frozen molecular H2* and ideal-gas
  behavior; high-temperature dissociation, non-equilibrium reactions
  and real-gas effects require additional physics.
- The real-fluid pump and the hot ideal-gas core are matched in
  **energy only**, not the full thermodynamic derivatives. A
  measured/validated EOS matching scheme remains future work.
- No nozzle jacket, reflector, tie-tube and turbine-bleed network
  yet shares this single reduced-order feed path. Pump hydraulic
  work is an externally supplied shaft input; turbine power,
  rotor speed and losses are not modeled.
- Core gas viscosity and conductivity remain screening approximations
  rather than high-temperature property fits.
- No neutron kinetics, detailed solid thermal conduction, full
  OpenMC feedback or experimental XE/NRX/NERVA validation in this CLI.



## Hot-core hydrogen dissociation: frozen vs local chemical equilibrium

The cryogenic-pump to hot-channel to nozzle model now has a
\`--chemistry-mode frozen|equilibrium\` option (default \`frozen\`
for backwards compatibility). It reuses NUKE's GRI-Mech NASA-7
H2/H species coefficients, valid from 200 to 3500 K; it is
**not** a NASA-CEA or finite-rate hydrogen kinetic model.

In \`equilibrium\` mode, every heated channel control volume
solves mass, total energy and momentum with the composition
determined by local chemical equilibrium:

\`\`\`text
H2 <-> 2H
alpha = alpha_equilibrium(T,P)
rho = P*M_H2 / [(1+alpha)*R_universal*T]

mdot[(h_out+v_out^2/2)-(h_in+v_in^2/2)]
    = Q_wall + Q_direct_H2

P_in-P_out =
    mass_flux*(v_out-v_in)
    + f_D*(dz/D)*mass_flux*(v_in+v_out)/4

h(T,P) = (1-alpha)*h_H2(T) + 2*alpha*h_H(T),
         expressed per conserved initial H2-equivalent kg
\`\`\`

Dissociation stores energy in chemical bonds instead of immediately
heating the gas. It changes density, channel acceleration, pressure,
chamber temperature and the choked nozzle mass-flow requirement.
The code inverts temperature against the *equilibrium* chemical
enthalpy rather than incorrectly using frozen heat capacity in
the energy Newton derivative; it uses a bounded bisection.

For subsonic gating, equilibrium-mode sound speed uses the local
thermodynamic derivative \`a^2=(dP/dRho)_s\`, including composition
response through the equilibrated entropy and density. It is a
numerical equilibrium acoustic-speed estimate and does not describe
finite-rate chemistry or dissociation relaxation.

The chamber and nozzle share the same chemistry assumption:
frozen molecular H2 throughout, or locally equilibrated H/H2.
The latter allows isentropic recombination during nozzle expansion.
The nozzle is a 1-D ideal-gas, ideal-chemistry approximation and
still uses a scalar nozzle kinetic-efficiency factor.

The cryogenic \`CoolProp\` pump/preheater to NASA-7 reference shift
remains matched at the warm-gas header, now with the chosen
composition model at that same pressure and temperature.
The bridge heat-capacity/density mismatch is exposed and the
overall pump + preheater + reactor heat budget is independently
audited. Actual reactor heating profiles and source curves remain
externally specified.

### Run and compare

\`\`\`bash
python -m ntr.nerva.validate_equilibrium_hot_hydrogen \
  --write-example build/hot_chemistry_case.json

python -m ntr.nerva.run_hot_lh2_cycle \
  --config build/hot_chemistry_case.json \
  --chemistry-mode frozen \
  --output build/hot_frozen.json

python -m ntr.nerva.run_hot_lh2_cycle \
  --config build/hot_chemistry_case.json \
  --chemistry-mode equilibrium \
  --output build/hot_equilibrium.json
\`\`\`

The example is a hypothetical subscale heat/mass-flow case; no
new OpenMC neutron transport, measured NERVA geometry or historical
test calibration is performed by this feature.

**Remaining physics:** equilibrium in every segment is the fast
reaction limit. Real finite-rate chemistry, vibrational non-equilibrium,
variable transport properties of atomic/molecular mixtures, ionization,
hot wall materials, feedback to nuclear data, multiple branch
hydraulics, turbine-drive dynamics, and validated high-temperature
fuel thermal conduction are absent. Existing Dittus-Boelter and Darcy
transport correlations still use screening-level fluid properties,
and cannot be taken as NERVA qualification without external data.
