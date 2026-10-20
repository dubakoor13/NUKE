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



## Pump–turbine shaft closure using warm H2 bleed

The \`turbine_bleed_cycle.py\` operating-point solver now represents
the first closed pump drive balance in the NUKE reduced-order
LH2 -> hot hydrogen -> nozzle cycle.

**Flow path:** subcooled tank LH2 passes through the CoolProp pump
and lumped preheater. Heated gas divides into a main reactor flow
and a separate hydrogen turbine bleed. The turbine expands from
the warm header to a prescribed exhaust pressure, and supplies
the pump shaft demand after mechanical losses. Only the main
flow enters the NASA-7 hot reactor-channel model and choked nozzle.
The turbine exhaust is modeled as a **separate outlet**, not
silently counted as main-nozzle propellant.

At a proposed total tank feed \`m_tank\`, the solver evaluates:

\`\`\`text
delta_h_pump_actual = (h_2s - h_1) / eta_pump
W_pump = m_tank * delta_h_pump_actual

h_4s = h(P_turbine_exhaust,s_warm_header)
w_turbine = eta_turbine*(h_warm_header - h_4s)

m_bleed = W_pump / (eta_mechanical*w_turbine)
m_core  = m_tank - m_bleed

m_core = throat_area*choked_mass_flux(P_chamber,T_chamber)
W_turbine*eta_mechanical = W_pump

m_core*h_chamber + m_bleed*h_turb_exhaust + W_mechanical_loss
    = m_tank*h_tank + Q_preheater + Q_core
\`\`\`

\`h_chamber\` is in the same energy-reference convention as the
CoolProp tank and warm header, using the matched NASA-7 enthalpy
shift from the previous upgrade. The hot channel and nozzle accept
either frozen molecular H2 or equilibrated H2/H thermochemistry;
the **turbine branch** uses CoolProp at valid, warm temperatures.

The root solver varies **total tank mass flow** while analytically
solving the required turbine bleed fraction at each trial. The
nozzle closure uses **core mass flow only**. It rejects flow rates
which would exhaust pump head, exceed specified allowable turbine
bleed, leave the gaseous domain, exceed channel Mach limits, or
fail to produce an admissible source/nozzle operating point.

### Example (assumed parameters, not historical NERVA data)

\`\`\`bash
python -m pip install CoolProp
python -m ntr.nerva.validate_turbine_bleed_cycle \
  --write-example build/turbine_bleed_case.json

python -m ntr.nerva.run_turbine_bleed_cycle \
  --config build/turbine_bleed_case.json \
  --chemistry-mode frozen \
  --output build/turbine_bleed_result.json
\`\`\`

The report distinguishes **main-nozzle Isp** based on
\`m_core\` from **effective Isp using total tank flow**:
\`Isp_effective_no_bleed_thrust = thrust_main/(g0*m_tank)\`.
This is a partial-cycle performance estimate; the turbine exhaust
might contribute propulsion, but *its thrust is not calculated* here.

**Remaining physics:** no real turbine speed/torque/blade/efficiency
map, dynamic shaft inertia, pressure/temperature-dependent turbine
efficiency curve, nozzle-jacket/reflector/tie coolant branches,
cryogenic chilldown/startup, bleed exhaust expansion/thrust,
or OpenMC-derived hot-core nuclear heat in this example. The
preheater heat source is externally specified rather than obtained
from a routed regenerative cooling jacket or reflector.



## Parallel nozzle jacket / reflector / tie-tube regenerative cooling (new)

\`regenerative_cooling.py\` implements **explicit** first-order
coolant heat balances and pressure-matched flow splitting rather
than the previous single abstract preheater. The modeled topology
is derived from NASA/TM-2013-216553, *Nuclear Thermal Rocket Simulation
in NPSS* (Belair et al., July 2013), "Engine Flow Path", p. 6:
https://ntrs.nasa.gov/api/citations/20140006199/downloads/20140006199.pdf

\`\`\`text
tank LH2 -> pump -> feed valve ->+-> nozzle cooling -> reflector -----+
                               |                                     |
                               +-> tie-tube supply -> tie return -----+--> hot gas header
                                                                           |
                                                               turbine bleed / core
\`\`\`

The tie supply flows in the axial direction opposite the return, but
the **thermal wall-to-gas power split is prescribed**; this is not a
conjugate zirconium-hydride solid conduction solver. This branch model
does not yet implement NASA's expander-cycle downstream turbine
bypass/flow recombination: it retains the existing separately
exhausted turbine-bleed simplification. This distinction matters
when comparing against specific historical NERVA engines; classical
NERVA hot-bleed engine routing differs from later small NERVA-derived
expander cycles.

For four user-specified cooling legs, the finite-volume physics is:

\`\`\`text
m_path*(h0_out-h0_in) = Q_leg
h0 = h(P,h_static)+v^2/2,    v=G/rho(P,h_static)
P_in-P_out = G*(v_out-v_in)+K*G*(v_in+v_out)/4
G = m_path / flow_area

m_reflector + m_tie = m_total
P_reflector_out = P_tie_return_out
h0_header = (m_reflector*h0_reflector +
             m_tie*h0_tie_return) / m_total
\`\`\`

\`K\` combines Darcy path friction and fitting/local losses; it is
**an assumed hydraulic loss coefficient**, not yet calculated from
a validated two-phase friction, wall boiling, or temperature-dependent
flow correlation. CoolProp supplies homogeneous-equilibrium H2
density/enthalpy (including phase change) to make these equations
self-consistent at **fluid-bulk screening level**. No separate
liquid/vapor velocities, slip, critical heat flux, film boiling,
or flow-instability criterion is yet implemented. The solver
refuses nonphysical temperatures, failed hydraulic balance,
and bad heat/mass/enthalpy input.

The pump compresses TOTAL tank hydrogen, with the same temperature-
dependent CoolProp isentropic-efficiency model from earlier work.
The downstream turbine and hot-channel/nozzle follow the already
implemented shaft-closed operating-point equations. Crucially,
preheater heat is NOT added twice: the sum of the four independent
cooling-leg powers must equal the \`LH2Cycle.preheater_thermal_power_w\`
bookkeeping field.

### Example run

\`\`\`bash
python -m ntr.nerva.validate_regenerative_cooling \
  --write-example build/regenerative_cooling_case.json

python -m ntr.nerva.run_turbine_bleed_cycle \
  --config build/regenerative_cooling_case.json \
  --chemistry-mode frozen \
  --output build/regenerative_cooling_result.json
\`\`\`

The JSON adds a \`regenerative_cooling\` object with \`nozzle_jacket\`,
\`reflector\`, \`tie_supply\`, and \`tie_return\` entries. Each entry
contains \`heat_power_w\`, \`flow_area_m2\`, and
\`pressure_loss_coefficient\`. Leaving the object out runs the
previous lumped-preheater model unchanged. Reports record all
per-leg states, two-phase endpoint quality diagnostic, pressure
and energy residuals, tie-vs-reflector mass-flow split, mixed
header enthalpy, turbine drive and nozzle performance.

The program performs synthetic regressions for branch pressure
equality, exact mass split, enthalpy mixing, total energy,
shaft power and nozzle flow; sensitivity to tie pressure loss;
and rejection of double-counted heat. Numeric example constants
are NOT NASA calibration values.

**Next engineering tasks:** use actual nozzle/jacket and reflector
cooling geometry, individual tie-tube supply/return cross sections,
realistic two-phase friction and acceleration constitutive laws;
advance thermal wall dynamics, couple nuclear deposition from
OpenMC, and implement full NASA-derived expander turbine recombination.



## Geometry-dependent coolant hydraulic resistance (new)

Cooling legs in \`regenerative_cooling.py\` now support a
**geometry-derived Darcy pressure-loss mode** in place of a
fixed, guessed \`pressure_loss_coefficient\`.

For a parallel bank of coolant passages, enter the **total open
flow area** (sum of all passages), each individual passage's
hydraulic diameter \`D_h\`, effective flow length \`L\`,
absolute wall roughness \`epsilon\`, and dimensionless fitting-loss
coefficient \`K_minor\`. With the local hydrogen viscosity
\`mu(P,h)\` obtained from CoolProp, each cooling control volume solves:

\`\`\`text
G       = mdot_branch / total_flow_area
Re      = G * D_h / mu_mid
f_D     = 64/Re                    for Re < 2300
f_D     = Haaland(Re,epsilon/D_h)  for Re >= 2300
K_total = f_D*L/D_h + K_minor

deltaP_wall_and_minor = K_total * G*(v_in+v_out)/4
deltaP_acceleration   = G*(v_out-v_in)

P_in-P_out = deltaP_wall_and_minor + deltaP_acceleration

mdot*(h0_out-h0_in) = Q_leg
\`\`\`

This scheme retains the existing enthalpy-conservative, two-parallel
branch pressure solve and common-header enthalpy mixing, followed by
the pump/turbine/hot-core/nozzle mass and power closure. As fluid
heating, viscosity, mass flow or roughness change, the computed
Reynolds, Darcy factor, effective K and hydraulic split update.
Independent components of pressure loss, outlet Mach, and Reynolds
are included in each \`CoolingLegState\` record.

### Input format (per leg)

\`\`\`json
{
  "heat_power_w": 110000.0,
  "flow_area_m2": 0.0008,
  "pressure_loss_coefficient": 0.0,
  "length_m": 2.0,
  "hydraulic_diameter_m": 0.008,
  "roughness_m": 0.000003,
  "minor_loss_coefficient": 0.4
}
\`\`\`

Enter each of these inside the existing \`regenerative_cooling\` JSON
object, with one entry each for \`nozzle_jacket\`, \`reflector\`,
\`tie_supply\` and \`tie_return\`. The original aggregate-K
mode is still supported, with \`pressure_loss_coefficient\` set
and \`length_m\`/\`hydraulic_diameter_m\` omitted. **Combining an
aggregate legacy K with geometry-based Darcy friction is prohibited**
to prevent double-counting.

Run the regression:

\`\`\`shell
python -m ntr.nerva.validate_geometry_regenerative
\`\`\`

### Physical applicability limits

- The Haaland wall-friction closure is **single-phase / supercritical
  screening physics only**. It is explicitly rejected for a
  two-phase H2 midpoint. Two-phase slip, pressure-drop multipliers,
  critical heat flux, flow instability and hot wall heat transfer
  remain unresolved. The older aggregate-K mode is a surrogate and
  does not validate two-phase hydraulics.
- Length, area, hydraulic diameter, roughness and local loss
  coefficients are still assumed inputs. Actual NERVA nozzle jacket,
  reflector and tie geometry has not been dimensioned from drawings.
- The turbulent Haaland transition at Re=2300 is a simplified
  correlation, not a validated transitional-flow friction model.
- Heat transferred from nozzle wall, tie solids and reflector is
  STILL supplied externally as total watts in each passage; no
  conjugate solid wall-to-coolant heat-transfer closure or spatial
  OpenMC power map feeds this reduced-order cycle yet.
- The existing separate turbine bleed remains; expander turbine
  exhaust recombination and turbine map/dynamics are pending.



## Reconstructed OpenMC fuel/tie heating to cooling-cycle power ledger

The opt-in `openmc_cycle_coupling.py` takes the existing normalized
mesh and element/tie NPZ power reconstructions through the strict
`read_resolved_heating` integral-conservation reader. It maps axial
fuel-matrix deposition to the hot-channel **wall power** and direct
fuel-hydrogen deposition separately to the hot-channel **direct H2
power**. It also places direct nuclear deposition into tie supply and
return coolant into their corresponding regenerative legs. The tie
solid's deposition is distributed to the two coolant legs ONLY through
explicitly declared transfer and supply/return split fractions.

- The model **does not** assign the unresolved OpenMC remainder to
  nozzle-jacket or reflector cooling. The nozzle and reflector
  wall-to-H2 heat rates are explicit **non-OpenMC-source prescribed
  boundary inputs**, not measured nuclear deposition.
- Nuclear power retained in untransferred fuel/tie solids and the
  unresolved mesh compartment remains in a separate accounting ledger.
- The aggregate reduced-order hot-core solver uses its existing user-
  supplied effective channel count/geometry, so mapped total axial
  powers do **not** establish a local 19-channel mass-flow distribution.
- Both instance reconstructions must carry axial edges matching the
  engine core axial grid; no unverified axial-bin resampling is used.
- Real transport provenance is **not independently verified** by this
  command. Run the repository transport/analysis QA before supplying
  reconstructed files. A synthetic fixture can also exercise it, but
  never qualifies as real neutronics validation.

The energy-ledger identities are

```text
Q_OpenMC = Q_routed_nuclear_to_gas + Q_solid_retained + Q_unresolved
Q_cycle = Q_routed_nuclear_to_gas + Q_prescribed_wall_to_gas
Q_regen = Q_jacket_prescribed + Q_reflector_prescribed
          + Q_tie_supply_direct + Q_tie_return_direct
          + Q_tie_solid_transferred
```

The existing shaft-closed nonlinear engine cycle is then solved with
these revised source powers. It retains its mass/momentum/energy
residual gates. The mapping is **steady thermal-source bookkeeping**,
not transient conjugate heat transfer or a claim of NASA hardware
equivalence.

To enable it with an existing `regenerative_cooling` JSON configuration
and matching `run_analysis` output:

```bash
python -m ntr.nerva.run_turbine_bleed_cycle \\
  --config build/regenerative_cooling_case.json \\
  --openmc-mesh-fields build/nerva_analysis/power/nerva_mesh_fields.npz \\
  --openmc-fuel-reconstruction build/nerva_analysis/element_channels/element_channel_power_reconstruction.npz \\
  --openmc-tie-reconstruction build/nerva_analysis/tie_instances/tie_instance_reconstruction.npz \\
  --fuel-solid-transfer-fraction 1.0 \\
  --tie-solid-transfer-fraction 0.8 \\
  --tie-solid-supply-fraction 0.5 \\
  --nozzle-wall-heat-w 110000 \\
  --reflector-wall-heat-w 90000 \\
  --output build/nerva_mapped_cycle.json
```

The fractions, wall heat rates and example file paths above are
**illustrative** and MUST be supplied from a separately justified
thermal model/engineering study; they are not extracted NERVA values.
Depending on reactor power, the example may not produce a feasible
engine cycle. The supplied `LH2Cycle.preheater_thermal_power_w` and
channel heat arrays are **replaced** by reconstructions, not added;
the output reports both original and effective parameters. A failed
CoolProp thermo/hydraulic/nozzle closure is an invalid operating point,
not evidence that a particular historical engine would fail.

Validation: `python -m ntr.nerva.validate_openmc_cycle_coupling`
checks independent source heat ledgers, routing fractions, unchanged
full-cycle solutions for equivalent input powers, failure modes and
actual NPZ reconstruction shape/energy parsing.

## Opt-in subcritical homogeneous-equilibrium two-phase H2 pressure losses

\`regenerative_cooling.py\` now implements an explicit
\`two_phase_friction_model="homogeneous-mcadams"\` option for
**geometry-derived coolant passages**. The default is \`"reject"\`
to avoid quietly applying a single-phase Darcy friction factor to
coexisting liquid/vapor hydrogen.

At **subcritical** coolant pressure, CoolProp gives the equilibrium
vapor *mass quality* \`x(P,h)\`, and saturated dynamic viscosities
\`mu_l(P)\` and \`mu_v(P)\`. The mixture correlation is the
mass-quality weighted inverse-viscosity rule attributed to McAdams:

\`\`\`text
1/mu_mix = (1-x)/mu_liquid + x/mu_vapor

Re_mix = mass_flux*D_h/mu_mix
f_D = 64/Re_mix             (laminar)
      Haaland(Re_mix,e/D_h) (turbulent)

K = f_D*L/D_h + K_minor

p_in-p_out =
    mass_flux*(v_out-v_in)
    + K*mass_flux*(v_in+v_out)/4

mdot*(h0_out-h0_in) = deposited_coolant_heat
\`\`\`

The **homogeneous-equilibrium assumption** forces vapor and liquid
to share one velocity; mixture specific volume is provided by
CoolProp's equilibrium hydrogen density (no adjustable void
fraction or slip ratio). As before, \`G=mdot/A\`; acceleration and
wall/minor losses remain separate. The flow is restricted to
subsonic screening conditions. To avoid misusing CoolProp's
single-phase sound-speed derivatives, the two-phase sound speed
is approximated by a finite isentropic density derivative:

\`\`\`text
1/a_HEM^2 = (d rho / dP) at constant entropy, using
            rho(P+deltaP,s) and rho(P-deltaP,s)
\`\`\`

This derivative permits equilibrium phase redistribution but is
not a model for metastable boiling, acoustic relaxation or shocks.
The report includes inlet/outlet vapor quality, midpoint quality,
mixture dynamic viscosity, Reynolds, effective friction K and
estimated outlet Mach. The old single-phase/supercritical solution
remains unchanged if no two-phase state occurs.

Example, for one cooling leg:

\`\`\`json
{
  "heat_power_w": 1000.0,
  "flow_area_m2": 0.003,
  "length_m": 0.8,
  "hydraulic_diameter_m": 0.007,
  "roughness_m": 0.000003,
  "minor_loss_coefficient": 0.5,
  "pressure_loss_coefficient": 0.0,
  "two_phase_friction_model": "homogeneous-mcadams"
}
\`\`\`

Test with:

\`\`\`bash
python -m ntr.nerva.validate_two_phase_cooling
\`\`\`

**Critical engineering caveats:** The McAdams model is empirical,
and the homogeneous-equilibrium approximation cannot by itself
resolve flow regimes, slip, wall dryout, critical heat flux,
cavitation, pressure-wave propagation, flashing instabilities, or
two-phase heat-transfer coefficients. It is *not NERVA-qualified*,
and its temperature and pressure domain is checked by CoolProp.
A hydrogen coolant circuit at approximately 6 MPa is **above**
H2's critical pressure, so this two-phase branch is exercised
separately in a low-pressure subcritical numerical test, not
claimed for the original high-pressure engine example.

References:
- CoolProp documentation warns that ordinary thermodynamic
  derivatives are single-phase only:
  https://coolprop.org/coolprop/HighLevelAPI.html
- McAdams homogeneous viscosity and pressure-drop models reviewed:
  https://www.sciencedirect.com/science/article/pii/S0894177708001027
