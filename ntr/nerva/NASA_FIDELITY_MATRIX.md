# NERVA Fidelity Upgrade — NASA Heritage and Verification Map

This is a **staged scientific model**, not a claimed reconstruction of a
specific historic flight-ready nuclear reactor. Source-backed 1972 flight
engine **design targets** must not be conflated with experimental XE-Prime
**test points** or later NESS/ENABLER **conceptual designs**.

## NASA primary references

- NASA-CR-191080: NESS Vol. 1 (1993), https://ntrs.nasa.gov/citations/19930014687
- NASA-CR-191081: NESS 2.0 (1993), https://ntrs.nasa.gov/citations/19930014686
- NASA RP-1311: CEA equilibrium/nozzle model, https://www.nasa.gov/glenn/research/chemical-equilibrium-with-applications/
- NASA TP-2002-211556: NASA Glenn thermodynamic coefficients (NASA-9).
- NASA TM-107092 (1995), https://ntrs.nasa.gov/citations/19960002567
- NASA NERVA experience 1969-1972, https://ntrs.nasa.gov/citations/19920001870
- NASA MSFC thermal-hydraulics methods, https://ntrs.nasa.gov/citations/20130001802

## Historical reference data versus simulations

- **XE-Prime**: approximately 1140 MW class and 55.4 klbf design thrust.
  XE-Prime has TESTED operating data and a different expansion ratio and
  thermodynamic state from the 1972 flight engine. Do not average them.
- **1972 NERVA design**: 75 klbf; 450 psia; 2350–2500 K;
  nozzle expansion ratio 100:1; roughly 825–850 s depending cycle.
  These are DESIGN reference values, not an OpenMC output.
- **NESS/ENABLER**: follow-on systems program, not interchangeable with the
  historical 1972 graphite flight-engine configuration.

## New thermochemical research path

Run:
    python -m ntr.nerva.validate_nasa_hydrogen
    python -m ntr.nerva.run_nasa_hydrogen --at-m2 0.02 --mode equilibrium
    python -m ntr.nerva.run_nasa_hydrogen --at-m2 0.02 --mode frozen

The new nasa_hydrogen.py implements:

1. NASA-7 species polynomial cp(T), h(T), s(T) for H2 and H, using
   GRI-Mech 3.0 coefficients. **These are not NASA-9 Glenn tables.**
2. Equilibrium mass action H2 <-> 2H with mixture entropy, pressure, and
   conserved H-atom-based mass. Or frozen composition.
3. Choked mass flux via constrained isentropic maximum, and supersonic
   exit pressure from nozzle area ratio, not a user-fixed exit pressure.
4. Isentropic velocity from actual enthalpy drop rather than constant gamma.
5. Reactor-gas power balance Q = mdot * (hout - hin) with root solving and
   a strict 200–3500 K limit (gas-phase only).

The nozzle remains 1-D ideal-gas, quasi-equilibrium/frozen, with scalar
kinetic-efficiency loss. Equilibrium recombination is assumed instantaneous.
**This is CEA-style thermochemical bookkeeping, not NASA CEA parity.**

## NASA-exact acceptance gates (NOT YET SATISFIED)

| Gate | Status | Required evidence |
|---|---|---|
| Immutable NASA source extraction | Partial | Page/table/uncertainty in traceable machine-readable data |
| 1972 versus XE Prime architecture separation | Partial | Complete cycle diagrams and state station map |
| OpenMC reference neutronics | NOT VALIDATED | Data pedigree, convergence, benchmark comparisons |
| 2D/3D thermal hydraulics | NOT VALIDATED | Wall thermal profiles, pressure loss, mesh study |
| NASA Glenn/CEA thermochemistry parity | NOT VALIDATED | NASA-9 database, CEA frozen/equilibrium regression |
| Full NESS cycle and feed hydraulics | NOT IMPLEMENTED | Components, pump/turbine matching, nonlinear closure |
| Validated full-engine test points | NOT VALIDATED | Multiple XE/NRX points with error tolerances |
| Integrated uncertainty propagation | Partial | Material properties, operational variation and covariance |

The supplied --at-m2 input is NOT a historically sourced NERVA throat area.
It is explicitly independent; do not fit the throat to 75 klbf and then
report the resulting thrust as an independent historical validation.

## Safeguard against overstating fidelity

No output from this module should be labelled a NASA-exact flight-engine
prediction. Published NESS, CEA, MSFC thermo-fluids analyses, and individual
historic NERVA records form distinct verification levels.

## Independent NASA CEA thermodynamics regression

The optional module ntr/nerva/compare_nasa_cea.py uses NASA's maintained
CEA package (https://github.com/nasa/cea) and the NASA Glenn thermo database
to independently compare hot-hydrogen properties against the in-house
two-species NASA-7 model.

Run on Python >= 3.11:

    python -m pip install cea numpy
    python -m ntr.nerva.compare_nasa_cea \
        --output build/nerva_nasa_cea_comparison.json

The official NASA CEA Python EqSolver TP interface is used with pressure
in **bar** and temperature in **K**. The restricted species set is H2 and H;
this is essential for a like-for-like comparison with the reduced solver.
Exported diagnostics contain equilibrium dissociation fraction, density,
and frozen heat capacity, together with relative errors.

The comparison does NOT turn our custom NASA-7 solver into official NASA CEA,
nor does it validate reactor neutronics or historical flight performance.
For NASA-exact thermo comparison at more demanding tolerances, replace
the GRI-Mech NASA-7 data by NASA Glenn NASA-9 and extend the species set.


## Direct NASA CEA rocket/nozzle solver (new)

The optional \`ntr/nerva/nasa_cea_rocket.py\` calls the **official NASA CEA**
\`RocketSolver\` with a chamber at prescribed pressure and temperature.
Unlike \`nasa_hydrogen.py\`, this is not a custom thermochemistry integrator.
It uses the NASA Glenn CEA database and supports:

- pure H2 feed with H2/H species in chamber and nozzle;
- IAC equilibrium expansion, and frozen composition from the throat
  (\`n_frz=2\`, *not* chamber-frozen);
- fixed Ae/At, CEA choked mass flux and c*, exit state and Mach;
- explicit SI pressure conversion (CEA API takes bar);
- vacuum thrust, delivered thrust, mass flow and Isp with separate
  nozzle-kinetic efficiency and ambient pressure assumptions.

Usage:

\`\`\`shell
python -m pip install cea==3.3.4 numpy
python -m ntr.nerva.nasa_cea_rocket --pc-mpa 3.10264 --tc-k 2425 \
  --at-m2 0.02 --area-ratio 100 --mode equilibrium
python -m ntr.nerva.validate_nasa_cea_rocket
\`\`\`

\`validate_nasa_cea_rocket\` compares public NERVA design reference conditions
with the independent CEA solver and NUKE's reduced NASA-7 approximation.
**Throat area 0.02 m² is an arbitrary comparison input**, not an extracted
historical part dimension. Neither a match to thrust nor a complete NASA
NERVA reproduction is claimed from these idealized tests.

This is still IAC/1-D and does not replace NESS 2.0, feed machinery, radiator,
reactor neutronics, or experimentally constrained transport/heat transfer.
