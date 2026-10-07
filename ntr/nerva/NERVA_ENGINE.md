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
