# Generic Nuclear-Airbreathing Demonstrator

This directory contains a deliberately generic nuclear-heated ramjet cycle
model suitable for educational and thermodynamic studies.

It is **not** an operational Burevestnik model and intentionally excludes:

- reactor criticality or fuel-loading calculations,
- reactor core dimensions or power-density optimization,
- inlet capture-area or vehicle sizing,
- guidance or control,
- range/endurance estimation,
- weapon payload or mission modeling.

## Cycle

```text
ambient air
    |
    v
inlet / diffuser
    |
    v
high-pressure stagnated air
    |
    v
nuclear heat addition
    |
    v
hot compressed air
    |
    v
nozzle
    |
    v
jet thrust
```

The user supplies freestream Mach, ambient temperature/pressure, and air mass
flow. The code computes diffuser stagnation conditions, required thermal power,
idealized nozzle expansion, gross thrust, ram drag, and net thrust.

Air mass flow is an explicit input rather than being derived from an inlet
geometry, preventing the model from becoming a vehicle-sizing tool.

## Run

```bash
python -m ntr.airbreathing.run_nuclear_ramjet
```
