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


## Detailed OpenMC path

The airbreathing demonstrator now has a separate OpenMC fixed-source transport
path:

```text
build_openmc_airheater.py
        |
        v
OpenMC fixed-source transport
        |
        +-- 3-D mesh flux
        +-- 3-D deposited heating
        +-- absorption
        +-- fission
        +-- nu-fission
        +-- fuel energy spectrum
        +-- air energy spectrum
        +-- fuel / air / vessel regional heating
        +-- outer-boundary current / leakage diagnostic
        +-- tally standard deviations
        |
        v
postprocess_openmc_airheater.py
        |
        +-- normalize from source rate
        |        OR
        +-- normalize to a requested deposited-power scale
        |
        v
openmc_airheater_fields.npz
openmc_airheater_metadata.json
        |
        v
couple_openmc_airheater.py
        |
        v
air stagnation-temperature rise
        |
        v
generic ramjet nozzle / thrust model
```

The OpenMC model uses:

- **fixed-source mode**, not eigenvalue mode;
- a fixed, arbitrary cylindrical irradiation/heater geometry;
- a deliberately low-loading, 5 wt% enriched surrogate;
- an explicit central air working-fluid region;
- a generic Fe-Cr-Ni vessel region;
- vacuum outer boundaries;
- a Watt-spectrum isotropic neutron source constrained to fissionable material;
- optional coupled photon transport when the installed data library supports it;
- a 3-D regular mesh tally;
- cell-filtered regional transport/heating tallies;
- energy-filtered spectra;
- surface-current leakage diagnostics.

It intentionally has **no** control drums, reflector optimization, enrichment
search, criticality calculation, geometry optimization, vehicle sizing or
weapon-specific calibration.

### Export the OpenMC inputs

```bash
python -m ntr.airbreathing.build_openmc_airheater \
  --output build/openmc_airheater
```

This produces normal OpenMC XML inputs:

```text
geometry.xml
materials.xml
settings.xml
tallies.xml
plots.xml
```

### Run a real transport calculation

A compatible OpenMC nuclear-data library must be installed and
`OPENMC_CROSS_SECTIONS` must point to its `cross_sections.xml`.

```bash
python -m ntr.airbreathing.build_openmc_airheater \
  --output build/openmc_airheater \
  --run
```

For compatible libraries that include photon interaction data, coupled
neutron/photon transport can be requested with:

```bash
python -m ntr.airbreathing.build_openmc_airheater \
  --photon-transport \
  --output build/openmc_airheater_photons \
  --run
```

### Postprocess a statepoint

Normalize from a known source intensity:

```bash
python -m ntr.airbreathing.postprocess_openmc_airheater \
  build/openmc_airheater/statepoint.40.h5 \
  --source-rate-s 1.0e15 \
  --output build/openmc_airheater_post
```

or normalize the tally shape to a requested generic deposited-power scale:

```bash
python -m ntr.airbreathing.postprocess_openmc_airheater \
  build/openmc_airheater/statepoint.40.h5 \
  --target-deposited-power-mw 10 \
  --output build/openmc_airheater_post
```

The latter is only a normalization of a fixed-source tally field; it is not a
criticality or reactor-sizing calculation.

### Couple OpenMC heating to the air cycle

```bash
python -m ntr.airbreathing.couple_openmc_airheater \
  build/openmc_airheater_post/openmc_airheater_metadata.json \
  --mach 2.0 \
  --ambient-temperature-k 220 \
  --ambient-pressure-kpa 25 \
  --air-mass-flow-kg-s 10 \
  --heat-source fuel \
  --heat-transfer-efficiency 0.85 \
  --output build/openmc_airbreathing_cycle.json
```

The coupling computes the working-air temperature rise from the selected
OpenMC deposited-power field and then evaluates the same generic diffuser/nozzle
cycle used by the non-OpenMC demonstrator.
