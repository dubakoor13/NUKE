# NERVA-derived OpenMC demonstrator

This directory builds a staged OpenMC representation of a NERVA-derived nuclear
thermal rocket reactor. The geometry follows public NASA heritage dimensions,
while the fuel loading remains a deliberately conservative, non-calibrated
low-enrichment surrogate.

## Implemented

- 1.905 cm flat-to-flat hexagonal fuel element.
- 132 cm active length.
- 19 axial hydrogen coolant channels.
- 0.2565 cm coolant bore diameter.
- 0.40894 cm bore pitch.
- 50 micrometer ZrC channel and exterior coating.
- Explicit coaxial NERVA-derived tie/support element:
  - hydrogen supply passage;
  - inner Inconel tube;
  - ZrH moderator sleeve;
  - hydrogen return passage;
  - outer Inconel tube;
  - ZrC sleeve;
  - graphite filler;
  - explicit cold-condition gaps.
- Seven-position cluster: one tie tube surrounded by six fuel elements.
- Parameterized small all-fuel core retained for regression comparison.
- SNRE-style mixed fuel/tie full-core lattice using a three-color triangular-lattice pattern.
- Homogenized graphite peripheral filler that rounds the lattice into a cylindrical core envelope.
- Beryllium radial reflector with 12 embedded rotatable control drums.
- Each drum uses a parameterized Be body with a natural-B4C absorber half-shell surrogate.
- Aluminum-alloy pressure-vessel surrogate around the reflector.
- Finite axial geometry and vacuum outer boundaries.
- Eigenvalue settings and fissionable source.
- 3-D mesh tallies for flux, fission rate, and local heating.
- Statepoint postprocessor that normalizes raw tallies to requested reactor thermal power.
- Separate fuel-material and tie-solid mesh heating tallies for thermal-hydraulic coupling.
- NPZ, CSV, and JSON export of 3-D power-density, fuel-power, fission-rate, and flux fields.
- First-pass 1-D hydrogen fuel-channel solver with energy balance, heat-transfer coefficient, wall-temperature estimate, Reynolds number, and Darcy pressure loss.
- Explicit counterflow tie-tube thermal model with central H2 supply and annular H2 return paths.
- First-pass ZrC + fuel-ligament conduction estimate for fuel-surface and local peak-fuel temperature.
- XY material plot definitions.
- Result plotting CLI for power-density maps, axial power, fuel-channel thermal profiles, and tie-tube counterflow profiles.
- Pure-Python geometry validation.
- GitHub CI export checks for cluster, all-fuel core, mixed-core, and full reactor modes.

## Important model limitation

The fuel matrix, Inconel, ZrH, coolant state, and reflector properties are
engineering surrogates rather than a validated reconstruction of a particular
Rover/NERVA test article. The demonstration enrichment is constrained below
20 wt% U-235. Do not interpret a future `k_eff`, temperature, or performance
result as a historical NERVA benchmark until the model is explicitly calibrated
against an appropriate public experiment.

## Validate geometry

From the repository root:

```bash
python -m ntr.nerva.validate_layout
```

## Export the seven-position fuel/tie cluster

```bash
python -m ntr.nerva.build_model \
  --assembly cluster \
  --output build/nerva_cluster
```

This produces OpenMC `geometry.xml`, `materials.xml`, `settings.xml`,
`tallies.xml`, and `plots.xml`.

## Export the mixed fuel/tie core

```bash
python -m ntr.nerva.build_model \
  --assembly mixed-core \
  --rings 8 \
  --output build/nerva_mixed_core
```

The mixed-core pattern assigns one of three triangular-lattice colors to tie
tubes. This gives every interior tie tube six fuel neighbors and every interior
fuel element three tie-tube neighbors.

## Export the full reactor-periphery assembly

```bash
python -m ntr.nerva.build_model \
  --assembly reactor \
  --rings 8 \
  --drum-angle 0 \
  --output build/nerva_reactor
```

The control-drum angle is parameterized. By convention in this demonstrator,
`0 deg` places the B4C-bearing half-shell toward the core and `180 deg`
rotates it outward. This is a geometry convention only; it is not yet a
validated drum-worth calibration.

The periphery is intentionally staged. The partial filler is presently a
homogenized graphite region outside the finite lattice but inside the
cylindrical core envelope. Explicit machined partial-hex filler pieces can be
substituted later without changing the core lattice API.

## Export the small all-fuel regression core

```bash
python -m ntr.nerva.build_model \
  --assembly core \
  --output build/nerva_core
```

Add `--run` only after an OpenMC executable and compatible nuclear-data
library are configured.




## Instance-resolved OpenMC and channel-level thermal coupling

The NERVA workflow now extracts repeated-cell OpenMC tallies rather than
assuming that all repeated fuel elements receive the same integrated power.

### Exact OpenMC repeated-cell outputs

The statepoint can now contain:

```text
nerva_fuel_element_instances
nerva_hydrogen_channel_01_instances
...
nerva_hydrogen_channel_19_instances
nerva_tie_instance_*
```

These use OpenMC `DistribcellFilter` objects. The postprocessor

```bash
python -m ntr.nerva.postprocess_instance_tallies \
  statepoint.80.h5 \
  --power-metadata build/nerva_power/metadata.json \
  --output build/nerva_instances
```

writes integrated OpenMC scores for each repeated fuel-matrix instance, each
coolant-channel instance, and each explicit tie-tube solid component.

It preserves Monte Carlo standard deviations for the repeated-cell heating
scores and maps the 19 coolant-channel instances back to their parent fuel
element using OpenMC geometry-path metadata when available. A clearly reported
instance-index fallback is used if path metadata is unavailable.

### Direct nuclear heating of hydrogen

OpenMC now separately tallies direct deposited nuclear heat in:

- the 19 fuel-element hydrogen passages;
- the central tie-tube hydrogen supply passage;
- the annular tie-tube hydrogen return passage.

The thermal solvers treat these energy paths differently from solid-to-fluid
heat transfer:

```text
fuel/tie solid heating -> wall heat flux -> convection -> bulk H2
direct H2 nuclear heat ---------------------------> bulk H2
```

Direct H2 heating therefore raises fluid enthalpy but does not artificially
increase wall heat flux or the solid-conduction temperature rise.

### Element x 19-channel reconstruction

Run:

```bash
python -m ntr.nerva.solve_element_channels \
  build/nerva_power/nerva_mesh_fields.npz \
  build/nerva_instances/instance_power_fractions.npz \
  --fuel-mass-flow-kg-s 2.0 \
  --inlet-temperature-k 500 \
  --inlet-pressure-mpa 8 \
  --hydrogen-model coolprop \
  --output build/nerva_element_channels
```

The solver evaluates every repeated fuel element and all 19 channels.

The current reconstruction deliberately distinguishes measured quantities from
assumptions:

- **OpenMC measured:** integrated fuel-element heating fraction;
- **OpenMC measured:** integrated direct-H2 heating of each channel instance;
- **OpenMC measured:** fuel-element-specific axial fuel-matrix heating from a
  Distribcell x axial-mesh tally;
- **OpenMC measured:** channel-instance-specific axial direct-H2 heating from
  Distribcell x axial-mesh tallies;
- **legacy fallback only:** the global axial shapes are used separably when an
  older statepoint does not contain the direct instance-axial tallies;
- **OpenMC measured:** integrated and axial solid heating in 19 same-material
  nearest-channel Voronoi fuel sectors;
- **current thermal assignment:** each fuel sector's solid heat is assigned to
  its associated coolant channel;
- **legacy fallback only:** older statepoints without sector tallies divide
  solid wall power equally among the 19 channels.

For new statepoints built by the current model, the element axial shape,
channel-associated solid-heating shape, and direct-H2 channel axial shape are
Monte Carlo tally outputs. The remaining approximation is the thermal mapping
of each Voronoi sector's deposited heat to its associated coolant channel.

Outputs include the complete element/channel table, reconstructed axial power
arrays, and a full axial profile for the hottest reconstructed channel.



### Fuel-sector scoring partition

The fuel matrix is now split into 19 scoring cells using nearest-channel
Voronoi boundaries. Every sector uses the same fuel material and every internal
Voronoi plane is transmission-only, so this is a scoring-resolution change
rather than a new material interface.

For channel `i`, OpenMC now produces:

```text
nerva_fuel_sector_ii_instances
nerva_fuel_sector_ii_axial_instances
```

The sector-integrated and sector-axial `heating-local` tallies are aligned to
the repeated fuel-element instance and stored as:

```text
fuel_channel_sector_heating_w
fuel_channel_sector_heating_std_w
fuel_channel_sector_axial_heating_w
fuel_channel_sector_axial_heating_std_w
```

The element/channel thermal solver uses these arrays directly for wall-power
assignment. It still renormalizes the complete sector field to the global
OpenMC fuel-solid power total so energy closure is exact.

`validate_fuel_sectors.py` independently reconstructs the pre-partition fuel
region, samples it densely, and checks that every sampled fuel point belongs
to exactly one scoring sector. It also verifies that all 171 shared Voronoi
planes are transmission boundaries.


### Per-tie OpenMC thermal reconstruction

Current statepoints also carry direct repeated-cell axial tallies for every
tie-tube solid component plus the central supply and annular return hydrogen
passages.

Run:

```bash
python -m ntr.nerva.solve_tie_instances \
  build/nerva_power/nerva_mesh_fields.npz \
  build/nerva_instances/instance_power_fractions.npz \
  --tie-mass-flow-kg-s 0.5 \
  --inlet-temperature-k 500 \
  --inlet-pressure-mpa 8 \
  --hydrogen-model coolprop \
  --output build/nerva_tie_instances
```

For each tie tube the solver uses:

- the summed OpenMC axial heating of the explicit tie-tube solid components;
- the OpenMC axial direct nuclear heating of the central supply H2 passage;
- the OpenMC axial direct nuclear heating of the annular return H2 passage.

The instance fields are renormalized to the corresponding global OpenMC mesh
totals so that the per-tie reconstruction preserves total solid and direct-H2
power exactly. Each tie is then solved independently through the existing
counterflow model.

Tie-tube hydrogen mass flow is divided equally among repeated tie instances by
default. With `--balance-flow` on the per-tie solver, or
`--balance-parallel-flow` on the orchestrator, the same fixed total flow can
instead be redistributed iteratively toward a common pressure drop. The older
representative tie-tube solver is retained for backward compatibility and
cross-checking.

Outputs include:

```text
tie_instance_summary.csv
tie_instance_summary.json
tie_instance_reconstruction.npz
hottest_tie_supply_profile.csv
hottest_tie_return_profile.csv
```

`plot_tie_instances.py` generates the per-tie power, outlet-temperature,
wall-temperature, and hottest-tie axial plots.




### Equivalent-annulus fuel-sector conduction

The legacy solid-temperature screening model uses a minimum-ligament slab.
A second opt-in model now uses the deterministic area of the actual
nearest-channel Voronoi fuel sector:

```bash
python -m ntr.nerva.solve_element_channels \
  build/nerva_power/nerva_mesh_fields.npz \
  build/nerva_instances/instance_power_fractions.npz \
  --fuel-mass-flow-kg-s 2.0 \
  --inlet-temperature-k 500 \
  --inlet-pressure-mpa 8 \
  --solid-conduction-model equivalent-annulus-sector \
  --output build/nerva_element_channels_annulus
```

For each of the 19 channel sectors, the deterministic geometry layer clips the
inner fuel hex by the same nearest-channel half-planes used by the OpenMC
scoring partition. The local coated channel area is removed to obtain the fuel
cross-sectional area. An equivalent outer radius is then chosen so that

```text
pi * (R_eq^2 - r_coated^2) = A_fuel,sector
```

The ZrC coating temperature rise uses the cylindrical resistance, while the
fuel rise assumes uniform volumetric heating in an annulus with an adiabatic
outer boundary. The axial heat input is the direct OpenMC fuel-sector wall
power for that channel.

This is a higher-fidelity reduced-order solid model, not a 2-D/3-D finite
element conduction solution. The legacy `ligament-slab` model remains the
default so existing runs do not silently change. The orchestrators expose the
same choice with `--solid-conduction-model`.

`validate_fuel_sector_geometry.py` checks exact hex/sector area closure and
the equivalent-radius reconstruction. `validate_fuel_conduction.py` checks
the annulus zero-power limit and positive temperature-rise behavior.


### Optional parallel-flow balancing

By default, the resolved fuel coolant channels and tie-tube branches still use
equal shares of their user-supplied total mass flows. An opt-in hydraulic
network iteration can instead redistribute those fixed totals toward a common
parallel-branch pressure drop:

```bash
python -m ntr.nerva.run_analysis \
  statepoint.80.h5 \
  --power-mw 100 \
  --rings 5 \
  --fuel-mass-flow-kg-s 2.0 \
  --tie-mass-flow-kg-s 0.40 \
  --inlet-temperature-k 500 \
  --inlet-pressure-mpa 8 \
  --hydrogen-model coolprop \
  --balance-parallel-flow \
  --flow-balance-max-iterations 6 \
  --flow-balance-tolerance 1e-3 \
  --flow-balance-relaxation 0.5 \
  --output-root build/nerva_analysis_balanced
```

Each iteration reruns the complete branch thermal-hydraulic solve, forms an
iteration-local effective resistance

```text
R_i = DeltaP_i / mdot_i^2
```

and updates the branch target flows from the common-pressure-drop relation
`mdot_i proportional to 1/sqrt(R_i)`. Relaxation is applied before the branch
flows are renormalized to preserve the requested total mass flow exactly.

The output summaries record whether balancing was enabled, whether the flow
update met the requested tolerance, iteration count, minimum/maximum branch
flow, total-flow closure, and the final pressure-drop spread.

This is a screening parallel-network balance. It does **not** yet represent
explicit inlet/outlet manifolds, plenum pressure fields, pump/turbopump maps,
branch minor-loss coefficients, or inter-channel crossflow.


### Stochastic OpenMC volumes

OpenMC stochastic-volume calculations can be prepared or executed with:

```bash
python -m ntr.nerva.calculate_volumes \
  --cross-sections /path/to/cross_sections.xml \
  --assembly reactor \
  --rings 5 \
  --samples 1000000 \
  --output build/nerva_volumes \
  --run
```

The output estimates total material volumes and total repeated-cell volumes.
When `volume_results.json` is supplied to
`postprocess_instance_tallies`, cell-integrated tracklength flux tallies are
converted to approximate per-instance `cm^-2 s^-1` values using the average
stochastic volume of the repeated cell.

### Statistical stopping

A relative-error trigger can be enabled without changing the reactor model:

```bash
python -m ntr.nerva.run_openmc_transport \
  --cross-sections /path/to/cross_sections.xml \
  --target-rel-error 0.05 \
  --trigger-max-batches 500 \
  --trigger-batch-interval 5 \
  --output build/nerva_transport
```

This changes Monte Carlo stopping behavior only.

### Material x energy transport

In addition to fuel/H2 spectra, OpenMC now produces a material-by-energy
diagnostic matrix for flux, absorption, fission, nu-fission, and local heating.
When coupled photon transport is enabled, a particle-filtered heating tally
also separates neutron and photon contributions.

The diagnostics package adds:

```text
material_energy_transport.csv
particle_heating.csv
```

### Nuclear-data temperature audit

Every reproducible transport run now performs a read-only audit of the
temperatures available in each required neutron HDF5 file and records the
nearest data temperature for materials with an explicit model temperature.

```bash
python -m ntr.nerva.audit_nuclear_data_temperatures \
  --cross-sections /path/to/cross_sections.xml
```

The audit does not change OpenMC interpolation, material temperature,
composition, or thermal-scattering treatment.

### Visualization and numerical QA

Normalized OpenMC fields can be exported directly for ParaView:

```bash
python -m ntr.nerva.export_vtk \
  build/nerva_power/nerva_mesh_fields.npz \
  --output build/nerva_fields.vtk
```

Element/channel heatmaps can be generated with
`plot_element_channels.py`.

Two completed analysis directories can be compared with
`compare_runs.py`, while multiple runs made with different Monte Carlo
sampling/resolution settings can be summarized with
`summarize_convergence.py`. These tools report numerical differences only and
do not select or optimize reactor configurations.

### Screening uncertainty propagation

`propagate_uncertainty.py` reruns the hottest reconstructed channel with the
integrated OpenMC fuel-element and direct-H2 heating shifted by ±1 standard
deviation. This is explicitly a screening sensitivity envelope; it does not
include tally covariance, nuclear-data covariance, geometry/material-property
uncertainty, or flow-distribution uncertainty.


## Reproducible real-data OpenMC run

The repository now has a nuclear-data preflight and provenance layer for real
OpenMC transport runs. It does not download or modify nuclear data; it validates
the library you explicitly provide.

### 1. Check the nuclear-data library

```bash
python -m ntr.nerva.check_nuclear_data \
  --cross-sections /path/to/cross_sections.xml \
  --output build/nerva_nuclear_data_report.json
```

The report records:

- SHA-256 of `cross_sections.xml`;
- neutron/photon/thermal library inventory;
- nuclides required by the current NERVA material definitions;
- missing neutron nuclides;
- missing referenced HDF5 files;
- photon-data availability when photon transport is requested.

No composition or enrichment is changed during this check.

### 2. Export or run one explicit transport configuration

Export/preflight only:

```bash
python -m ntr.nerva.run_openmc_transport \
  --cross-sections /path/to/cross_sections.xml \
  --assembly reactor \
  --rings 5 \
  --output build/nerva_transport \
  --export-only
```

Run OpenMC:

```bash
python -m ntr.nerva.run_openmc_transport \
  --cross-sections /path/to/cross_sections.xml \
  --assembly reactor \
  --rings 5 \
  --particles 4000 \
  --batches 80 \
  --inactive 20 \
  --output build/nerva_transport
```

The transport directory contains the exported XML, the nuclear-data preflight
report and `transport_provenance.json`. The provenance file records:

- Git commit when available;
- Python/OpenMC versions;
- platform;
- complete run settings;
- geometry summary;
- cross-sections path and SHA-256;
- SHA-256 of the exported XML inputs;
- completed statepoint path/checksum after a run.

The runner executes one supplied configuration only. It contains no automatic
enrichment, geometry or control-drum search.

### 3. Run the complete real-data pipeline

```bash
python -m ntr.nerva.run_full_openmc_analysis \
  --cross-sections /path/to/cross_sections.xml \
  --assembly reactor \
  --rings 5 \
  --power-mw 100 \
  --fuel-mass-flow-kg-s 2.0 \
  --tie-mass-flow-kg-s 0.40 \
  --inlet-temperature-k 500 \
  --inlet-pressure-mpa 8 \
  --hydrogen-model coolprop \
  --output-root build/nerva_full_run
```

This performs:

```text
nuclear-data preflight
        ↓
OpenMC input export
        ↓
OpenMC transport
        ↓
statepoint + provenance
        ↓
power normalization
        ↓
advanced OpenMC diagnostics
        ↓
fuel-channel thermal analysis
        ↓
tie-tube thermal analysis
        ↓
analysis-package QA
        ↓
plots
```

The final QA gate is:

```bash
python -m ntr.nerva.validate_analysis_outputs \
  build/nerva_full_run/analysis
```

It checks internal consistency only:

- requested vs normalized power closure;
- OpenMC source-rate consistency;
- fuel/tie deposited-power cross-checks;
- material-power and spectrum-fraction closure;
- finite Monte Carlo uncertainty statistics;
- finite reported `k_eff` if present;
- fuel/tie temperature and pressure trends;
- shape/finite-value integrity of every normalized 3-D field.

A PASS does not constitute historical NERVA criticality or performance
validation.


## Full OpenMC diagnostics path

The NERVA-derived model now uses OpenMC for substantially more than a single
3-D fission/heating field. The default eigenvalue model remains the same
non-calibrated low-enrichment surrogate, but the statepoint now carries a much
richer diagnostic set.

### Transport/settings diagnostics

- eigenvalue transport with the existing fissionable source;
- source Shannon-entropy mesh for convergence diagnostics;
- optional coupled neutron-photon transport via `--photon-transport`;
- configurable logarithmic neutron-energy group count;
- configurable high-resolution axial tally mesh;
- vacuum-boundary current/leakage tally.

Example:

```bash
python -m ntr.nerva.build_model \
  --assembly reactor \
  --rings 5 \
  --diagnostic-energy-groups 120 \
  --axial-mesh-bins 132 \
  --photon-transport \
  --output build/nerva_reactor_detailed
```

Photon transport requires a nuclear-data library with compatible photon
interaction data.

### 3-D transport fields

The `nerva_3d_neutronics` mesh now scores:

```text
flux
absorption
fission
nu-fission
heating
heating-local
```

The normalized engineering NPZ/CSV therefore includes:

```text
power_density_w_cm3
power_density_std_w_cm3
total_heating_power_density_w_cm3
flux_cm2_s
absorption_rate_cm3_s
fission_rate_cm3_s
nu_fission_rate_cm3_s
fuel_power_density_w_cm3
tie_power_density_w_cm3
```

The standard postprocessor also reports maximum/mean-active mesh power peaking
and Monte Carlo heating relative-error statistics.

### Material-resolved transport

A separate material tally reports flux, absorption, fission, nu-fission,
`heating`, and `heating-local` for:

```text
fuel
hydrogen
ZrC
beryllium reflector
graphite
ZrH
Inconel
B4C
aluminum vessel
```

This makes it possible to quantify where deposited nuclear heat is going
without inferring the split from geometry alone.

### Energy spectra

The statepoint now carries independent energy-filtered spectra for:

- the conceptual fuel material: flux, absorption, fission and nu-fission;
- the hydrogen coolant: flux, absorption and local heating.

The advanced postprocessor integrates these spectra into thermal
(<0.625 eV), epithermal and fast (>100 keV) diagnostic fractions while also
retaining the complete energy-bin data.

### Fine axial transport

A separate `1 x 1 x N` OpenMC mesh provides a fine axial history of:

- deposited power;
- flux;
- fission rate;
- nu-fission production.

This is independent of the coarser 3-D engineering mesh and is intended for
axial power-shape and peaking diagnostics.

### Leakage/current

All vacuum boundaries discovered from the geometry are included in a
`SurfaceFilter` current tally. The advanced postprocessor exports the signed
boundary currents and their Monte Carlo standard deviations.

### Advanced statepoint package

After normalizing a real statepoint:

```bash
python -m ntr.nerva.postprocess_statepoint \
  statepoint.80.h5 \
  --power-mw 100 \
  --output build/nerva_power
```

run:

```bash
python -m ntr.nerva.postprocess_openmc_diagnostics \
  statepoint.80.h5 \
  --power-metadata build/nerva_power/metadata.json \
  --output build/nerva_openmc_diagnostics
```

Outputs:

```text
build/nerva_openmc_diagnostics/
├── openmc_diagnostics.json
├── openmc_diagnostics.npz
├── material_transport.csv
├── spectra.csv
├── axial_profile.csv
└── leakage_current.csv
```

The JSON report includes:

- reported `k_eff` and its uncertainty;
- source-entropy history when present in the statepoint;
- material power fractions;
- 3-D and axial power peaking;
- mesh-heating uncertainty statistics;
- fuel and hydrogen spectral fractions;
- vacuum-boundary currents.

The reported `k_eff` is diagnostic only. There is no automatic enrichment,
geometry, or control-drum tuning loop.

### OpenMC diagnostic plots

```bash
python -m ntr.nerva.plot_openmc_diagnostics \
  --diagnostics build/nerva_openmc_diagnostics/openmc_diagnostics.npz \
  --material-csv build/nerva_openmc_diagnostics/material_transport.csv \
  --output build/nerva_openmc_plots
```

This generates separate plots for:

- axial deposited power;
- fuel neutron spectrum;
- hydrogen-region neutron spectrum;
- material heat deposition;
- 3-D midplane local heating;
- 3-D midplane Monte Carlo relative uncertainty.

The one-command `run_analysis` workflow now runs the diagnostics
postprocessor and these plots automatically unless `--no-plots` is used.

### Deliberate fidelity boundary

The expanded diagnostics do **not** change the underlying surrogate reactor
into a validated historical NERVA reconstruction. In particular, the workflow
does not automatically optimize enrichment, calibrate control-drum worth, or
iterate temperatures/densities back into criticality. Those require a separate
published-benchmark validation step before they should be interpreted
physically.


## Normalize a completed OpenMC statepoint

After a transport run has produced a statepoint file:

```bash
python -m ntr.nerva.postprocess_statepoint \
  statepoint.80.h5 \
  --power-mw 100 \
  --output build/nerva_power
```

The postprocessor uses the mesh `heating-local` score to derive the source-rate
normalization that makes the summed voxel heating equal the requested total
thermal power. It then applies the same source-rate scale to fission rate and
tracklength flux.

Outputs:

```text
build/nerva_power/
├── nerva_mesh_fields.npz
├── nerva_mesh_fields.csv
└── metadata.json
```

Field units are W/cm3, reactions/cm3/s, and particles/cm2/s respectively.
The NPZ file also contains `fuel_power_density_w_cm3` and
`tie_power_density_w_cm3`. The first comes from a fuel-material-filtered
OpenMC heating tally. The second sums heating in the explicit tie-tube solid
cells (Inconel tubes, ZrH, ZrC sleeve/coating, and graphite filler). Both are
normalized with the same source rate as the whole-reactor heating field.

## Solve the representative hydrogen fuel channel

After statepoint postprocessing:

```bash
python -m ntr.nerva.solve_thermal \
  build/nerva_power/nerva_mesh_fields.npz \
  --rings 5 \
  --fuel-mass-flow-kg-s 1.0 \
  --inlet-temperature-k 500 \
  --inlet-pressure-mpa 8 \
  --output build/nerva_thermal
```

The fuel thermal model distributes fuel-deposited power equally over all 19
coolant holes in every fuel element, then solves one representative channel.
The default hydrogen model remains the constant-property ideal-gas surrogate.
An optional state-dependent backend is now available through CoolProp:

```bash
python -m pip install CoolProp
python -m ntr.nerva.solve_thermal \
  build/nerva_power/nerva_mesh_fields.npz \
  --rings 5 \
  --fuel-mass-flow-kg-s 1.0 \
  --inlet-temperature-k 500 \
  --inlet-pressure-mpa 8 \
  --hydrogen-model coolprop \
  --output build/nerva_thermal_real_h2
```

With the CoolProp backend, density, heat capacity, viscosity, conductivity, and
Prandtl number are re-evaluated from the local temperature and pressure during
the axial march. Dittus-Boelter heat transfer and Darcy-Weisbach pressure loss
remain the governing correlations.

## Solve the tie-tube counterflow path

The tie-tube model uses the separate OpenMC tie-solid heating field:

```bash
python -m ntr.nerva.solve_tie_tube \
  build/nerva_power/nerva_mesh_fields.npz \
  --rings 5 \
  --tie-mass-flow-kg-s 0.20 \
  --inlet-temperature-k 500 \
  --inlet-pressure-mpa 8 \
  --hydrogen-model constant \
  --output build/nerva_tie_thermal
```

For each representative tie tube, hydrogen flows down the central inner tube
and returns through the annulus between the ZrH moderator and outer Inconel
tube. The default first-pass split of tie-solid heat between supply and return
is proportional to their wetted perimeters; `--supply-heat-fraction` can
override it explicitly. The supply and return paths preserve separate
temperature, pressure, Reynolds-number, wall-temperature, heat-flux and
heat-transfer-coefficient histories.

The fuel-channel output now also estimates solid temperatures by adding
1-D conduction through the channel ZrC coating and a symmetry-slab rise across
half of the minimum fuel ligament. Fuel and ZrC conductivities are explicit CLI
parameters so this screening model does not hide material assumptions.

The current detailed path already includes optional real-fluid H2 properties,
OpenMC-resolved local fuel/tie power, and optional parallel-flow redistribution.
2-D/3-D fuel conduction, radiation, explicit tie-tube turn/minor losses,
manifold/plenum hydraulics, and temperature-density feedback remain
next-fidelity layers.

## Run the complete post-OpenMC analysis in one command

After OpenMC has produced a statepoint, the orchestrator runs normalization,
fuel-channel thermal analysis, tie-tube counterflow, and plotting:

```bash
python -m ntr.nerva.run_analysis \
  statepoint.80.h5 \
  --power-mw 100 \
  --rings 5 \
  --fuel-mass-flow-kg-s 2.0 \
  --tie-mass-flow-kg-s 0.40 \
  --inlet-temperature-k 500 \
  --inlet-pressure-mpa 8 \
  --hydrogen-model constant \
  --output-root build/nerva_analysis
```

Use `--hydrogen-model coolprop` after installing CoolProp, or `--no-plots`
when matplotlib is not installed.

## Historical NERVA reference presets

The repository now carries source-tagged NASA reference presets in
`ntr/nerva/historical_presets.py`. They are system-level comparison data and
do not contain fissile loading or an exact criticality recipe.

Implemented references:

- **XE-Prime tested prototype** — 1140 MW, 55.4 klbf (~246.4 kN), 710 s net
  Isp, 10:1 nozzle area ratio, ~2281 K chamber temperature, ~3.90 MPa chamber
  pressure, 70.2 lb/s (~31.84 kg/s) H2 core flow, 1584 fuel elements, 19
  channels per element, 35 in core diameter and 52 in active length.
- **1972 75-klbf NERVA design reference** — 75 klbf (~333.6 kN), 2350–2500 K
  chamber temperature, 450 psia (~3.10 MPa), 100:1 nozzle expansion ratio and
  825–850 s specific impulse.
- **75-klbf graphite NERVA derivative** — 2500 K, 500 psia, 200:1 and 875 s.
- **NESS / R-1 baseline reactor** — ~1500 MW, nominal 38 in diameter x 52 in
  long core, 19 mm / 0.75 in NERVA-type hex fuel elements with 19 channels,
  tie-tube heating 3–7% and reflector heating 1–2%.

NASA sources are stored in each preset and displayed directly by the dashboard.

## Web dashboard: thrust, Isp, temperatures and run values

A Streamlit dashboard is included at the repository root:

```bash
python -m pip install -r requirements-web.txt
streamlit run streamlit_app.py
```

The dashboard has four modes:

- **Historical NERVA reference** — select XE-Prime, 1972 75-klbf NERVA,
  graphite derivative, or NESS/R-1 and see the published NASA values.
- **Synthetic demo** — starts immediately with deterministic CI values. These
  values are explicitly labeled synthetic and are not an OpenMC transport run.
- **Auto-load analysis directory** — reads a local `build/nerva_analysis`
  result tree directly.
- **Upload run outputs** — upload `metadata.json`,
  `thermal_summary.json`, and `tie_thermal_summary.json` from an actual
  analysis directory.

Displayed quantities include reactor thermal power, fuel/tie heating split,
fuel and tie-tube outlet state, combined hydrogen mass flow and chamber state,
estimated thrust, Isp, effective exhaust velocity, exit Mach, nozzle expansion
ratio, equivalent exit diameter, and fuel/tie temperature screening values.

The engine layer uses the loaded thermal outlet state and a choked ideal-gas
nozzle estimate. Nozzle gamma, efficiency, exit pressure, and ambient pressure
are visible dashboard assumptions. This keeps the displayed thrust/Isp
traceable rather than presenting them as direct OpenMC outputs.

The built-in synthetic example is intentionally low-power (1 MW) and produces
only a low-temperature hydrogen outlet. Its thrust/Isp are UI demonstration
values, not representative NERVA performance.

## Plot postprocessed results

The plotting CLI accepts any subset of the available result products:

```bash
python -m ntr.nerva.plot_results \
  --power-fields build/nerva_power/nerva_mesh_fields.npz \
  --fuel-profile build/nerva_thermal/fuel_channel_profile.csv \
  --tie-supply-profile build/nerva_tie_thermal/tie_supply_profile.csv \
  --tie-return-profile build/nerva_tie_thermal/tie_return_profile.csv \
  --output build/nerva_plots
```

It produces mid-plane deposited-power maps, an axial power comparison, fuel
temperature/pressure plots, and tie-tube supply/return temperature/pressure
plots. Matplotlib is loaded only when plots are actually requested.

## Current hierarchy

```text
19-channel fuel element
        |
        +---- six fuel elements
        |          |
        |          v
        +---- 1 coaxial tie tube
                   |
                   v
          seven-position cluster
                   |
                   v
        repeated 2:1 FE/TT lattice
                   |
                   v
       representative mixed core
                   |
                   v
      homogenized peripheral filler
                   |
                   v
      Be reflector + 12 control drums
                   |
                   v
       Al-alloy pressure vessel
```

## Roadmap

1. 19-channel fuel element. **DONE**
2. Coaxial tie/support element. **DONE**
3. Six-fuel + one-tie cluster. **DONE**
4. Repeat the fuel/tie pattern into a representative full core. **DONE**
5. Add cylindrical filler, Be reflector, control drums, and pressure vessel. **DONE (homogenized filler / surrogate drum dimensions)**
6. Replace homogenized filler with explicit partial-hex filler pieces.
7. Add axial/radial power extraction and normalization. **DONE**
8. Couple fuel heating to a 1-D hydrogen fuel-channel thermal model. **DONE (constant-property surrogate)**
9. Add explicit tie-tube supply/return thermal-hydraulic paths. **DONE (constant-property surrogate)**
10. Add first-pass channel-coating/fuel-ligament conduction. **DONE (screening estimate)**
11. Add optional state-dependent / real-fluid H2 properties. **DONE (CoolProp backend; optional dependency)**
12. Add local fuel-element/tie-tube power peaking.
13. Add 2-D/3-D fuel conduction and temperature/density feedback iteration.
14. Calibrate only against published Rover/NERVA benchmark information.
