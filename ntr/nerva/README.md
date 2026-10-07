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
- NPZ, CSV, and JSON export of 3-D power-density, fission-rate, and flux fields.
- XY material plot definitions.
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
8. Couple heating to a 1-D hydrogen coolant/tie-tube thermal model.
9. Add temperature/density feedback iteration.
10. Calibrate only against published Rover/NERVA benchmark information.
