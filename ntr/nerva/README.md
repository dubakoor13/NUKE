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
- Finite axial geometry and beryllium radial reflector.
- Eigenvalue settings and fissionable source.
- 3-D mesh tallies for flux, fission rate, and local heating.
- XY material plot definitions.
- Pure-Python geometry validation.
- GitHub CI export checks for cluster, all-fuel core, and mixed-core modes.

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

## Export the small all-fuel regression core

```bash
python -m ntr.nerva.build_model \
  --assembly core \
  --output build/nerva_core
```

Add `--run` only after an OpenMC executable and compatible nuclear-data
library are configured.

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
```

## Roadmap

1. 19-channel fuel element. **DONE**
2. Coaxial tie/support element. **DONE**
3. Six-fuel + one-tie cluster. **DONE**
4. Repeat the fuel/tie pattern into a representative full core. **DONE**
5. Add cylindrical/partial filler elements, Be reflector, and control drums.
6. Add axial/radial power extraction and normalization.
7. Couple heating to a 1-D hydrogen coolant/tie-tube thermal model.
8. Add temperature/density feedback iteration.
9. Calibrate only against published Rover/NERVA benchmark information.
