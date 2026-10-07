# NERVA-derived OpenMC demonstrator

This directory is the first NTR model built on the OpenMC source copied into `NUKE`. It starts with the historically recognizable NERVA fuel-element geometry and deliberately keeps the material model conservative and non-calibrated.

## Implemented

- 1.905 cm flat-to-flat hexagonal fuel element.
- 132 cm active length.
- 19 axial hydrogen coolant channels.
- 0.2565 cm coolant bore diameter.
- 0.40894 cm bore pitch.
- 50 micrometer ZrC channel and exterior coating.
- Parameterized OpenMC hexagonal lattice.
- Finite cylindrical core envelope.
- Beryllium radial reflector.
- Eigenvalue settings and a fissionable source box.
- 3-D mesh tallies for neutron flux, fission rate, and local heating.
- XY material plot definition.
- Pure-Python layout validation.

The geometric dimensions above are taken from the NASA NERVA-derived point-of-departure analysis listed in `REFERENCES.md`.

## Important model limitation

The default `(U,Zr)C-graphite` material is a **surrogate**, not a reproduction of a historical NERVA fuel loading. Its uranium enrichment is deliberately limited to low-enriched fuel (<20 wt% U-235) in `NervaConfig`. Therefore the first checkout is intended to validate geometry, OpenMC input generation, tallies, and coupling architecture. Do not interpret the default `k_eff` as a NERVA benchmark.

Tie tubes, control drums, filler blocks, thermal-hydraulics, temperature feedback, and historical benchmark calibration are the next model layers.

## Quick geometry validation

From the repository root:

```bash
python -m ntr.nerva.validate_layout
```

This requires only Python and checks the 19-hole pattern for overlap and edge clearance.

## Export OpenMC input

With the OpenMC Python package and nuclear-data library configured:

```bash
python -m ntr.nerva.build_model --output build/nerva_demo
```

To execute OpenMC after export:

```bash
python -m ntr.nerva.build_model --output build/nerva_demo --run
```

The `OPENMC_CROSS_SECTIONS` environment variable (or an equivalent OpenMC data configuration) must point to a compatible cross-section library before a transport calculation is run.

## Roadmap

1. Validate one fuel element and produce geometry plots.
2. Add an explicit tie-tube/support-element universe.
3. Build a representative fuel/tie-tube cluster.
4. Add cylindrical filler elements, Be reflector, and control drums.
5. Add axial/radial power extraction and normalization.
6. Couple the OpenMC heating map to a 1-D hydrogen coolant-channel solver.
7. Add temperature/density feedback iteration.
8. Calibrate only against published Rover/NERVA benchmark information.
