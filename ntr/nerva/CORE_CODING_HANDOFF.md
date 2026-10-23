# NUKE NERVA reactor-core physics: Python coding handoff

**Scope:** actual reactor-core heat transport and neutron transport interfaces.
No engine/nozzle, turbopumps, control tuning or general repository cleanup.

**Branch:** feat/core-implicit-conjugate-fv-20261023
**PR:** https://github.com/dubakoor13/NUKE/pull/109 (DRAFT; do not
call the complete core physically validated, and do not merge unchecked)

## Implemented code available on the PR

### A. 19-channel reduced-order conduction + H2 energy
- core_conjugate_fv.py: implicit radial/axial annular fuel and ZrC layers
- core_hex_lateral_fv.py: 42 real Voronoi shared-face lateral links
- core_hex_radial_lateral_fv.py: full combined annular-radial and lateral thermal
  energy system
- corresponding independent local/first-law audits, executable CLIs and
  deterministic numerical checks

### B. True conforming fuel cross section and extruded 3-D solid thermal FEM
- core_hex_conforming_mesh.py: 19 actual coolant holes and coating rings,
  meshed with Gmsh in a shared fuel/coating XY triangle mesh
- core_hex_conforming_fem.py: solid temperature P1(XY) x P1(z) continuous
  6-node triangular-prism Galerkin; variable solid k(T) and cp(T),
  Newton/Picard-like coefficient iteration, transient BE thermal storage
- 19 H2 channels with implicit upwind enthalpy and mass inventory, monolithic
  solid/gas matrix; shared wall convection matrix
- run_core_hex_conforming_fem.py: run one or more repeated fuel elements
  using the existing NUKE [element,channel,axial] reconstruction NPZ;
  optional cached mesh and detailed per-prism energy sources
- core_hex_conforming_power.py: exact Cartesian voxel/triangle XY clipping,
  exact axial overlap and material-separated fuel/ZrC heating transfer to
  triangular prisms (source requires explicit W/cm3, not flux or keff)
- CORE_CONFORMING_3D_THERMAL.md: detailed formulas and validity boundaries

### C. Executable code evidence (synthetic inputs, NOT physical validation)
- Real 19-bore Gmsh mesh built in GitHub Actions:
  34,024 XY nodes, 65,796 triangles, all 19 H2 boundary sets
- One full conforming extruded 3-D solid/H2 thermal transient ran to
  completion; measured first-law residual -9.420249469e-8 J on
  prescribed source: https://github.com/dubakoor13/NUKE/actions/runs/39991389955
- Core numerical smoke, optional mesher smoke and NUKE model regression
  CI configurations in .github/workflows.
- No actual hot-energy-deposition or nuclear reference validation claimed.

## Confirmed geometry issue

The NUKE inner-fuel hexagonal envelope (1.895 cm flat-to-flat) overlaps
six outer coated coolant bores by about 0.00363 cm. Those holes cannot
be embedded in the smaller planar domain. The conforming mesher uses the
declared 1.905 cm external fuel-element hex instead; the margin to these
six bores is about 0.00137 cm. The external perimeter ZrC skin is not yet
explicitly represented; a physically consistent 3D geometry is still needed.
Do not silently shift the holes or adjust nuclear geometry to make it fit.

## CODE pending before a full first-principles NERVA core simulation

1. **Material-correct full 3-D solid geometry.** Resolve external coating,
   bore coating near outer perimeter, fuel/coating interface/gap and mesh
   quality. Expand heat diffusion to interfaces between repeated fuel elements,
   tie-tubes and thermal boundaries using physically specified interfaces.
2. **Temperature-dependent fuel/coat property providers.** Add actual
   material/phase specific conductivity, cp, density and contact models,
   plus robust interpolation/extrapolation limits, rather than constant or
   linear assumed properties.
3. **Conservative coolant flow + momentum.** Extend the channel thermal
   network to nonuniform pressure, compressible mass/energy/momentum and
   property-consistent H2 states with wall conjugate heat flux. Current code
   uses prescribed flow, mass inventory and heat-transfer coefficient.
4. **True spatial OpenMC→solid energy deposition.** Existing new mapper
   is implemented, but upstream material-separated evaluated transport
   heating and correct coating/H2 deposition scores must be added or
   qualified; the cold NNDC test data are insufficient for hot local KERMA.
5. **Spatial solid/H2 temperature feedback into OpenMC.** Uniform bulk
   surrogates are insufficient. The actual transport material/cell or
   component zoning and data provenance need nonuniform thermal feedback,
   evaluated XS temperature coverage and geometry/material consistency.
6. **Neutron–thermal nonlinear fixed point.** Finish reactor-core-only
   coupled transient stepping and source/neutronic feedback with strict
   provenance and no invented criticality or reactor power.
7. Later: independent NASA historical reference verification, refined
   multiseed statistical confidence and physical validation; requested
   deferment applies here, not to basic software smoke checks.

## Next coding recommendation

Implement fully material-correct exterior/coating geometry **without
altering canonical nuclear design parameters**, then connect the
conforming thermal mesh to multiple neighboring element thermal faces.
The new Gmsh/FE FEM and voxel-power mapping are the foundation; avoid
adding yet another disconnected thermal demonstration model.

Use Python for core numerics and software smoke first. Do not claim
C++ port complete until this physical code scope is actually implemented.

## Latest coding increment: tabulated thermophysics and H2 momentum coupling

New modules (see CORE_TABULATED_THERMAL_HYDRAULICS.md):
- core_solid_material_tables.py — temperature-dependent fuel and coating
  conductivity/specific heat tables, exact cp(T) internal-energy integral,
  no silent extrapolation beyond supplied temperature range.
- core_hex_conforming_hydraulics.py — optional steady outer fixed point
  joining conforming 3-D fuel/ZrC/H2 thermal FEM to the existing NUKE
  compressible hydrogen 1-D mass/momentum/stagnation-energy solver.
- core_hex_conforming_fem.py accepts signed steady kinetic-energy flux
  correction, essential when the pressure solver updates H2 velocity.
- run_core_hex_conforming_fem.py supports --fuel-material-npz,
  --coating-material-npz, --couple-hydraulics and --inlet-pressure-pa;
  exports separate hydraulic NPZ and energy/momentum residual metrics.
- validate_core_solid_material_tables.py and
  validate_core_hex_conforming_hydraulics.py perform numerical-only
  execution checks, including CLI source/table/pressure integration.

The coupled hydrogen solver remains SUBSONIC and QUASI-STEADY only,
with prescribed flow rates/inlet pressure and screening correlations.
True hydrodynamic transient and detailed property sourcing are
still major *coding* work, not simply deferred validation.
