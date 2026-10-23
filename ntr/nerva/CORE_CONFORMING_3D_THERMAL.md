# Conforming 3-D coated 19-bore hexagon: Python thermal physics

**Source:** core_hex_conforming_mesh.py, core_hex_conforming_fem.py,
run_core_hex_conforming_fem.py

**Status:** implemented numerical conduction + hydrogen conjugate equations.
No hot OpenMC neutron-source, nuclear KERMA or historical NERVA validation
is implied.

## Geometry and mesh

The geometry generator uses the optional Python Gmsh CAD and 2-D
triangulation API. The cross section includes:

- A real hexagonal outer *element* perimeter and its interior fuel region
- 19 circular coolant voids at the existing NUKE channel coordinates
- An explicit concentric ZrC annulus around each coolant void
- Conforming fuel/coating triangle interfaces: material changes occur on
  **shared triangle nodes**, not a prescribed annulus-to-sector resistor
- Individual physical boundary arc groups for all 19 hydrogen channels

The 2-D triangle mesh is tensor-extruded through the full set of axial
edges supplied by the NUKE channel reconstruction. Every triangular prism
has six P1(x,y) x P1(z) nodal unknowns for the fuel/coating solids; the
19 gas streams each have a separate axial finite-volume temperature.

### Known outer coating incompatibility in NUKE demonstration geometry

The repository's smaller inner-fuel hexagon ends at a flat-to-flat of
1.895 cm, while six existing outer channel coating rings intersect that
boundary by 0.00363 cm. The geometry cannot be meshed with those holes
as a simple planar domain, without changing the underlying design.

The mesher therefore uses the *configured full fuel-element hex* with
flat-to-flat 1.905 cm, which strictly contains all 19 coating rings.
Even there, the outer-six minimum clearance is only 0.00137 cm
(13.7 micrometers). It is a difficult narrow passage needing mesh
convergence studies, and the external perimeter ZrC skin has **not**
been explicitly constructed. The mesher fails closed if future
configuration data put any coated bore outside the physical envelope.
It never shifts the coolant channels or secretly shrinks coatings.

The corresponding geometry is the **existing NUKE surrogate**, not
guaranteed identical to a historical flight or test reactor.

## Governing thermal equation

For solid material m (fuel or ZrC coating):

    rho_m du_m(T)/dt = div[k_m(T) grad(T)] + q'''_m

The FEM discretizes the thermal weak form with continuous P1 shape
functions on XY triangles and linear shape functions in axial z.
For each extruded six-node prism:

    K_e = k_e [ M_z tensor K_xy + K_z tensor M_xy ]

The transverse mass and Fourier stiffness matrices are exact for a
linear triangular element. The axial mass and stiffness are exact for
a linear line element. Each prism's conductivity is evaluated at its
current six-node average during Picard iteration. Thermal storage uses
lumped node volume and the exact secant internal-energy increment
from the linear cp(T) law specified in CoreSolidLaw.

At a coolant wall segment with length L and axial height dz, convection
is represented by a **consistent** 4-node tensor-product wall matrix:

    B_wall = h [ M_z tensor M_edge ]

The same integrated heat flux appears with opposite signs in the solid
weak form and in that channel's hydrogen finite volume. Exterior solid
surfaces and axial ends use adiabatic natural Neumann conditions.

For every channel c and axial gas cell i:

    m_(i,c) [h(T_new)-h(T_old)]/dt
       + mdot_c [h(T_(i,c))-h(T_(i-1,c))]
       = Q_wall_(i,c) + Q_directH2_(i,c)

Constant-cp and frozen NASA-7 molecular hydrogen enthalpy are supported.
The fluid network has prescribed positive mass flows, inventory masses
and wall heat-transfer coefficients; no pressure, viscosity, momentum,
flow reversal, dissociation or compressible acoustic physics is solved.

Solid and coolant unknowns are **solved in one monolithic nonlinear
sparse system**, not as a decoupled thermal postprocessor. Each
nonlinear Picard iterate recomputes solid conductivity and variable
hydrogen enthalpy slopes; it does not advance physical time until a
converged backward-Euler step exists.

## Power assignment

Preferred input is full spatial source deposition in watts:

    solid_prism_power_w[element, axial_bin, XY_triangle]

Such an array can represent physically detailed fuel/coating energy
deposition if a caller supplies it and separately qualifies its source
nuclear-data and transport statistics.

For compatibility with existing NUKE data, a conservative projection
can distribute

    channel_wall_axial_power_w[element, channel, axial_bin]

among **fuel material** XY triangles according to nearest Voronoi
channel-site centroid and triangle area. This is an approximation to
the original channel-associated power map; it does NOT reconstruct
neutron collision positions, nor invent coating-local heat deposits.
Direct hydrogen energy deposition remains separate and is not counted
twice. The projection preserves all input channel totals to floating-
point roundoff.

## Execution and outputs

First generate and optionally cache a conforming mesh (requires gmsh
Python and GLU on the computer):

    python -m ntr.nerva.run_core_hex_conforming_fem --help

Use the runner with the existing
element_channel_power_reconstruction.npz, explicit material and gas
property inputs. Its --mesh-cache and --save-mesh controls let the mesh
be constructed once and reused for iterative thermal solves.
Use --detailed-source for an explicit per-triangle power field.

Outputs are:

    conforming_core_3d_fields.npz
    conforming_core_3d_report.json

They contain full solid nodal fields, gas temperatures, wall heat rates,
material/triangle mesh, z coordinates, and a global first-law ledger.
Physical source qualification remains explicitly false.

## Code verification and future physical validation

The lightweight finite-element smoke test uses two conforming
fuel/coating square-annulus regions, an axially extruded mesh,
convection into one of 19 H2 control volumes, and an isolated
zero-source equilibrium. It exercises time integration and global
energy closure, not empirical reactor validation.

A separate optional Gmsh workflow generates the actual configured
19-bore triangulation and checks physical-region and wall-boundary
tagging. Production accuracy needs more: XY and z mesh convergence,
element geometry/conditioning assessment of the narrow outer-six
ligaments, independent 3-D heat-conduction benchmark solutions,
actual coating interface resistance/contact and possible gaps,
temperature-dependent property tables with uncertainty, measured
film transfer, spatial nuclear energy deposition from evaluated
hot cross-section data, and historical NASA NERVA comparison.

The present code is *thermal physics software*, not a qualified hot
reactor-core multiphysics solution or a reactor design optimizer.
