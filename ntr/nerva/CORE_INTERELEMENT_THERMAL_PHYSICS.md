# 3D reactor fuel-element perimeter and between-element thermal physics

**Implemented Python files:**
- core_hex_exterior_surface.py
- core_hex_interelement_mortar.py
- core_hex_two_element_fem.py
- validate_core_hex_exterior_surface.py
- validate_core_hex_interelement_mortar.py
- validate_core_hex_two_element_fem.py

These modules extend NUKE's existing conforming XY triangle × axial P1
fuel/coating conduction solver. The output is not calibrated to NASA NERVA
reactor data, and no nuclear fission or OpenMC multiplication calculation
is performed here.

## External hexagonal perimeter thermal boundary

The exposed outer hex surfaces are identified by mesh topology: exactly
one-owner 2D triangle edges that are NOT any of the 19 H2-coolant-bore
wall segments. The internal fuel/ZrC edges have two triangle owners and
cannot be mistaken for an exposed perimeter. External faces inherit the
axial FE mesh by tensor extrusion.

The user can supply a spatially uniform scalar or [z,external_edge] fields:

- external contact conductance h_contact [W m^-2 K^-1]
- contact reservoir absolute temperature T_bath [K]
- greybody surface emissivity epsilon [0..1]
- radiative environment temperature T_rad [K]

The signed outward heat flux is

    q_out = h_contact (T_surface-T_bath)
          + epsilon * sigma (T_surface^4-T_rad^4).

To preserve a positive thermal linearization, the greybody component
uses an exact secant radiative film coefficient evaluated at the
area-mean surface temperature during each nonlinear Picard pass:

    h_rad = epsilon*sigma*(T_surface+T_rad)
                          *(T_surface^2+T_rad^2).

The full 4-node P1(edge) × P1(z) *consistent* FE wall matrix is inserted
in the solid system. The boundary temperature is the coefficient-weighted
contact-plus-radiation environmental bath. The global first law now
accounts for signed heat lost or gained at the external perimeter.
The zero-contact, zero-emissivity limit restores adiabatic insulation.

Runner: run_core_hex_conforming_fem.py now has
--outer-contact-w-m2-k, --outer-contact-temperature-k,
--outer-emissivity, --outer-radiation-temperature-k.
Nonzero heat-transfer coefficients require explicit temperature inputs.
These are **external prescribed bath** conditions, not a self-consistent
adjacent element temperature solve. Do not interpret them as automatically
computed neighboring reactor elements or a certified gap law.

## Conservative mortar coupling across neighboring fuel elements

core_hex_interelement_mortar.py constructs a symmetric, positive-semidefinite
thermal contact stiffness for two physically *aligned*, flat, facing
outer fuel-element sides. The two sides may have DIFFERENT mesh edge
subdivisions; coincidence of individual P1 nodes is not required.

For the exact geometric overlap of one edge from A and one from B,
the 3-D contact energy functional is

    E_contact = 1/2 integral H*(T_A-T_B)^2 dS

and the corresponding assembled local element matrix is

    K_contact = integral H * [N_A,-N_B]^T [N_A,-N_B] dS.

The code finds overlapping intervals in a common side tangent
coordinate. It integrates P1 edge shape functions (different on A
and B) and P1 axial shape functions by tensor 2×2 Gaussian quadrature,
which is exact for products of these linear shape functions. The
result is one symmetric global sparse matrix with no prescribed
bath or artificial heat source. It satisfies K_contact*1 = 0,
and heat leaving A equals heat entering B.

User inputs are each element's physical XY origin, a common contact
face normal from A to B, a shared axial grid, and a *prescribed*
thermal contact conductance H>=0. Noncoincident, partially overlapping
or unphysical contact sides fail explicitly. Gap conduction and
mechanical contact deformation are NOT inferred or optimized.

This operator is not a substitute for a complete reactor core array
mesh or a calibrated contact-pressure law.

## Actual two-element monolithic thermal solver

core_hex_two_element_fem.py supplies a 2-fuel-element, 38-H2-stream
thermal calculation rather than merely assembling an unused interface.

It builds the SAME P1 triangle×line weak-form conduction and consistent
coolant wall-matrix for each local element, each with separate 19-channel
upwind H2 enthalpy, user-provided volumetric prism power and optional
backward-Euler solid/H2 energy storage.

The block-diagonal element systems are augmented with the symmetric
nonmatching interface mortar, remapped from [A-solid,B-solid] to the
global [A-solid,A-H2,B-solid,B-H2] unknown ordering.

The global sparse matrix and nonlinear material/enthalpy Picard
updates are solved TOGETHER. The first law for each element includes
the shared contact heat with equal/opposite signs, and the total
first law cancels the shared interface exactly:

    dE_A/dt = Q_A + H2_in_A-H2_out_A - Q_AtoB
    dE_B/dt = Q_B + H2_in_B-H2_out_B + Q_AtoB

All quantities are in W, except stored energy terms in J over a
transient time step.

The model does not yet assemble arbitrary N-element packing, tie tubes,
moderator paths, chamber/manifold hydraulics or external ZrC shell.
The original 3D conforming fuel-element geometry's thin outer ligaments
and omitted perimeter coating remain physical-geometry coding gaps.

## Software checks and remaining code

Lightweight synthetic smoke checks provide:

- Exact greybody Stefan-Boltzmann surface flux and contact flux
- Energy-loss accounting for transient 3D conjugate fuel/H2 system
- Correct zero-film adiabatic limit
- Matched and *unmatched* external P1 edge interface support
- Positivity/symmetry/conservation of the contact mortar
- Monolithic two-element thermal coupling with 38 H2 streams
- Shared interface flux equality and per-element first-law ledger
- Steady and backward-Euler numerical paths

These are **software execution checks**, not historical reference
validation. In the next coding iteration, generalize the block solver
from 2 elements to arbitrary packed fuel/tie-tube elements, with
physical fuel/ZrC coatings and external material contact geometry.
Then implement spatial nonuniform OpenMC thermal feedback and qualified
nuclear source reconstruction; physical reactor validation follows
only after the underlying Python code is complete.
