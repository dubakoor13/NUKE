# Stage 12 — conservative axisymmetric neutron diffusion + family-resolved kinetics

Date: 2026-10-24. Parent: Stage 11 PR #123.

## Governing physics and discretization

This milestone connects the Stage 7 cellwise family-resolved delayed-neutron transient to a newly assembled, conservative **cylindrical r-z multigroup diffusion operator**. The earlier imported NUKE `diffusion_2d.py` (PR #116) returns only homogeneous centimeter-based eigen solutions; it does not export cell-integrated sparse operators and does not handle a generally nonuniform material mesh. The new solver uses SI meters and accepts nonuniform radial/axial cell edges, cell/group-dependent D and macroscopic nuclear data, explicit intergroup scattering, and distinct physically defined boundary options.

For cell (i,j) with annular radial faces r_i and r_{i+1} and axial faces z_j and z_{j+1}:

V_ij = pi*(r_{i+1}^2-r_i^2)*(z_{j+1}-z_j).

The radial shared face has area 2*pi*r_face*dz, with reciprocal conductance t=A_face/[dr_i/(2D_i)+dr_neighbor/(2D_neighbor)]. The shared axial face area is pi*(r_outer^2-r_inner^2), with the analogous axial half-cell resistance. Interior faces have exactly equal and opposite coefficients, hence conserve neutron current. The cylindrical axis (r=0) has zero radial current; Dirichlet zero-flux at a nonaxis boundary face uses a HALF-cell distance, not a whole-cell distance.

Loss matrix A is cell-integrated removal (absorption + outscatter) plus leakage and negative incident-to-emitted scattering; fission F is cell-integrated chi_g * nuSigmaF_h. Flux unknown indexing is ((radial_index*n_axial+axial_index)*energy_group+g).

Static multigroup eigenproblem: A phi = F phi/k_eff. Adjoint: A.T psi = F.T psi/k_eff. Adjoint normalization psi.T F phi=1. Gamma/kinetics uses M[c,g]=V[c]/v[g] and exactly the SAME A, geometry and group ordering, with distinct precursor-family beta[c,incident_g,f], chi_d[c,f,emitted_g] and lambda_f.

At criticality, when prompt/delayed spectra match the static fission spectrum, equilibrium precursor concentration C_eq[c,f] = V[c]*sum_h(beta[c,h,f]*nuSigmaF[c,h]*phi[c,h])/lambda_f. With k=1 this state must be stationary under the full Stage-7 implicit time step. When delayed emission spectra differ, the historical static F operator changes: the code must reject an inappropriate old eigenvector reference instead of claiming equilibrium.

## Independent numerical reference tests

- Full annular-volume integration and zero-current axis symmetry.
- One-cell, reflecting and bare Dirichlet analytical fission eigenvalues.
- Harmonic interface conductance on a nonuniform material/radial grid.
- Global neutron loss = volume-weighted absorption + external leakage, with interior currents and multigroup scattering canceling.
- Forward and discrete adjoint against an independent dense generalized eigensolver.
- Mesh refinement against the homogeneous cylindrical J0(alpha*r/R)*sin(pi*z/H) bare eigenmode, alpha=first root J0.
- Analytic Bessel volume-integrated radial power bands and sine-integrated axial power bands.
- Critical one/two-group family precursor equilibrium and subcritical fixed-source response.
- Full transient eigen-mode convergence against an independently constructed dense matrix exponential.
- Fail-closed mismatched geometry, birth spectra and unphysical input arrays.

Runs:
  python -m ntr.nerva.validate_fx2th_family_spacetime
  python -m ntr.nerva.validate_fx2th_rz_family_diffusion

## Physical qualifications NOT claimed
1. Geometry is a rotationally symmetric cylindrical benchmark, NOT hexagonal-core NERVA geometry, nonaxisymmetric reflector/control mechanisms or flight-qualified reactor dimensions.
2. No evaluated neutron cross-section library, actual fuel isotope composition, validated delayed-neutron spectra or calibrated material-temperature coefficients were used. All numerical tests have SYNTHETIC cross sections.
3. The outer zero-flux boundary is Dirichlet at the specified/extrapolated face and is NOT an exact vacuum neutron-transport boundary condition. Thermalization, hydrogen moderation, transport angular effects and real reflector drums require additional qualified models.
4. No actual OpenMC neutron/photon statepoint, criticality convergence or historical hot-core measurement was reproduced. Mathematical verification != NERVA physical validation.
5. The current solver uses standard inverse source iteration and implicit backward Euler, not the historical FX2-TH FORTRAN implementation or its full improved-quasistatic algorithm.

Existing Stage 11 Cartesian solver is preserved for independent dimensional/boundary checks. The older NUKE 2D cylindrical code from PR #116 is preserved unchanged on its branch; this new solver is a conservative FV integration layer for time-dependent core equations. No engine cycle or NUKE main source was changed.

Next priority: physically qualified cross-section and delayed-spectrum inputs, real-statepoint OpenMC comparisons, and refined thermal feedback mapped to the same core mesh.