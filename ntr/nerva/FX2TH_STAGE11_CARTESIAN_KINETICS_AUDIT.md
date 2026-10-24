# NUKE / FX2-TH Stage 11 — *actual* 2D multigroup FV to delayed-family spatial kinetics

Date: 2026-10-24. Parent: Stage 10 PR #122. This branch links previously disjoint numerical solver development trees, but is NOT an OpenMC transport run, or experimental NERVA reactor qualification.

## What changed
The original 2D conservative finite-volume diffusion implementation from Stage 2 PR #112 and its 11 independent numerical regressions have been included unchanged on this stacked integration branch. They remain tracked by their original Stage 2 PR for separate code review. New **fx2th_cartesian_family_bridge.py** transfers the assembled physical 2D mesh and full multigroup loss matrix into the Stage 7 spatial delayed-neutron family transient WITHOUT a new discretization.

Stage 2: group-structured 2D scalar flux phi[y,x,g]; A phi=F phi/k, with A formed from cell-integrated absorption, outscattering and harmonic-interface diffusion currents. Each cell contains x*y area times *one meter out-of-plane depth*.

Stage 7: M dphi/dt=-A phi + F_prompt phi + delayed emission + Vq; local precursor birth V beta nuSigmaF phi and family-specific decay. Spatial neutron inverse-speed mass M=V/v.

The adapter maps cell index c=y*nx+x, retains A exactly, and transfers V and nuSigmaF ONCE. It does not recompute reaction coefficients or infer real delayed fractions/speeds.

## Static/dynamic physics identity
At stationary precursor equilibrium C[c,f]=(V[c]/lambda[f])*sum_g(beta[c,g,f]*nuSigmaF[c,g]*phi[c,g]), and if the prompt and every delayed-family birth spectrum match the Stage 2 eigenproblem chi[g], their sum is EXACTLY Stage 2 F phi, including energy-group scattering and cell-volume source integration. Thus a Stage 2 eigenvector with k=1 must be stationary under Stage 7's implicit time integrator at every time step. Noncritical eigenmodes are never misrepresented as equilibria. Distinct delayed spectra correctly invalidate this particular Stage 2 reference: their stationary fission operator is different.

The raw Stage 2 left eigenvector satisfies A.T psi=F.T psi/k and is normalized psi.T F phi=1 with F **already cell integrated**. Therefore the Stage 6 source-pairing sum_c,g V[c]*psi[c,g]*fission_rate[c,g] is equal to psi.T F phi *without further left-vector conversion or another multiplication of V*. This branch explicitly verifies this equivalence on UNEQUAL cell volumes, correcting an imprecise warning in the Stage 6 header concerning conversion of cell-integrated left eigenvectors.

The prompt neutron generation-time parameter derived from this reference is lambda_prompt_time=(psi.T M phi)/(psi.T F phi), with M=V/v. It is a mathematically defined multigroup diffusion spectral parameter, NOT a measured NERVA value. The family effective fractions require externally supplied delayed spectra, betas and the same discrete adjoint.

## Numerical verification
- Stage 2 original numerical suite: 11 tests of forward/adjoint eigen problems, harmonic face currents, reflecting/zero-flux boundaries, analytic sine-mode mesh refinement and subcritical fixed source.
- Stage 7 spatial-family suite: 19 tests of independent full-state ODE and matrix exponential, delayed group evolution and neutron balance.
- New Stage 11 suite: actual 2D FV matrix handoff, nonuniform 2D cell volumes, scattering/face cancellation, mixed boundaries, stationary critical one/two-group with local precursor inventories, cell-integrated adjoint pairing, analytic one-group generation time, subcritical forced steady-state, independent dense matrix-exponential time convergence and deliberate incompatible-spectrum/source/data rejections.

Use:
    python -m ntr.nerva.validate_fx2th_diffusion
    python -m ntr.nerva.validate_fx2th_family_spacetime
    python -m ntr.nerva.validate_fx2th_cartesian_family_bridge

## Physical limitations
All values are synthetic, because qualified NASA NERVA multigroup nuclear data, delayed-family spectra, Doppler/temperature dependence and real hot-core OpenMC benchmark statepoints were not supplied. This is numerical verification of mathematical equations, NOT the original 1978 FX2-TH FORTRAN source or real transport physics. The 2D Cartesian FV model must not be confused with the NERVA hexagonal/axial core geometry or OpenMC continuous-energy particle transport. A zero scalar-flux face is Dirichlet, not a true vacuum/Marshak boundary. Geometry/XS/source consistency and independent neutron balance/heat deposition remain physical validation tasks.

No NUKE engine cycle or main code changed. This PR is stacked on Stage 10 #122 and copies original Stage 2 files from PR #112 exactly. For actual integration, eliminate duplicated Stage 2 patch history appropriately rather than merging two different versions silently.
