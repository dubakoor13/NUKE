# NUKE / FX2-TH Stage 16 — exact adjoint first-order neutronic sensitivity

Date: 24 October 2026. Parent: Stage 15 PR #127. Scope: core neutron diffusion, fission birth and numerical perturbation verification only.

## Physical and mathematical problem

Stage 15 established the complete prompt+delayed-family multigroup fission operator with separate emitted-neutron energy spectra and the corresponding forward/adjoint eigenproblem. Stage 16 adds the **first-order response of k-effective and reactivity** to externally specified changes in diffusion, absorption, scattering, total nu-fission neutron production, incident-group family delayed fractions, and prompt/family-delayed birth spectra.

No physical cross-section-versus-temperature law, nuclear design reactivity coefficient, NERVA operating point or evaluated data is constructed here. The user of this module must supply meaningful dXS/d(theta) and dSpectrum/d(theta). Its output is per unit of that external theta (e.g., K only if correctly supplied coefficients are per K). All synthetic tests use arbitrary dimensionless perturbation directions, not real NERVA temperature worths.

## Adjoint differential of the generalized eigenproblem

The spatial loss/transfer operator A, fission birth operator F, flux phi and left adjoint psi satisfy:

 A phi = (1/k) F phi
 A.T psi = (1/k) F.T psi

For differentiable A(theta) and F(theta) with simple real dominant eigenvalue, constant spatial geometry and fixed group structure, the exact first-order identity is:

 d(ln k)/d(theta) = [psi.T (dF/d(theta) - k*dA/d(theta)) phi] / [psi.T F phi]

 dk/d(theta) = k*d(ln k)/d(theta)
 rho = 1-1/k ; d rho/d(theta) = (dk/d(theta))/k**2.

With Stage 15 adjoint normalization psi.T F phi=1, the denominator is nominally unity, but this implementation evaluates it explicitly to remain normalization invariant. This is a *local linear sensitivity*, not a finite-reactivity insertion solver. Near degenerate/near-defective eigenvalues and large perturbations can invalidate the linearization.

The sensitivity report separates negative loss contribution, prompt fission contribution and each delayed family. These terms add to the derivative of ln k. It DOES NOT report the derivative of beta_eff: that would also require forward/adjoint eigenvector response, not just the eigenvalue derivative.

## Analytic fixed-mesh derivatives of conservative axisymmetric r-z diffusion

Stage 12’s cell-integrated material operator has:
  cell volume = pi*(r_out**2-r_in**2)*dz,
  radial shared-face area = 2*pi*r_face*dz,
  axial shared-face area = pi*(r_out**2-r_in**2).

The shared-face conductance is T=area/(l_i/D_i + l_j/D_j), including the harmonic discontinuous diffusion coefficients; its exact derivative is:

 dT = area*(l_i*dD_i/D_i**2+l_j*dD_j/D_j**2) / (l_i/D_i+l_j/D_j)**2.

Boundary zero-flux conductance uses ONLY centre-to-face resistance. Its derivative is area*dD/(half_width); a zero-current reflecting face has no conductance contribution. At the symmetry axis the radial face area is zero. Cell removal tangent includes V*dSigma_a and V*sum of outgoing dSigma_s; offdiagonal group in-scattering adds -V*dSigma_s(incident,emitted). Thus scattering and internal face currents continue to cancel globally, while true absorption and external leakage contribute to the total loss. Geometry is not differentiated in Stage 16.

## Full family prompt/delayed fission derivatives

For each spatial cell c, incident group h, emitted group g and delayed family f:

 F_prompt[c,g,h] = V[c]*chi_p[c,g]*nuSigmaF[c,h]*(1-sum_f beta[c,h,f])
 F_delayed_f[c,g,h] = V[c]*chi_d[c,f,g]*nuSigmaF[c,h]*beta[c,h,f].

Their derivatives include every product-rule term: d(nuSigmaF), d(beta), d(chi_prompt) and d(chi_delayed_family). The emitted spectrum tangents MUST sum to zero over emitted group, maintaining chi normalization. Delayed fraction tangents may have either sign, but should only be used within their physically admissible parameter interval, which is not determined here. Static equilibrium has no derivative contribution from lambda_f alone (family decay rates affect inventories and time dependence, not F_total).

## Independent numerical checks

The test program uses the ORIGINAL Stage 12 r-z finite-volume loss matrix assembler, ORIGINAL Stage 15 fission matrix assembler, and scipy.linalg.eigvals to compute numerical k for positive and negative small perturbations. Those independent re-assembled eigenvalues are central-differenced in the declared input parameter; these comparisons do not reuse the new analytic tangent.

Checks include:
- Full matrix-level dA and dF vs independently re-assembled centered differences.
- Absorption, nonuniform radial diffusion/leakage, up/down energy-group scattering, group-specific nuSigmaF, delayed beta, prompt birth spectrum and individual delayed-family birth-spectrum eigenvalue sensitivities.
- Mixed-material perturbation and d reactivity/dtheta against independent dense generalized eigenvalue differences.
- Analytic one-group reflected-cell k=nuSigmaF/Sigma_a, absorption derivative -nuSigmaF/Sigma_a**2, and uniform nuSigmaF scale giving d(ln k)/dtheta=1.
- Exact one-cell Dirichlet radial/axial half-face derivative, cancellation of internal diffusion and scattering for reflecting volumes, production conservation for beta-only and spectrum-only variations.
- Shape/input/unit guards and fail-closed stale r-z material coefficients.

Run:
  python -m ntr.nerva.validate_fx2th_full_family_fission_eigen
  python -m ntr.nerva.validate_fx2th_rz_adjoint_sensitivity

## Remaining physical qualification

This is numerical *verification of differential operators*, not real NERVA nuclear-data qualification, neutron-transport accuracy, temperature-dependent Doppler coefficient, hot-hydrogen feedback, control-drum differential worth, measured reactor reactivity, source convergence or physical NASA NERVA reactor validation.

Required next: verified evaluated nuclear-data derivations of XS and prompt/delayed spectra with temperature/material and uncertainty, independent OpenMC transport sensitivity comparisons on a matched geometry and energy grid, cross-section condensation and covariance-aware propagation. The Stage 16 output must not be identified as a physical d(k)/dT unless dA/dT and dF/dT are independently qualified.

Stage 16 is stacked on Stage 15 PR #127; NUKE main is unchanged.
