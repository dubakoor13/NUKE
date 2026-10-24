# FX2-TH / NUKE Stage 6 — delayed-family source and kinetics contract
**Date:** 2026-10-24. **Status:** synthetic numerical interface checks only.
**Parent:** NUKE PR #117 (feature/adjoint-delayed-operator-audit-20261024).
**Archive:** User-uploaded NERVA-CORE-BUNDLE.zip; evaluated historical fission data is not supplied by this module.

## Why this module exists
Stage 5 (delayed_importance_operator.py) distinguishes prompt and delayed EMISSION spectra but collapses delayed neutrons into one spectrum and one delayed fraction. For kinetics, each delayed family has its own DECAY constant and potentially distinct BIRTH spectrum. The original NERVA bundle adjoint.py and delayed_parameters.py do not provide a general multigroup delayed source operator.

Stage 6 (delayed_family_contract.py) explicitly takes:
- cell/group forward and adjoint fluxes, nu-fission cross sections and cell volumes;
- incident-group-by-family beta[cell, incident_group, family];
- prompt spectrum chi_p[cell, emitted_group];
- family spectra chi_d[cell, family, emitted_group];
- positive decay constants lambda[family] and a nonempty provenance/source identifier.

No NERVA nuclear defaults are embedded.

## Production ledger

For cell c, family i and outgoing group g:
D[c,i,g] = V[c] * chi_d[c,i,g] * sum_h(phi[c,h] * nuSigmaF[c,h] * beta[c,h,i]).

Prompt emission:
P[c,g] = V[c] * chi_p[c,g] *
           sum_h(phi[c,h] * nuSigmaF[c,h] * (1 - sum_i(beta[c,h,i]))).

Then:
beta_eff[i] = sum_c,g(psi[c,g] * D[c,i,g]) /
              sum_c,g(psi[c,g] * (P[c,g] + sum_j(D[c,j,g]))).

Sum over families gives beta_eff for the SUPPLIED FROZEN state. Unweighted emission-based family shares and total neutron-source conservation are checked separately. Inputs must have positive volumes, positive precursor decays, finite arrays, normalized outgoing spectra, physically bounded beta per incident group, and nonzero weighted fission production.

The bilinear pairing uses cell-INTEGRATED emission with unintegrated forward/adjoint importance: do not double-apply cell volumes if your discrete solver already integrated its operator.

## Compatibility bridges

- Stage 5 aggregate delayed_importance: when all family birth spectra equal one another, summing the family emission produces exactly the aggregate operator. This is tested.
- Stage 1 FX2-TH KineticsCoefficients: result.point_kinetics_kwargs(generation_time_s, reactivity, source_amplitude_per_s) yields named inputs. All three scalars must come from the caller; none are inferred from k-effective. The effective family fractions and individual decay rates are preserved.
- Stage 3 FX2-TH DelayedData: legacy_spacetime_contract returns RAW physical beta and lambda values only if beta is uniform by cell and incident group, and prompt/delayed birth spectra match the same grid spectrum. Otherwise it raises rather than silently degrade physics. Group speeds are not fabricated.

The Stage 1 and Stage 3 modules remain on separate review branches; this PR does not modify their internal code.

## Numerical verification

Run: python -m ntr.nerva.validate_delayed_family_contract

15/15 local tests passed on 24 October 2026, including direct tensor-product reference, conservation, aggregate Stage 5 identity, single-group identity, normalization independence, family preservation, and strict refusal of lossy Stage 3 transformations. Local compatibility smoke tests independently instantiated original Stage 1 KineticsCoefficients and Stage 3 DelayedData from the results.

Also run Stage 5 original 12-check suite on this stacked PR.

## Physical qualification gates

1. Independently evaluated nuclide-specific delayed-family yields, group-resolved prompt/delayed emission spectra and uncertainty.
2. Consistent forward/adjoint operator grouping, normalization, mesh and material temperatures.
3. Externally derived effective generation time and reactivity with uncertainties, not a k-effective shortcut.
4. Genuine spatial precursor and thermal feedback and converged time integration, before claiming NERVA transients.
5. Independent applicable historical experimental benchmarks.

No historical FX2-TH FORTRAN source was reconstructed here, no OpenMC transport was executed, and no real NERVA prediction is claimed.
