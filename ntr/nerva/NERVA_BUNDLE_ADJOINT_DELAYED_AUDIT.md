# NERVA bundle — delayed-neutron and adjoint numerical audit (2026-10-24)

**Status:** source review and synthetic mathematical verification only. No real NERVA nuclear data or experimentally validated reactor kinetics.

## Verified source and original tests

Source: NERVA-CORE-BUNDLE.zip on NUKE main (SHA-256 ede9990ddeac28c2537699b4af838429d4b0e7936686bc9b2e15f292b569bb28).

Inspected archive files: adjoint.py, delayed_parameters.py, heritage/kinetics.py, kinetics_transient.py, heritage/data/emrich2023_delayed_neutrons.json, and related original validators.

Executed the original archive tests in a clean temporary GitHub Actions environment using its corrected package root. Results: validate_adjoint **24/24**, validate_delayed_parameters **23/23**, validate_kinetics_transient **26/26**, validate_heritage_kinetics **41/41**. Total **114/114** checks passed. This is evidence of internal consistency, not historical reactor validation.

## Finding 1: beta_eff in the original archive is not a general multigroup delayed-production operator

The source's adjoint.py / importance_weighted_beta uses the peak forward and adjoint flux at each incident energy group to weight its delayed fraction. It does not supply the outgoing delayed birth-energy spectrum, and it discards spatial integration. Therefore it cannot establish a physically qualified effective delayed neutron fraction for general multigroup diffusion/transport.

Correct formal expression: beta_eff = (psi, F_delayed phi)/(psi, (F_prompt + F_delayed) phi).

At cell c, emitted group g, the delayed operator has production proportional to chi_delayed[c,g] times the sum over incident group h of beta[c,h] * nuSigmaF[c,h] * phi[c,h]. The prompt operator uses its separate emitted spectrum and factors (1-beta[c,h]). It is the ADJOINT IMPORTANCE AT EMISSION ENERGY, not merely at the incoming fission group, that matters. Family-dependent delayed spectra would be needed for family-specific beta_eff.

## Finding 2: original delayed_parameters.py assumptions require correction

Its mixture_delayed_fraction restricts importance_correction to (0,1]. A physically valid ratio beta_eff/beta_unweighted is not necessarily <=1; it may be greater or less than 1 depending on prompt and delayed birth importance. Its BetaEffBudget.is_fully_sourced method infers sourcing from correction != 1, which is also not valid provenance evidence. Neither behavior should be integrated unchanged.

## Finding 3: existing kinetics code is overlapping, not interchangeable

FX2-TH stage 1 (NUKE PR #111): fully implicit reduced point kinetics, effective parameter arrays supplied externally.
FX2-TH stages 3–4 (PRs #113–114): space-time multigroup diffusion / IQS-inspired shape evolution and spatial delayed precursors.
Archive heritage/kinetics.py: reference-based six-group point kinetics and RK4 integrator.
Archive kinetics_transient.py: lumped thermal-feedback transient with prompt-jump approximation.
These approximate different equations and regimes. Do not silently replace one with another or treat any as validated NERVA startup prediction.

## New standalone code in this PR

delayed_importance_operator.py explicitly requires forward flux, adjoint flux, cell volumes, nu-fission cross sections, incident-group beta, and BOTH prompt and delayed emission spectra. Returns effective/unweighted fractions, group contributions, and prompt/delayed production ledgers. It has **no default reactor nuclear parameters**. It labels all outputs as dependent on supplied, not independently qualified physical data.

validate_delayed_importance_operator.py contains 12 synthetic independent tests, including two-group hand calculation, one-group identity, adjoint/flux scaling invariance, both sides of importance-ratio unity, spatial volume weighting, contribution closure and bad-input rejection.

Test: python -m ntr.nerva.validate_delayed_importance_operator

## Required further qualification

1. Nuclear data and uncertainty for isotope-specific prompt and delayed fractions/spectra, including delayed families, and multigroup nu-fission cross sections.
2. Physically consistent forward/adjoint solver normalization on the same energy groups, mesh and cell-integrated matrix convention; never double-apply cell volume factors.
3. Independent reference transport comparisons, spectrum collapse verification, thermal-state validation, and applicable historical NERVA experimental results.
4. No conversion of Emrich U-235 tabulated constants into NERVA-specific effective kinetics without actual mixture and neutron-spectrum evidence.

No historical FX2-TH source was recovered from this bundle. No nuclear operating condition, geometry, enrichment or OpenMC transport model is changed.
