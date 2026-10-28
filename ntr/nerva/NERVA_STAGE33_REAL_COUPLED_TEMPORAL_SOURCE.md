# Stage 33 — real coupled OpenMC temporal fission-source history (core physics)

**Baseline:** Stage 32 PR #146 proved that four *actual*, independently seeded OpenMC 0.15.3 coupled neutron/photon simulations ended with quite different final fission-source banks: the observed between-seed average 4 × 4 × 10 spatial total-variation distance was **0.5860555556**. Only final banks were available; no temporal convergence analysis was possible.

This experiment changes **only the saved fission-source statepoint schedule** to `30,45,65` for each of the *same* RNG seeds `13007, 23011, 33013, 43019`. The actual neutron/photon coupled OpenMC C++ transport is rerun at fixed 3-ring cold conceptual geometry, 3000 neutron starts per generation, 65 batches (20 inactive), the same material temperatures, evaluated data library, scoring meshes and tallies as Stage 31. There is **no enrichment, material, geometry, control or nuclear-data tuning**.

## Physical and numerical observables

The original 12 HDF5 `StatePoint` files are saved, with corresponding XML, transport-provenance SHA256 and evaluated-data reports. Stage 31's original-HDF5 four-seed **neutron/photon/electron/positron heating energy partition audit** is rerun on the newly executed *final* checkpoint statepoints. That establishes identical physical run configuration, evaluated cross-section index and particle/material/mesh heating conservation for the new four-seed experiment. Final original SHA256s must match this new genuine heating ledger.

The Stage 33 independent raw-HDF5 reader rejects absent/forged source data or mismatched `photon_transport`, RNG seeds, batch counts, source presence, particle count and active realization totals. It enforces **bit-exact nested prefixes** of the same native `k_generation` and entropy time histories at batches 30/45/65. Inconsistent prefixes mean the apparent source-time series did not come from one unmodified Monte Carlo run.

The source-bank reader measures each checkpoint's **weighted spatial source probability**, centroid, exact repeated positions, spatial entropy and effective distinct-position diversity (*not* an estimate of statistically independent fission ancestors). Its 4 × 4 × 10 source histogram is integrated into:

- Within each seed: three observed source spatial total-variation distances, 30→45, 45→65 and 30→65, plus axial source profiles.
- Across independent seeds: six pairwise spatial TV distances at **each** of the three source checkpoints, including means and maxima.
- Coarsened 2 × 2 × 5 version of the same distributions, with mathematical assertion that coarsening **cannot increase TV**. This is numerical indexing/resolution verification, not a physical convergence claim.
- Disjoint *observed* active-generation mean `k_generation` and Shannon entropy windows: batches 21–30, 31–45, and 46–65. These source generations remain correlated; these means do **not** supply independent Monte Carlo confidence intervals.

## Acceptance and limitations

New GitHub Actions workflow `nerva-stage33-real-coupled-source-time.yml` executes **four actual OpenMC v0.15.3 C++ coupled transport runs**, saves 12 native HDF5 snapshots, invokes the verified Stage31 real-statepoint heating covariance cross-check, and then requires Stage33 tests that reopen the actual 12 binaries. Synthetic adversarial tests run separately and cannot substitute for transport.

**Hard qualification flags remain FALSE** even if all tests pass: critical NASA NERVA core model, stationary fission source, independent neutron-history uncertainty, hot evaluated data coverage, physical deposited watts or benchmark match. These cold, highly subcritical conditions are not a validated historical NERVA operating point. Three snapshots per chain provide a measured spatial drift diagnostic only, without a valid convergence hypothesis test.

No genuine Stage33 numerical values are claimed until dedicated Actions execution succeeds and original resulting HDF5 passes this audit.
