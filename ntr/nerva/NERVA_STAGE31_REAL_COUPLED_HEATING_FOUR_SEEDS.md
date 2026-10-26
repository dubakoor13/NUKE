# NUKE Stage 31 — executed four-seed coupled neutron/photon heating and correlated energy spatial uncertainty

**Date:** 2026-10-26  
**Parent:** [Stage 30 PR #144](https://github.com/dubakoor13/NUKE/pull/144) — successful actual OpenMC0.15.3 cold 3-ring coupled neutron/photon/electron/positron energy-deposition closure.  
**Goal:** execute *four genuinely different* coupled neutron/photon source replicas on the same exact geometry and evaluated data, then calculate **joint per-seed heating covariances** rather than making a statistical independence assumption about correlated mesh bins.

## Original physics execution

The dedicated [Stage31 workflow](https://github.com/dubakoor13/NUKE/blob/feature/nerva-coupled-heating-four-seeds-stage31-20261026/.github/workflows/nerva-stage31-coupled-four-seed.yml) pulls the original OpenMC 0.15.3 image and runs its genuine C++ coupled executable four independent times with RNG seeds **13007, 23011, 33013, 43019**. Each case uses the same **cold 3-ring NERVA-derived surrogate** (3000 starting neutrons/gen, 65 total generations/batches, 20 inactive, 45 active, 12×12×10 energy deposition mesh, 24 diagnostic neutron energy bins, hydrogen temperature 293.6059129982851 K). No changes to material enrichment, geometry, control drums, neutron XS, solver transport methodology, or reactor operating power.

The Stage30 native HDF5 reader was extended to preserve the **actual OpenMC expanded XYZ `heating` mesh shape**, its physical bounds, and energy integrated separately over each of 10 native axial slices. Each independent seed's `statepoint.65.h5` is reopened by the **official installed OpenMC high-level reader**, never by a fake source. For every seed the workflow requires:

- Original actual native `photon_transport=True`, appropriate eigenvalue tally mode, correct seed and active batch count.
- Recompute native statepoint SHA-256; check against that seed's transport provenance and input XML SHA-256.
- Same geometry/material/tallies XML file hashes **and** unchanged nuclear-data index, photon-element HDF5 hashes, declared model configuration (apart from seed), mesh bounds and dimensions, and filtered material ID ordering across all four seeds.
- The actual whole-core heating eV/source mean must close to `1e-10` relative across native mesh, all material bins and the four scored particle types: **neutron, photon, electron, positron**.
- Preserve each seed's native HDF5 statepoint and `summary.h5`, original exported physics XML, nuclear-data report and transport provenance alongside derived covariance JSON.

## Statistical computation

For four paired observed heating vectors `x_s` from independent seed runs, with the same component ordering and spatial geometry, compute sample covariance

```text
C_ij = [ sum over seeds s of (x_si - mean_i)(x_sj - mean_j) ]/(4 - 1)
Var_across_seeds(sum over bins i of x_i) = 1^T C 1
```

This is computed separately for the four particle-type bins, native material partition bins, and 10 actual axial heating bins. The entire observed between-seed covariance, including **off-diagonal cross-bin covariance**, is retained. It is cross-checked against the directly computed variance of per-seed totals. In particular, individual bins may fluctuate substantially even when their **sum is nearly constant**. It would be physically/numerically wrong to sum their marginal conditional error bars in quadrature.

The sample covariance's rank is at most **3**, because only 4 independent seed vectors are available. The reported `sample_SD/sqrt(4)` is a **descriptive seed-mean statistic under a hypothetical stationary-source assumption**: it is not calibrated error coverage for the present highly subcritical, source-cloning model, and cannot determine thermal hot-spot acceptance or fuel failure margins.

## Test gates

- `validate_stage31_coupled_heating_ensemble.PairedSeedStatisticsMathTests`: 8 numerical/adversarial synthetic vector math tests, including anticorrelated bins with exactly zero variance in their total, correlated components that amplify total variance, sum-of-covariance identity, negative/NaN rejection, insufficient replicas, empty bins and degenerate all-zero heat.
- `validate_stage31_coupled_heating_ensemble.ActualOriginalCoupledOpenMCFourSeedTests`: 2 mandatory tests against the **four new original OpenMC HDF5 statepoints**; no mock fallback.
- Dedicated GitHub Action must run the physics first; passing math-only tests is not physical evidence.

## Limitations / next actual physics milestones

This iteration does NOT claim:
1. Spatial fission-source stationarity or physical source-bank independence from fission genealogical cloning; independent RNG seeds are insufficient by themselves.
2. Reactor operating criticality: the preceding Stage30 cold model had nominal `k_eff ≈ 0.01208`, not ~1.
3. Evaluated nuclear reactions across actual hot NERVA temperature/pressure/density state, validated bound scattering, chemical phase and fuel loading.
4. Full photon escaping-energy balance and radionuclide release, or physical power in watts. Energy remains **eV per Monte Carlo source**.
5. Qualified full-core thermal fuel/gas mapping, coupled hydrogen enthalpy/pressure feedback, converged spatial mesh uncertainty, or NASA historical reactor benchmarks.

**At code authoring:** real Stage31 HDF5 results must be separately confirmed by the new Actions run; this document does not predeclare the data as passed. This is an actual core transport/numerics continuation, not CI-only dashboard work.


## Executed genuine four-seed HDF5 verification result (2026-10-26)

**[Dedicated original OpenMC coupled HDF5 workflow 40325725662](https://github.com/dubakoor13/NUKE/actions/runs/40325725662) COMPLETED SUCCESS**, at checked head `e2b8ee953d822adca529ae9bf3bcd10027cf2a91`.

This is actual OpenMC 0.15.3 C++ executable neutron+photon transport, **not synthetic reactor output**. The official source image and evaluated NNDC neutron/photon cross sections were used. Each original HDF5 statepoint and its actual exported model XML were retained, and all four statepoints were independently checked for identical nuclear-data index, photon isotope HDF5 hashes, geometry/material/tally XML hashes, mesh dimensions/material bin order and distinct source RNG seeds.

**Actual execution acceptance:** 8/8 adversarial paired-covariance numerical tests and 2/2 tests reopening and physically checking the four *new actual coupled HDF5* passed. All four `heating` material / XYZ mesh / neutron+photon+electron+positron energy integrals conserved to the configured native tolerance. Native axially expanded mesh heating and true paired between-seed cross-bin sample covariance also reconciled to directly observed per-seed totals.

| Independent OpenMC seed | Combined k-effective (nominal per-run SEM) |
|---|---|
| 13007 | 0.01208 ± 0.00021 |
| 23011 | 0.01173 ± 0.00019 |
| 33013 | 0.01201 ± 0.00020 |
| 43019 | 0.01192 ± 0.00026 |

- **Native mean total coupled neutron/photon heating across four actual seeds:** **879437.4771363477 eV per starting Monte Carlo neutron**.
- **Observed between-seed sample SD of total deposited heating:** **10230.500892604987 eV per starting Monte Carlo neutron**.
- **Corresponding sample mean SEM under a hypothetical stationary independent source assumption:** **5115.2504463024935 eV/source**; this is **NOT** a scientifically calibrated confidence limit because the far-subcritical, highly cloned fission source has not passed stationarity/independence tests.
- The generated JSON stores all four individual original statepoint SHA256 hashes, raw particle-wise heating, spatial material partition statistics, axial-bin means and their **paired** covariance matrices; **do not** replace these with quadrature-summed per-bin standard errors.
- Archived original generated HDF5 evidence: GitHub Actions artifact **12651167528**, `nuke-stage31-original-four-seed-coupled-heating-hdf5`, **3,649,821 bytes** with reported expiration **2026-11-16**. Saved archive ZIP SHA256 `a6aab0dd3fd9abd092b5aa7d3d29ff587b560f0440a54f04e9fee03d505f9b14`.

**Outcome:** a new genuine four-seed **coupled neutron + photon + locally deposited charged-secondary Monte Carlo** dataset exists and its source-normalized material/axial/particle energy bookkeeping and paired seed covariance are verified.

**Major physical blockers remain:** this cold conceptual reactor still has nominal k-effective ~0.012 rather than a qualified operational critical configuration; fission-source bank stationarity is not demonstrated; evaluated hot fuel / hydrogen / reflector interactions and thermal-scattering coverage, real material-resolved hot watts, fuel-channel coupling, mesh sensitivity, independent historical benchmarks and true operating source normalization remain unresolved. This validation must NOT be described as historical NERVA performance.

**Broader CI caveat:** the dedicated original four-seed workflow is independently SUCCESS; other repository checks or artifact-update triggered runs are not assumed passed unless individually verified.
