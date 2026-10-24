# NUKE Stage 23 — real OpenMC spatial-source resolution sensitivity

**Date:** 2026-10-25  
**Branch:** feature/nerva-real-source-multiresolution-stage23-20261025  
**Parent:** Stage 22 PR #135  
**Scope:** Read-only spatial fission-source evidence; **not an implementation of new transport physics and not a qualified historical NERVA reactor**.

## Source and authenticity

The exact real neutron-only OpenMC 0.15.3 cold-core evidence originated in GitHub Actions [run 39933618638](https://github.com/dubakoor13/NUKE/actions/runs/39933618638), artifact ID 12492499186, original ZIP SHA256 \`cf05ab907bee76b198d3dc4736a84c6d560f3a8c2d753b78b5605c4826e53274\`. Twenty original HDF5 eigenvalue source banks were saved at batches 40/60/80/100/120 for each of four independent RNG seeds 13007/23011/33013/43019.

Stage 21 verifies the independently executed HDF5 source provenance and reported native k; Stage 22 verifies ALL twenty original HDF5 file digests, saved k/entropy prefixes and positional source concentration. Stage 23 **reuses those strict gates** and independently recomputes the sensitivity of actual, physically weighted source-shape differences to spatial scoring resolution.

Do not confuse a genuine observed OpenMC source bank with a validated NERVA reactor state. The initial archive is a highly subcritical cold NERVA-derived surrogate with missing fission-source stationarity, hot evaluated neutron data and usable neutron/photon local heat scores.

## Why source grid resolution matters

A source bank contains 3000 fission-source records but only 58–96 **exact distinct positions** per snapshot; the 20.244–32.321 inverse-Simpson numbers are **effective position** counts and must NEVER be labelled statistically independent neutron trajectories or genealogies.

If these source sites are scored on a fine Cartesian grid, most fine voxels contain no source sites; a naive between-run total-variation metric approaches its maximum of 1 even when coarse source variation is smaller. At the same time, coarse-grid variation that remains large means simply coarsening the histogram does not erase observable spatial discrepancies. Stage 23 reports **both** and explicitly refrains from assigning a p-value or valid source-convergence probability.

## Actual implemented numerical comparisons

- The same weight-conserving spatial source probability is first binned at 16×16×16 over the ORIGINAL archived physical scoring envelope. It is then **exactly rebinned by integral summation** into 8³, 4³ and 2³ Cartesian voxels; NO interpolation, arbitrary pseudocount, or renormalization is used.
- The probability conservation relation is \`sum_cube p = 1\` under every coarsening.
- For any two normalized source distributions p and q, \`TV(p,q)=0.5*sum |p−q|\`, and \`JS(p,q)=0.5*KL(p||m)+0.5*KL(q||m)\`, m=(p+q)/2. By data processing, both TV and JS MUST NOT INCREASE as spatial probabilities are coarsened. This is enforced in both pure numerical tests and measured real HDF5 outputs.
- Independently compare all six between-seed pairs at each batch, and all four within-seed trajectories across the four checkpoint intervals on every spatial resolution.
- Additionally project source positions onto a physical **cylindrical radius r** from the declared core axis and axial coordinate z. Report radial and axial 8-bin differences separately. Radial projection integrates azimuthal structure and is NOT equivalent to validating a true axisymmetric diffusion model.
- Reject missing/changed original HDF5 files and mismatched native source identity using the complete Stage 21+22 binary provenance chain; do not silently substitute a randomly generated source bank.

## Measured real-source results

*Mean across all six between-seed pairs at the batch-120 checkpoint.*

| Spatial grid | Mean between-seed TV, final checkpoint | Interpretation |
|---|---:|---|
| 2×2×2 | ~0.36378 | Material coarse source shape still differs |
| 4×4×4 | ~0.68572 | Reproduces prior Stage 22 archived source-shape variation |
| 8×8×8 | ~0.93194 | Spatial support is increasingly sparse |
| 16×16×16 | ~0.98822 | Almost disjoint fine-voxel distributions; high sparsity |

At 16³, averaged across all snapshots, roughly 98% of 4096 bins are empty. At 2³, almost all bins are occupied. These statistics show extreme **resolution dependence**; assigning a single, unqualified “source TV” value without mesh and source genealogical information would be misleading.

For the same final HDF5 checkpoint, eight-bin **cylindrical radial projection** mean between-seed TV is approximately **0.27306**. The independent eight-bin **axial projection** TV is approximately **0.30967**. These projections reveal that nontrivial source disagreement persists even after aggregating the azimuthal/spatial information into core-relevant coordinates, but they cannot uniquely separate source convergence from strong cloning noise or prove physical reactor geometry.

The previously archived 4³ source TV values for batches 40,60,80,100,120 were (0.69572222, 0.62655556, 0.66638889, 0.68172222, 0.68572222), and Stage 23 must reproduce each to within rounding tolerance.

## Scientific limitations and interpretation

1. **Still no source-stationarity demonstration.** The 3000 source records are not independent site genealogies; massive exact-source-position replication persists, and stored checkpoints from a given seed are correlated.
2. **No justified p-values or formal confidence intervals.** Total variation, Jensen–Shannon information, occupancy and projection scores are descriptive measurements of this finite saved bank, not a known distribution of a stable physical eigenmode.
3. **Finite-scoring resolution biases a metric.** Coarser grids reduce TV and JS by a provable mathematical inequality. This is not an improvement to actual neutron physics or evidence of newly converged transport.
4. **Cold, far-subcritical source physics only.** No change to fuel, geometry, neutron populations, source algorithm, XS, enrichment or criticality is implemented in this stage. The parent archive had \`k_eff≈0.011698\`; no physically qualified NERVA core is established.
5. **Do not replace real source-shape evidence by a convenient scalar.** Spatial and source history checks must accompany any future transport benchmark, with independently justified source convergence and fission-site ancestry where obtainable.

## Code, tests, and reproduction

- \`ntr/nerva/real_source_multiresolution.py\`: independent physical 2³/4³/8³/16³ source comparison, exact coarsening TV+JS contraction and axial/radial source projections.
- \`ntr/nerva/validate_real_source_multiresolution.py\`: exact histogram probability math, strict bad-input guards, and genuine original twenty-HDF5 numerical observations.
- \`.github/workflows/nerva-source-multiresolution.yml\`: official GitHub Actions cross-run download of the exact original 2026-10-23 OpenMC artifact; no generated substitute. Reports source statistics in small JSON, with native HDF5 files authenticated via parent SHA checks.

\`\`\`bash
python -m pip install numpy h5py
python -m ntr.nerva.real_source_multiresolution /path/to/original_extracted_openmc_artifact \\
    --output build/real_source_multiresolution.json
NUKE_REAL_CORE_EVIDENCE_PATH=/path/to/original_extracted_openmc_artifact \\
    python -m ntr.nerva.validate_real_source_multiresolution
\`\`\`

## Prioritized NEXT scientifically consequential work

- **P0 / EVIDENCE-02:** Preserve exact original October 23 HDF5 artifact with immutable SHA and evaluated XS manifests beyond GitHub Actions expiration on **2026-11-13**.
- **P0 / TRAN-02:** Independently source-stationary physically motivated reference transport ensemble and **source genealogy** diagnostics. Do not claim that increasing the number of batches by itself resolves highly cloned source-bank instability.
- **P0 / DATA-01–05:** Material-resolved and temperature-/density-qualified nuclear cross sections and thermal-scattering kernels for the selected actual reference core.
- **P0 / HEAT-01–05:** Genuine neutron+photon local heating with valid source-rate/physical-power normalization and material-disjoint thermal volume projection.
- **P0 / REF-01–05:** Historically traceable NERVA test geometry/material/coolant/measurement records and separately held-out experimental benchmark data.

**Outcome of this stage is quantitative evidence about the limitations of existing cold neutron source banks; NOT a new accepted hot-core transport, reactivity or heating prediction.**

## Verified Stage 23 CI

The final dedicated source-evidence [Actions run 40136269625](https://github.com/dubakoor13/NUKE/actions/runs/40136269625) **completed successfully: 9/9 tests passed, zero skips**. Official GitHub Actions download resolved the original 2026-10-23 real OpenMC archive. Stage 21/22 checked the real native HDF5 source-bank integrity before this stage's multiresolution physics observations were recomputed. The CI log reported original batch-120 mean between-seed TV: 0.363778 on 2³, 0.685722 on 4³, 0.931944 on 8³ and 0.988222 on 16³. The parent conclusions remain: actual source stationarity, independent genealogical uncertainty, hot evaluated XS, real neutron/photon local heating and historical NERVA qualification are NOT established.
