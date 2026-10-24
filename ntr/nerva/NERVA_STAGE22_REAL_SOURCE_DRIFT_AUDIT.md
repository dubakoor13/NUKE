# Stage 22 — Real four-seed OpenMC source-bank concentration and checkpoint drift

**Date:** 2026-10-25
**Repository:** https://github.com/dubakoor13/NUKE
**Parent:** Stage 21 PR #134; branch `feature/nerva-real-source-checkpoint-drift-stage22-20261025`
**Original evidence:** GitHub Actions [run 39933618638](https://github.com/dubakoor13/NUKE/actions/runs/39933618638), artifact 12492499186, genuine HDF5 files produced by OpenMC 0.15.3.
**ZIP SHA-256:** `cf05ab907bee76b198d3dc4736a84c6d560f3a8c2d753b78b5605c4826e53274`.
**Provenance limitation:** GitHub reports this original artifact expires on **2026-11-13**. A pinned SHA is necessary but not sufficient to preserve a long-term copy.

## Why another source-checkpoint audit matters

Stage 21 independently authenticated four final native HDF5 statepoints at batch 120. A final k value and small apparent uncertainty **cannot establish fission-source stationarity**, particularly in a highly subcritical, strongly cloned source population. NUKE's existing `source_bank_convergence.py`, `source_bank_trajectory.py`, `source_resolution_audit.py`, `source_clone_moment_closure.py`, and `real_core_keff_history_qualification.py` already provide separate observational algorithms; Stage 22 **does not replace them**. Instead it reopens and independently verifies **all twenty exact HDF5 checkpoint files** from the archived real transport execution, including their histories and spatial source distributions, and requires agreement with the original summarized observations.

The audit is strictly observational. It does not alter fuel loading, geometry, enrichment, cross sections, control drum position, source particle settings or reactor criticality.

## Actual HDF5 observations

Each of the four RNG seeds 13007, 23011, 33013, and 43019 saved OpenMC `statepoint.040/060/080/100/120.h5`, for **20 separate genuine statepoints**. Each native saved fission source bank contains **3000 weighted records**, but exact repeated source-site positions are abundant.

| Quantity | Observed |
|---|---:|
| Four independent RNG seeds | 4 |
| Saved HDF5 checkpoints per seed | 5 |
| Native binary statepoints checked | 20 |
| Recorded source-site records per snapshot | 3000 |
| Exact distinct source positions per snapshot | 58–96 |
| Inverse-Simpson effective **position** count | 20.244–32.321 |
| Mean within-seed adjacent source total variation at batch intervals 40→60, 60→80, 80→100, 100→120 | 0.72908, 0.66500, 0.71075, 0.70133 |
| Mean between-seed spatial total variation at batch 40, 60, 80, 100, 120 | 0.69572, 0.62656, 0.66639, 0.68172, 0.68572 |
| Mean between-seed Jensen–Shannon divergence (nats) | 0.37086, 0.33338, 0.35489, 0.37249, 0.36178 |

All spatial TV/JS results use the SAME declared 4 × 4 × 4 cell-wise source histogram and a matched physical scoring envelope for every seed/checkpoint. Jensen–Shannon divergence uses actual discrete probability fractions with no pseudocounts. A point's exact weighted count is a **location-multiplicity observation**, not an independently sampled neutron or fission genealogy.

## Native binary evidence checks

`ntr/nerva/real_source_checkpoint_drift.py` first invokes the Stage 21 final-statepoint real-data gate; it then requires:

1. The actual input HDF5 bytes match each original checkpoint `statepoint_sha256` in the saved multi-checkpoint provenance report. Missing or altered files fail.
2. Every native HDF5 is eigenvalue mode with source bank, correct original RNG seed, batch, inactive-batch count, particle count and active realizations.
3. Native `k_generation` and `entropy` prefix arrays are EXACTLY persistent as the snapshots progress through consecutive saved batch indices, checking checkpoint continuity.
4. Every weighted source-bank site has valid finite spatial coordinates, positive finite weight, lies within the unchanged source scoring envelope and conserves the observed 3000-record bank.
5. Unique exact positions, largest replicated position weight share, inverse-Simpson position concentration, spatial centroid and spatial entropy are independently recomputed, not copied from JSON; they must match the original archived observations.
6. Within-seed adjacent TV and between-seed final/checkpoint TV are directly reconstructed from actual weighted native bank histograms, and must match the historical archived source trajectory results. Jensen–Shannon is computed independently on all six between-seed pairs at each saved batch.
7. Explicit **source stationarity / independent genealogies NOT QUALIFIED** and **HOT NERVA NOT QUALIFIED** outcomes remain unchanged regardless of a successful numerical integrity audit.

The source TV total variation distance is `TV(p,q) = 0.5 * sum_bin |p_bin−q_bin|`. The JS information divergence is `JS(p,q) = 0.5*KL(p||m)+0.5*KL(q||m)`, where `m=(p+q)/2`. These describe source-shape differences and **are not p-values, statistically valid independent-particle uncertainties, or reactor qualification thresholds**. Severe source-site duplication indicates high concentration, but it alone cannot numerically estimate independent family genealogy count or certify how many source iterations would repair the model.

## Important apparent contradiction resolved

The original four-seed summary reports `k_eff≈0.011698` with formally small SEM. The separate eigenvalue-generation history can show stable-looking k, but the actual bank snapshots can STILL have strong between-seed and temporal spatial drift and fail the source-cloning screen. **Do not use formal k SEM to overrule a failed spatial fission-source stationarity gate.** The very low k is explicitly the observed output of a far-subcritical cold NERVA-derived *surrogate*; it is not a criticality target, and this stage does not modify the configuration to approach one.

The HDF5 source-bank histories are within-seed genealogically correlated. Four independent seeds provide several separate trajectories, but checkpoint snapshots within the same seed are not statistically independent generations and may be heavily resampled/cloned.

## Tests and reproduction

```bash
python -m pip install numpy h5py
python -m ntr.nerva.real_source_checkpoint_drift \
  /path/to/nerva-real-openmc-four-seed-core-evidence.zip \
  --output build/real_source_checkpoint_audit.json

NUKE_REAL_CORE_EVIDENCE_PATH=/path/to/nerva-real-openmc-four-seed-core-evidence.zip \
  python -m ntr.nerva.validate_real_source_checkpoint_drift
```

`.github/workflows/nerva-real-source-checkpoints.yml` downloads the original archived 2026-10-23 ZIP from GitHub, verifies its **whole-artifact pinned SHA-256**, reruns the native twenty-HDF5 audit, and uses real evidence for regression and corruption-detection tests. No synthetic statepoint is substituted if the source archive is unavailable. A workflow failing because the source artifact expires is an actual evidence-preservation issue, not permission to replace real transport with test doubles.

## Physical work remaining

- Preserve the real transport archive **before GitHub Actions artifact expiration 2026-11-13** (master TODO EVIDENCE-02). Keep original bytes and SHA; the small Stage 22 JSON is **not** a replacement for HDF5.
- Obtain independent fission-source convergence under a justified reference case, with spatial source shape, autocorrelation/clone genealogies, independent seeds, and statistically supported stopping criteria (TRAN-02 / EVIDENCE-03).
- Qualify temperature-, density- and composition-dependent nuclear data, actual material and thermal-scattering coverage in a named historical/declared core configuration (DATA-01–05 / EVIDENCE-04).
- Score material-disjoint, correctly normalized neutron **and** photon local energy deposition using real transport (HEAT-01–05 / EVIDENCE-05). The cold neutron-only archive gives zero usable `heating-local`.
- Independently compare genuine transport and thermal output with source-traceable NASA measurements before claiming historical NERVA reactor fidelity.

**Stage 22 acceptance: independently rechecked native source checkpoint observations, NOT validated reactor criticality, hot operation, heating or NASA historical fidelity.**
