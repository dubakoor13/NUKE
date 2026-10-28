# Stage 35 — genuine coupled-neutron/photon 3D heating stability and mesh sensitivity

**Historical operating NERVA validation: NOT ESTABLISHED.** This stage does **not** alter the surrogate reactor geometry, fuel, enrichment, control drums, nuclear data, or transport algorithm. Its job is to test whether genuine computed *spatial deposited heating* is stable enough to contemplate subsequent thermal coupling.

## Original executed physical evidence

- Stage33 successful original [OpenMC C++ 0.15.3 four-seed 12-StatePoint execution](https://github.com/dubakoor13/NUKE/actions/runs/40741032756): RNG seeds 13007/23011/33013/43019, actual neutron+photon transport, source generations 30, 45 and 65, 20 inactive, 3,000 starting particles/generation, and real evaluated source neutron/photon cross sections.
- Stage34 independently successful [actual disjoint active-generation heating](https://github.com/dubakoor13/NUKE/actions/runs/40742640556): three **nonoverlapping** active windows 21–30 (10 generations), 31–45 (15), 46–65 (20). Native mesh/material/neutron+photon+electron+positron energy tallies conserve for each original seed/window.
- **Observed Stage34 four-seed average heat deposition:** early 908558.831369155 eV/source, middle 850473.5911782507, late 886599.714488517; mean early-to-late axial *fraction* total variation 0.16460848172497028. These are actual cold-surrogate measurements, NOT reactor watts or hot NASA NERVA conditions.

## New actual 3D spatial numerics

Reopen all 12 HDF5 files through the **official pinned OpenMC v0.15.3 StatePoint reader**, validate original seed/filter/mesh/estimator/energy and score identities, and reconstruct the 4 × 3 × 12 × 12 × 10 array of nonoverlapping native mesh heating means from cumulative raw tally `sum` and `sum_sq`. Recompute and independently match each window's whole-mesh heat and integrated axial heat against the successful archived Stage34 report.

The physical mesh bounds (cm) are retained and used to compute **cell volume**. Saved fields include eV/source/voxel, eV/source/cm³ and normalized deposited-heat spatial fractions. No simulated-source eV is converted to W without independent physical neutron source strength.

For each active window and seed compute normalized 3D deposited-heat *shape*; report pairwise early–middle, middle–late and early–late total-variation distances. In parallel compute all six between-seed spatial-distribution total-variation distances for each window, plus observed spatial variance/hot-voxel activity across just four independently seeded calculations.

**Mesh coarsening test:** conservatively integrate exactly the same **physical** OpenMC deposition without interpolation at spatial resolutions 12×12×10 → 6×6×10 → 3×3×10 → 1×1×10. Since coarse bins are physical unions of fine bins, total deposited energy must be conserved and `TV(coarse) <= TV(fine)` for each time-window pair and each seed pair. This is a deterministic total-variation/data-processing inequality and is a strong, falsifiable indexing/numerics check. The final 1×1×10 profile must reproduce the *original independently reported Stage34 axial TV*, not a newly selected reference.

**Don't confuse these observables:** Stage33's source-bank TV 30→65 = **0.6015** measures last-generation **fission-source site distribution**; Stage34's axial heating TV = **0.16461** measures the *active-window average local deposited energy* after integrating X/Y. They are not the same field and have different time and particle ancestry interpretations.

## Test criteria

The dedicated Stage35 Actions downloads the **real** original Stage33 12-HDF5 archive and independent successful Stage34 JSON. Both are mandatory. No generated or mocked reactor transport data are accepted as physical tests.

- Eleven adversarial pure numerical tests (normalization, conservation, TV=0/1 extremes, coarsening inequality, invalid/nonfinite fields, fail-closed original-HDF5 evidence).
- Two mandatory real original-HDF5 tests re-opening all native statepoints, verifying finite and physically normalized 3D deposited-energy fields, independent axial conservation, and explicit nonqualification of physical reactor watts.
- Archive machine-readable JSON and compressed `npz` with original full physical 3D fields (all four seeds and all three disjoint windows).

The underlying cold ~0.012-k-effective surrogate remains **far subcritical**. Four RNG seeds, limited generations, genealogical correlations, sparse high-variation voxels, absence of validated hot-temperature evaluated cross sections and historically matched NERVA hardware prevent any criticality/real-W/source-stationarity/thermal-failure confidence claims. The spatial variation diagnostics can inform *what needs more sampling*, not certify acceptable heating.

## Next physics gates

Use larger independent neutron-source populations and multiple initial source-shape choices on the same fixed historical reference before any confidence claims; investigate neutron population clustering and inactive-batch sensitivity; independently establish reactor-state nuclear-data/geometry traceability before coupling power density to fuel and H₂ thermal limits.


## Executed original 3D OpenMC evidence — 2026-10-28

**Dedicated genuine HDF5 [workflow 40743595418](https://github.com/dubakoor13/NUKE/actions/runs/40743595418) COMPLETED SUCCESS** on tested SHA `1134560f27cdabe367d526ef3753ceab625a447a`. The workflow fetched **12 original C++ OpenMC v0.15.3 coupled neutron/photon statepoints** and the independent successful Stage34 original heat-window report, then verified all native 3D active-window mesh integrals and the unchanged original axial heat. **11/11 adversarial numerical tests and 2/2 mandatory real-native-HDF5 tests passed**.

**Actual observed mean across four independent seeds of early[21–30] vs late[46–65] deposited-heating *spatial fraction* total variation:**

| Conservatively integrated heating grid (x×y×z) | Observed mean TV |
| --- | ---: |
| 12×12×10 (finest recorded grid) | 0.2248836445788241 |
| 6×6×10 | 0.19200733166251294 |
| 3×3×10 | 0.17573591410546055 |
| 1×1×10 (axial) | 0.16460848172497022 |

All numbers are original OpenMC simulated heat-distribution-shape differences **not** temperature, reactivity, power-density in watts, or a source-convergence p-value. The coarsening monotonicity and independent archival Stage34 0.16460848172497028 axial-TV check passed. The 3D array of eV/source/voxel, eV/source/cm³ and normalized fractions from all 12 disjoint seed/win combinations is archived as **NPZ** alongside descriptive JSON in workflow artifact **12840232993**, `nuke-stage35-original-openmc-3d-spatial-heating-stability` (reported expiration 2026-11-18). Retain/checksum the full original Stage33 12-HDF5 archive separately for long-term reproducibility.

**Physical interpretation:** noticeable spatial deposited-heat shape variation remains even after integrating x/y into an axial profile, and variation increases when genuinely resolved x/y features are retained. This does not distinguish fission-source temporal drift, neutron/photon tally noise, spatial estimator variance and other numerical effects. The cold far-subcritical configuration with limited 45 active generations **cannot yet supply qualified 3D hot-NERVA fuel temperature/heat-loading inputs**.
