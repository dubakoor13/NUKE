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
