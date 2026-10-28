# Stage 34 — native OpenMC disjoint active-generation deposited heat

**Physical input:** archived [successful Stage33 real coupled OpenMC run 40741032756](https://github.com/dubakoor13/NUKE/actions/runs/40741032756) generated **12 original OpenMC C++ v0.15.3 HDF5 StatePoints** (four RNG seeds × batches 30/45/65). These statepoints were independently checked against the original four-seed native coupled neutron/photon heating ledger, with unchanged original nuclear data, mesh, materials, and source. The underlying cold highly subcritical NERVA-derived surrogate is **not** a validated operating NERVA reactor.

## New genuine numerical physics

OpenMC stores each tally's cumulative active-realization **raw SUM and SUM_SQ** in each native HDF5 checkpoint. Since the same code saved cumulative moments at 30, 45, and 65 with 20 inactive generations, those snapshots contain respectively **10, 25, and 45 active tally realizations**.

Rather than treating correlated cumulative means as independent or subtracting marginal standard errors, Stage34 reconstructs the exact *disjoint* active-batch moments:

```text
S[21:30] = S_cumulative[30]
S[31:45] = S_cumulative[45] - S_cumulative[30]
S[46:65] = S_cumulative[65] - S_cumulative[45]

window_average = delta_SUM / delta_active_realizations
window_batch_sample_variance = (delta_SUM_SQ - delta_SUM^2 / n)/(n - 1)
weighted_recovery = (10*mean_21_30 + 15*mean_31_45 + 20*mean_46_65)/45
```

OpenMC batch means share fission-source genealogy and **are not statistically independent**. The scalar per-bin sample batch variance is only a descriptive moment. There is no assumption that heating in adjacent bins, material partitions or particle channels is independent, and no per-bin SEM quadrature or confidence interval is inferred.

For each of the 12 genuine OpenMC checkpoints we reopen the installed official OpenMC `StatePoint` reader, compare native HDF5 physical identities and estimator/filter/bin ordering, and recover **3D energy-mesh, all material, and all four particle-type** `heating` raw cumulative moments. Each recovered disjoint window must conserve mesh heat vs all materials and neutron+photon+electron+positron to `1e-9` relative; cumulative raw sums and sum-squares must telescope to original final raw values. The weighted reconstructed total must match the independent Stage31 final native heat ledger of the same four runs. The score remains **eV per Monte Carlo starting source**, never reactor watts.

Then quantify genuine axial heating fractions independently for windows [21,30], [31,45] and [46,65], the first-to-last *heating* axial TV, window-specific deposited total and across-seed descriptive spread. Compare these observations alongside the existing Stage33 same-seed source spatial TV **without asserting that the spatial source and deposited-heat fields are identical or causally linked**.

## Validation and limitations

The dedicated Stage34 GitHub workflow downloads **the archived successful original Stage33 physical 12-HDF5 artifact**, not mock or reconstructed neutron transport. It runs 9 mathematical/adversarial tests and 2 tests that require actual native statepoints, using the pinned original OpenMC v0.15.3 reader.

The previous verified Stage33 spatial source-bank 30→65 mean total variation was **0.6015** on a 4×4×10 source grid; that value concerns source **bank positions**, not deposited heating. New Stage34 heat-window values must be obtained from real executed HDF5 and must not be imagined before the workflow succeeds.

This audit does not provide certified fission-source stationarity, independent Monte Carlo confidence coverage, physical NERVA criticality, hot-temperature cross-section coverage, spatial thermal feedback convergence, real reactor W, or comparison to NASA historical experiments.

**Next physics experiment after this read-only phase:** increase active source generations, vary initial source populations, record additional snapshots and investigate bias/stationarity for material-resolved deposition and flux before using source-normalized heating in thermohydraulic coupling.
