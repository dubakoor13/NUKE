# Stage 49 — verified full fission-source phase space and NERVA reference fidelity

**Status: original OpenMC C++ neutronics and all tests verified.** [Original GitHub Actions job 40812177940 — SUCCESS](https://github.com/dubakoor13/NUKE/actions/runs/40812177940) completed **eight adversarial source-mathematics tests and two mandatory original HDF5 neutron physics tests (10/10)**. Analysis reopened **all 16 unmodified genuine OpenMC v0.15.3 StatePoints** saved during Stage48: neutron eigenvalue banks at batches **60 and 100**, independent RNG seeds 13007, 23011, 33013, 43019 at physical neutron material temperatures 750 K and 900 K using the same evaluated multi-temperature nuclear library.

## New numerical physics — spatial clustering is NOT the same as repeated neutron states

OpenMC's native `source_bank` stores neutron birth **position** (`r`), **direction** (`u`), **energy** (`E`), time, particle weight, delayed group, surface id, and particle type. Stage48's spatial histogram found only **62–89 unique birth locations** in each 3,000-source-neutron bank. The Stage49 full-source inspection confirms **many particles share one birth location but differ in energy and/or direction**.

| **Original OpenMC C++ HDF5 source-state quantity** | **750 K** | **900 K** |
| --- | ---: | ---: |
| Independent seeds / native StatePoints | 4 / 8 | 4 / 8 |
| Saved source particle records per StatePoint | 3000 | 3000 |
| Exact distinct positions across 60/100 checkpoints | 62–89 | 73–88 |
| **Exact distinct complete source records across 60/100** | **2250–3000** | **2595–3000** |
| Exact distinct birth positions **at batch100** | **81–89** | **73–84** |
| Exact distinct complete neutron records **at batch100** | **2344–3000** | **2595–3000** |
| Maximum exact duplicated *complete* records in batch100 | 656 | 405 |
| Weight of birth source with E≥100 keV (mean at 100) | 98.6083% | 98.6583% |
| Mean original native combined `k_eff` across four seeds at batch100 | **0.010236730270972017** | **0.010150087679123048** |

**Important distinction:** Source particles at an identical location are **not** necessarily identical neutrons, even when sampled from the same parent fission site. Some original *complete* source records are exactly cloned; others at identical coordinates have distinct sampled direction/energy and/or other source fields. The resulting counts are **not** independent-fission-ancestor counts. Unique complete source records also do **not** prove independent genealogies. The original large source-bank total variation between generations 60 and 100 and across seeds (Stage48) still signals unresolved **spatial-source stability**; Stage49 refines, rather than invalidates, those measurements.

The energy fractions reported here characterize the **birth source sampled for eigenvalue generations**, **not an equilibrium track-length neutron-flux energy spectrum or fuel heating**. The code checks true HDF5 source record structure, unit direction norm, positive energies, positive weights, material and geometry XML SHA fingerprints, independent source seed, 100-generation provenance and genuine particle weight. Mathematical fixtures cannot qualify an original C++ source record without matching original HDF5 fields and SHA.

## Actual native neutron multiplication estimators

Stage49 independently reads original `k_combined`, the reported standard error and native collision-, absorption- and track-length-`k` estimators from the original OpenMC C++ `global_tallies` raw sum/squared-sum storage. It does **not** relabel the eigenvalue leakage score as a neutron leakage probability: in the actual source-normalized tallies the raw leakage estimator can exceed unity, and the physical neutron balance requires the solver's proper normalization, not a simplistic probability interpretation.

These native combined estimates remain `k_eff≈0.010–0.011` in this conceptual mini-core. **Independent fission-source convergence and physical operating criticality remain unqualified.**

## Historical geometry fidelity: why the surrogate cannot yet validate NERVA

The report also executes the repository's existing **source-backed reference discriminator**, `physical_core_reference_validation.assess_core_fidelity`, against `NervaConfig(core_rings=3)`.

- Public NASA heritage **fuel-element** data, 0.75 inch (1.905 cm) across-flats, **19 coolant bores per fuel element** and about **132 cm active element length**, are consistent with the modeled element-level nominal dimensions.
- **This is not a historically validated full reactor.** The actual current `assembly=core` is a three-ring **19-element conceptual all-fuel lattice**, without the documented historical tie-element population/core inventory.
- The separate **proposed, not-built SNRE** configuration uses approximately **564 fuel + 241 tie/support elements**, a different **88.9 cm** length and **14.7 cm** Be radial reflector. The code explicitly **rejects** comparing the three-ring mini-core with this full SNRE topology; neither the 132 cm heritage element match nor an isotope-temperature qualification establishes true reactor composition, atomic densities, criticality or energy release.
- The conceptual fuel matrix `C:Zr:U=0.90:0.08:0.02` atomic representation and its enrichment/density are not verified historical NASA composition. No claim of original NERVA chemical loading, corrected control worth or operating reactor construction is made.

Source identification: NASA NTRS [20150021278](https://ntrs.nasa.gov/citations/20150021278), [20180000405](https://ntrs.nasa.gov/citations/20180000405); existing repository [physical_core_reference_validation.py](physical_core_reference_validation.py) and [REFERENCES.md](REFERENCES.md). **Do not combine the distinct heritage element and proposed SNRE reactor parameters into one fictional NASA-validated geometry.**

## Corrected technical interpretation

**Stage48 established large changes in the neutron-source spatial distribution, not that 97–98% of complete source neutron states were identical.** Stage49's actual native source-bank field comparison demonstrates the latter interpretation is wrong. A neutron can share a fission birth location while differing in direction or energy.

Neither spatial source diversity, full phase-space diversity nor a nominal Monte Carlo particle count certifies uncorrelated fission genealogy or stationary eigenvalue-source statistics. The current `k≈0.01` model is deeply subcritical and its source distribution evolves dramatically on the observed generation interval.

The model still has no verified NASA historical full-core atomic/nuclear inventory, independently benchmarked neutron multiplication, realistic operating temperature bound scattering, coupled hydrogen EOS or energy-conserving neutron/photon thermal feedback. **This Stage49 audit is read-only and changes no reactor design or physical transport.**

## Original reproducible evidence

- [Stage49 original 16-StatePoint neutron physical phase-space audit — **10/10 PASS**, Actions 40812177940](https://github.com/dubakoor13/NUKE/actions/runs/40812177940).
- [Actual source-state physical diversity, original native k and reference fidelity JSON, artifact 12870438667](https://github.com/dubakoor13/NUKE/actions/runs/40812177940/artifacts/12870438667).
- [Unmodified source data: Stage48 original 16 genuine OpenMC C++ neutron HDF5 and XML provenance, artifact 12852585389](https://github.com/dubakoor13/NUKE/actions/runs/40767503400/artifacts/12852585389).

**Next numerical/physics priority:** source genealogy should be characterized by actual parent/descendant lineage where available (not inferred from repeated positions); validate against a physically justified historic reference first. Do not attempt to force `k_eff` upward through guessed enrichment, geometry or material changes and then call it historical NERVA.
