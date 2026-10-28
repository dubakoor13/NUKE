# Stage 42 — original OpenMC C++ fission-source genealogy without user scoring tallies

**Accepted original neutron/photon transport evidence:** [GitHub Actions 40756987058](https://github.com/dubakoor13/NUKE/actions/runs/40756987058). Stage42 extends the prior Stage38–41 generation-23 source-path divergence investigation while retaining the **same cold far-subcritical NERVA-derived demonstrator, fixed geometry, materials, temperatures, evaluated neutron/photon data and 3,000 MC source neutrons/generation**. It does not represent historical NASA hot NERVA core physics.

## Numerical experiment: isolate active user tally scoring

The original scored Stage41 runs used OpenMC v0.15.3, coupled neutron/photon transport, 1 requested OpenMP thread, 100 batches, initial fission source either `legacy-broad` or `central-compact`, RNG seed 13007 and either **20** or **40** inactive generations.

The new `--diagnostic-disable-user-tallies` flag affects **only OpenMC user score tallies**. It is disabled by default. Four **genuine C++ OpenMC 0.15.3 source-only reruns** were executed with identical physical/transport inputs; they retained full original neutron/photon transport, source bank, global multiplication-factor and entropy histories, but no user energy tallies (hence no source-only thermal-power result).

- Four original source-only neutron/photon OpenMC HDF5 archives: [Stage42 C++ source histories, artifact 12846765965](https://github.com/dubakoor13/NUKE/actions/runs/40756403944/artifacts/12846765965). The original C++ transport completed for all four source/seed/inactive cases. Initial workflow failure was a Python analysis variable lost to literal escaped newline characters in a comment; no neutron transport failed.
- Original Stage41 scored neutron/photon HDF5: [artifact 12845287445](https://github.com/dubakoor13/NUKE/actions/runs/40753528970/artifacts/12845287445).
- The successful verification workflow **reused** both archives without rerunning or replacing transport, verified actual original XML fingerprints, neutron/photon data index SHA256, fission source mode, inactive settings, source-bank and HDF5 identities, and presence of **zero user scoring tallies in the new native C++ StatePoints**. **Six adversarial numerical tests and two mandatory real HDF5 tests passed**, reading **16 original OpenMC HDF5**.

## Native physical transport results

| Native C++ configuration | Saved fission-source bank scored vs NO user tallies | Saved source entropy | Max absolute generation k discrepancy |
| --- | --- | --- | ---: |
| Broad initial source / 20 inactive | **Bit-for-bit identical** | **Identical** | **0** |
| Broad initial source / 40 inactive | **Bit-for-bit identical** | **Identical** | **0** |
| Compact initial source / 20 inactive | **Bit-for-bit identical** | **Identical** | **0** |
| Compact initial source / 40 inactive | **Bit-for-bit identical** | **Identical** | **0** |

**When there are NO user scoring tallies in EITHER run**, source bank differences between 20 and 40 inactive remain:

| Native source-only observable, 20 vs 40 inactive | Legacy broad | Central compact |
| --- | ---: | ---: |
| First substantive `k_generation` change, absolute difference >1e-8 | **Generation 23** | **Generation 23** |
| First original Shannon entropy history difference | **Generation 23** | **Generation 23** |
| Max absolute generation-k difference over 100 batches | **0.005220297921279387** | **0.0054485599255120944** |
| Full original 3,000-site fission-bank exact equality at batches 60,100 | **FALSE, FALSE** | **FALSE, FALSE** |
| Entropy sequence from all 100 generations equal | **FALSE** | **FALSE** |

Physical interpretation: **User scoring tallies are NOT necessary for the observed source genealogy divergence.** Turning user heat/mesh/material/particle tallies entirely off yields *exactly the same* native neutron fission-source histories for a fixed inactive count, while differing inactive counts cause differences around generation 23 even without any user tallies. It would be incorrect to claim that user tally activation alone consumes RNG and changes the source: this hypothesis is falsified for the tested OpenMC version and fixed model.

The inactive setting also affects **global** tally accumulation, batch normalization and reported `k`/statistics within the original OpenMC C++ solver. The reason for physical source-path divergence is **not established** without closely spaced original source-bank checkpoints around generations 20–24 and a source update RNG/weight accounting audit. Full source bank, entropy and k are distinct observables.

**No source stationarity, calibrated neutron genealogy uncertainty, independent hot-reactor physical watts, NASA NERVA configuration, true fuel temperatures or operating reactor qualification is inferred** from this cold small, highly subcritical (`k≈0.012`) conceptual core.

## Evidence

- [Successful actual-HDF5 Stage42 job 40756987058](https://github.com/dubakoor13/NUKE/actions/runs/40756987058): **8/8 tests passed**, four newly executed original coupled source-only C++ histories and four archived original scored runs.
- [Machine-readable verified original native source control JSON, artifact 12847280440](https://github.com/dubakoor13/NUKE/actions/runs/40756987058/artifacts/12847280440).
- HDF5 archive retention currently expires approximately **2026-11-18**; preserve HDF5 originals, original nuclear evaluated data and input SHA256 for future numerical reproduction.

**Next numerical gate:** retain generation-by-generation **original OpenMC source-bank HDF5 checkpoints at batches 19 through 30** with inactive 20 and 40 (no user tallies), verify the precise first native bank divergence and its timing relative to the solver's first-active batch. Do not change reactor geometry or fuel loading for that diagnostic. Follow with validation of a representative NASA core and temperature-resolved neutron/photon data before physical hot reactor claims.
