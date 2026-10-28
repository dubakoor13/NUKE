# Stage 32 — Verify original coupled OpenMC final-generation fission source banks

**Basis:** [Stage 31 PR #145](https://github.com/dubakoor13/NUKE/pull/145), which executed and checked four distinct OpenMC v0.15.3 neutron/photon transport histories, including native statepoints and 45 active batches of `heating` scores. This stage **does not tune the reactor** or generate synthetic substitutes for those original results.

## Why this matters

A `heating` tally is an average over the active batches. The final saved fission bank is only the **last-generation source snapshot**. Even four independently seeded, mutually consistent heating totals do not establish spatial source stationarity. Fission sites may be repeatedly copied from a small number of source ancestors in the highly subcritical cold surrogate, making the nominal bank population an unreliable indication of independent information.

The new read-only `real_coupled_source_bank_audit.py` opens the **same four original HDF5 files** as Stage 31. It checks each statepoint's raw SHA256 against the executed Stage 31 heating JSON, verifies RNG seed and last batch, and requires the native OpenMC structured `source_bank`. For each last-generation bank it measures exact distinct position count, largest cloned-position weight fraction, inverse-Simpson distinct-position count (a *diversity index*, **not** independent history count), centroid, Shannon entropy, and a common 4 × 4 × 10 spatial histogram. Then it computes all six observed cross-seed total-variation distances and the per-seed axial source-fraction profiles.

**This is an observational diagnostic, not a stationarity pass/fail test.** There is no same-seed early checkpoint in the archived Stage 31 artifact. Only a new time-sampled physical transport experiment could test within-seed source drift. The bank's sites are genealogically correlated, and the bank is not the active-batch averaged heating source distribution. No statistical p-value or calibrated source-independence confidence claim is made.

## Real-data execution

The Stage 32 GitHub workflow downloads the original **successful Stage31 workflow run 40326333031** artifact `nuke-stage31-original-four-seed-coupled-heating-hdf5`. The repo's `actions:read` permission is needed. It checks the original archive contents, runs the pure-math/adversarial tests, audits four actual native HDF5 source banks (without an OpenMC reactor rerun), and preserves the resulting JSON as an artifact. All four native HDF5 source banks are mandatory; missing banks/hashes cause failure, not fake replacements. The Stage 31 artifact expires on **2026-11-16**; a future archival workflow should republish or rerun originals with fixed checksums.

```bash
python -m unittest ntr.nerva.validate_stage32_coupled_source_banks.SourceBankSpatialMathTests -v
python -m ntr.nerva.real_coupled_source_bank_audit \
  --root build/stage31 --output build/stage32/real_final_source_bank_diagnostics.json
NUKE_STAGE32_ORIGINAL_FOUR_SEED_ROOT=build/stage31 \
  python -m unittest ntr.nerva.validate_stage32_coupled_source_banks.ActualOriginalCoupledFissionBankTests -v
```

**Scientific status:** cold low-enrichment three-ring NERVA-derived surrogate only; final-source spatial observations, heating in eV per starting Monte Carlo neutron, not a validated critical NERVA core, thermal reactor watts, a hot nuclear-data/coolant model, or NASA experiment comparison.

## Next experimental gap

Execute **early and late native fission bank checkpoints within each seed**, with *photon transport and identical evaluated data*, to quantify temporal source drift and fission genealogical cloning. Repeat with at least one independent source initialization and source-population/inactive-batch sensitivity. Only after those studies should stationary-source claims be considered. Historical hot-core material and experiment validation remain separate.


## Executed real-HDF5 acceptance — October 28, 2026

GitHub Actions [run 40740267218](https://github.com/dubakoor13/NUKE/actions/runs/40740267218) **COMPLETED SUCCESS** on audit-code commit `ad2654337f17d68e1a473bdfa85554cc0810a4cf` using the **original four real Stage31 native coupled-transport statepoints** from run `40326333031`. The workflow downloaded and SHA-checked the archived HDF5 binaries; 7/7 adversarial/mathematical tests passed, and both mandatory real-HDF5 qualification-guard tests passed. GitHub Actions artifact `12837864365` contains the 4-bank spatial diagnostic JSON.

**Observed mean pairwise final-bank spatial total-variation distance: 0.5860555555555554**, scored over the fixed 4 × 4 × 10 spatial histogram. This is a measured between-seed distributional difference, **not** a hypothesis-test p-value or proof of source convergence. The presence of large spatial variation warrants properly time-resolved bank stationarity/genealogy testing; do not infer calibrated uncertainty or operating-core validity from this number. Original transport was not rerun in Stage32: the Stage31 executed coupled HDF5 was reanalyzed.
