# NUKE Stage 21 — VERIFIED REAL OpenMC cold-core archive, physical blockers, and provenance audit

**Date:** 2026-10-25.  
**Source repo:** https://github.com/dubakoor13/NUKE.  
**Source GitHub Actions run:** https://github.com/dubakoor13/NUKE/actions/runs/39933618638 (completed successfully 2026-10-23).  
**Source artifact:** `nerva-real-openmc-four-seed-core-evidence`, ID `12492499186`, https://github.com/dubakoor13/NUKE/actions/runs/39933618638/artifacts/12492499186.  
**Original ZIP SHA-256:** `cf05ab907bee76b198d3dc4736a84c6d560f3a8c2d753b78b5605c4826e53274`.  
**Artifact expiration reported by GitHub:** 2026-11-13. Preserve an immutable externally managed copy and manifest before expiration; Stage 21's pinned artifact CI will fail closed once unavailable.

## Correction of earlier inventory

An earlier master TODO accurately noted that there was no NERVA statepoint checked into the Git source tree, but **incorrectly implied that executed NERVA-derived transport evidence had not been located**. The repo in fact has genuine four-seed OpenMC C++ cold-core transport results archived in GitHub Actions. The source run and the original binary HDF5 files have now been inspected; the archive was retrieved, its exact SHA verified, and per-seed OpenMC HDF5 `k_combined`, seed, batch, inactive, history population and version were independently read.

This finding establishes a successful actual OpenMC execution on a **cold, NERVA-derived surrogate core**, not a historically valid critical NERVA reactor. A successful GitHub workflow is not a pass for hot transport, source stationarity or nuclear heat.

## Unmodified actually observed neutron criticality data

| Independent RNG seed | Executed k_eff | OpenMC reported SEM |
|---:|---:|---:|
| 13007 | 0.01181822339995014 | 0.00015007440099844432 |
| 23011 | 0.01163544 (approximately) | 0.00015064 (approximately) |
| 33013 | 0.01169437 (approximately) | 0.00016461 (approximately) |
| 43019 | 0.01164200 (approximately) | 0.00015142 (approximately) |

The archived ensemble JSON reports **inverse-variance mean `k_eff = 0.01169802857789661` and formal SEM `7.69260125465282e-05`**, and it explicitly says the latter is a formal estimate **not a source-convergence certificate**. Four statepoints independently inspect as **OpenMC 0.15.3**, neutron-only eigenvalue mode, **3000 particles/generation, 120 batches, 30 inactive and 90 active realizations** each. Executed `geometry.xml`, `materials.xml` and `tallies.xml` have identical byte hashes across the four seeds (settings differ by RNG).

### Why this is STILL not an operating NERVA reactor

1. **Source convergence absent:** archived source history qualification flags `FAIL_SOURCE_CLONING_SCREEN`, while spatial fission-source stationarity is false. A formally small k standard error cannot resolve the persistent genealogical/source-shape dependence.
2. **Far subcritical:** independent actual OpenMC criticality audit reports `FAR_BELOW_CRITICAL`. This cold surrogate cannot be identified as a critical operating NERVA core. Do not change control, enrichment, geometry or loading based on this screening result and then call it validation.
3. **Hot nuclear data absent:** only a single ~293.605913 K evaluated isotope temperature is in the current library; only 1/4 root-reachable material regions had an explicitly exported physical temperature. The audit separately identifies implicit defaults, unqualified thermal scattering, and undocumented phase fractions. Hot fuel/coolant states are not nuclear-data-qualified.
4. **Physical heating absent:** the actual neutron-only local `heating-local` score is reported zero for all seeds and the existing audit says `BLOCKED_NO_VERIFIED_LOCAL_HEAT_DEPOSITION`; ordinary `heating` and `kappa-fission` differ and cannot substitute as a material-disjoint neutron/photon hot-core heating source. `thermal_power_normalization_allowed = false`.
5. **No historical qualification:** the geometry/material set is a conceptual NERVA-derived surrogate and has not been matched to independently validated NASA test hardware and multiple measured reactor observables.
6. **Data release name unverified:** the actual nuclear isotope HDF5 filenames/digests were retained by the source run, but `nuclear_data_release_edition_verified_externally = false`. Their presence is not a certification of a hot NERVA material library.

## Stage 21 implementation

### `ntr/nerva/real_transport_evidence_gate.py`

Read-only audit accepts the **original GitHub artifact ZIP** or its extracted directory. It:

- Requires genuine, previously executed OpenMC `statepoint.120.h5` HDF5 binaries, explicit independent seed count, neutron-only eigenvalue mode, OpenMC v0.15.3, saved fission-source evidence, physically meaningful active/inactive batch accounting and positive finite native k estimates.
- Independently recalculates the **binary native** per-seed k values and inverse-variance aggregate, and compares them with the source run's reported observation.
- Authenticates each executed statepoint's **actual binary** SHA against the saved transport manifest; with ZIP input it can also verify the external GitHub Actions archive SHA.
- Requires `geometry.xml`, `materials.xml`, and `tallies.xml` to be byte-identical for the independently seeded model. No silently mixed geometries or tallies.
- Reconciles SHA of the transport summary in criticality, temperature, local heating, source-resolution and history audits; checks the declared nuclear-data index digest matches across observational reports.
- Separately reports physical **PASS/BLOCKED** for source stationarity, hot material neutron data, actual local energy deposition/power-normalization validity, criticality and historical benchmarks.
- Intentionally **returns a successful read-only integrity observation while physical gates are blocked**; the optional `--require-physical-qualification` returns code 3 for an unqualified core. This disallows a real archive being mistaken for a real *hot* reactor.
- NEVER runs neutron transport itself, optimizes nuclear design, manufactures cross sections or infers physical heat from arbitrary eigenflux. The output is an *evidence snapshot*, not a NERVA design recommendation.

### Reproducible use

```bash
python -m pip install numpy h5py
python -m ntr.nerva.real_transport_evidence_gate \
  /path/to/nerva-real-openmc-four-seed-core-evidence.zip \
  --expected-archive-sha256 cf05ab907bee76b198d3dc4736a84c6d560f3a8c2d753b78b5605c4826e53274 \
  --output build/real_core_evidence_qualification.json

NUKE_REAL_CORE_EVIDENCE_PATH=/path/to/nerva-real-openmc-four-seed-core-evidence.zip \
  python -m ntr.nerva.validate_real_transport_evidence_gate
```

The dedicated `nerva-real-evidence-gate.yml` CI retrieves this exact GitHub Actions archive, requires the pinned ZIP SHA, runs the independent binary checks, tests deliberately mutated evidence rejection and publishes **only a small derived JSON observation report**. No source HDF5 files are checked into NUKE Git or fabricated in tests. CI depends on the actual original artifact staying downloadable; **artifact expiration is a blocker requiring deliberate renewal**.

## Priority after Stage 21

**P0** — Preserve the expired-soon authentic HDF5 evidence and required evaluated XS manifests in a persistent immutable archive; obtain an independent source-convergence-qualified transport case at the declared real/surrogate geometry; qualify hot material-temperature/phase nuclear data; establish real disjoint neutron/photon deposited power and normalization rather than reusing the zero neutron-only local-heating score. A real reactor-core claim remains **BLOCKED** until all of these are addressed and the target configuration is independently physically benchmarked.

**Status:** Stage 21 adds read-only evidence verification and corrects traceability; it does NOT promote Stage 20 synthetic neutron/thermal solver outputs to historical NERVA physical validation. Stack parent PR #132.
