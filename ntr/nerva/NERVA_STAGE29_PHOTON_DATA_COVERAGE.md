# NUKE Stage 29 — actual evaluated photon element coverage is required before coupled transport

**Date:** 2026-10-26  
**Parent:** Stage 28 PR #142 (photon-transport mode blocks neutron-only incomplete heat)  
**Physics domain:** continuous-energy OpenMC neutron/photon data preflight; no fuel composition, reactor criticality or geometry optimization.

## Defect found and corrected

Before this change, `ntr/nerva/nuclear_data.py` considered photon-data coverage sufficient in coupled mode when *any* `type="photon"` cross-section library entry existed in `cross_sections.xml`, even if none of the other modeled material elements had photon interactions. This was not a valid prerequisite for spatial photon energy-deposition modelling.

The new conservative preflight maps the neutron isotope names already assembled for every **declared NERVA model material** to the corresponding **elemental photon data identity**. Photon HDF5 data is elemental (e.g. a hydrogen isotope `H1` requires photon data indexed as `H`), not separate isotope-specific photon files. Using all declared materials matches the pre-existing conservative neutron preflight and may include unplaced materials; it does **not** assert every declared element participates in the selected root geometry.

For `photon_transport=True`, `inspect_nuclear_data` now:

1. Requires a photon cross-section index for **every** requested element, not just one arbitrary photon library entry.
2. Requires the corresponding actual HDF5 file to exist even if the caller selected an XML-only file-existence bypass.
3. Opens the photon HDF5 with the high-level `openmc.data.IncidentPhoton.from_hdf5` parser, verifying that the embedded atomic number and element name agree with the index.
4. Requires the three fundamental photon reaction MT identifiers **502 (coherent), 504 (incoherent) and 522 (photoelectric)**, plus an available atomic-relaxation record, and refuses unreadable or mismatched binaries.
5. Refuses ambiguous multiple HDF5 records for one element, or a multi-element photon HDF5 record whose single-element identity cannot be verified unambiguously.
6. Retains an SHA-256 digest for each inspected original file; reports missing elements and rejected binaries rather than silently declaring `ready=True`.
7. Records `photon_evaluated_release_verified=False`. Actual HDF5 parsing and completeness **do not establish** certified evaluated release/source identity, hot nuclear material fidelity, source convergence or physical energy-deposition accuracy.

`ntr/nerva/check_nuclear_data.py` now prints element-wise findings, and `run_openmc_transport.py` already fails before model export/transport when `inspect_nuclear_data(...).ready=False`, so the improvement is wired directly into the existing OpenMC transport command. Neutron-only transport is not newly blocked by missing photon data.

## Tests and source

`validate_stage29_photon_data_preflight.py` contains **11 negative/positive parser regressions** (synthetic bytes or mocked high-level reader *only for algorithm testing*): one arbitrary photon entry, fake HDF5, missing binary, swapped element identity, missing reaction data, atomic relaxation absent, duplicate index, ambiguous file, neutron-only backward compatibility, logically complete mocked inventory (NOT real evaluated physics), and file-check bypass prevention.

`.github/workflows/nerva-stage29-photon-data.yml` independently executes these regressions on the repo's Python OpenMC package and runs **real** `openmc/openmc:v0.15.3` Docker image with the **actual evaluated NNDC HDF5 index and files**. It accepts an honest negative physical-coverage result (exit code 2), but does not fabricate missing source binaries or change physics. It retains the original evaluated-data report and container image digest for follow-up. `verify_stage29_photon_run.py` independently checks logical consistency of the original HDF5 report; it never calls an HDF5 mock.

Relevant primary sources:

- <https://docs.openmc.org/en/stable/usersguide/data.html#photon-cross-sections>
- <https://docs.openmc.org/en/stable/methods/energy_deposition.html>

## Physical blockers not closed by this patch

**Even a green Stage 29 CI result cannot certify a hot NERVA reactor:** original four-seed 2026-10-23 core transport was cold, neutron-only, highly subcritical, with unqualified fission-source stationarity, zero usable `heating-local`, and missing isotope HDF5 bytes from the *saved archival artifact*. To proceed with physical neutron+photon power, independently qualify appropriate hot evaluated neutron/photon data, bound scattering and actual source model; perform genuine coupled transport, validate deposited heating material partitions and source-rate normalization, and compare with a consistent NASA reference configuration.

**Status:** implementation complete; CI must be verified before marking numerical acceptance.


## Verified actual-data CI (2026-10-26)

**[Dedicated Stage 29 workflow 40312173863](https://github.com/dubakoor13/NUKE/actions/runs/40312173863) COMPLETED SUCCESS**, both jobs:

- `photon-index-numerics`: **11/11 adversarial parser tests PASSED**; previous `validate_nuclear_data` also PASSED.
- `actual-evaluated-photon-binaries`: on the real `openmc/openmc:v0.15.3` image and the actual evaluated NNDC HDF5 index, the high-level Python parser opened and verified elemental photon nuclear interaction files for **10/10 required declared-model elements: Al, B, Be, C, Cr, Fe, H, Ni, U, Zr**.
- The evaluated-data report recorded `missing_photon_elements=[]`, `photon_binary_audit_errors=[]`, `photon_transport_requested=true` and **`ready=true` for this model's current *cold data inventory preflight***.
- The separate verifier `verify_stage29_photon_run` confirmed report arithmetic and retained `photon_evaluated_release_verified=false`, `hot_NERVA_neutron_photon_heating_physically_qualified=false`. The latter flag must remain false: **no real coupled neutron/photon OpenMC simulation was executed in Stage 29**.
- The physical model remains a conceptual cold surrogate with unqualified historical configuration, temperature-dependent hot neutron scattering, source stationarity and neutron/photon deposited-power observables. **Photon elemental HDF5 file availability is now established in the official image, and does not close those gaps.**

This progress is a concrete evaluated-library prerequisite for the subsequent genuine coupled transport run, not an assertion of useful thermal energy deposition.
