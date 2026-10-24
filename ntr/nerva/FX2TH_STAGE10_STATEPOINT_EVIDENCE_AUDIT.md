# FX2-TH / NUKE Stage 10 — OpenMC StatePoint heating evidence and independent-seed numerics
Date: 2026-10-24. Parent: Stage 9 PR #121. This branch is a numerical QA improvement, not proof that a NERVA OpenMC calculation has been run.

## Scope
Stage 9 accepted externally extracted mesh heating arrays. Stage 10 implements an opt-in reader using the **official openmc.StatePoint Python API**, plus tests of source shape, mesh filters, tally score, particle-source normalization, binwise uncertainty and reproducibility across independent random seeds.

The StatePoint reading entry point is `read_openmc_heating_statepoint(statepoint_path, nuclear_data_manifest_path, nuclear_material, *, tally_ids, source_id, ...)`. It **requires OpenMC's Python API installed separately** and does not run OpenMC. NUKE main and historical NERVA reactor inputs are unchanged.

## Strict tally scope and units
- Require 3 distinct named-by-ID tally objects: total integrated heating, solid material heating, and hydrogen material heating.
- All must contain ONLY the OpenMC `heating` score, `total` nuclide bin, same 3D Cartesian RegularMesh and same tally realization count. MeshFilter must be the FIRST filter. Any extra filter must be a SINGLE material/cell bin, never implicitly summed. Reject translated/rotated filters.
- Use `tally.get_reshaped_data('mean', expand_dims=True)` and the corresponding `std_dev`, rejecting mismatched xyz shapes rather than reinterpreting the axes.
- Tally energy is eV/source history/mesh-bin, NOT W/cm3. Use Stage 9 external fixed-source histories/s or external W normalization. An **eigenvalue** statepoint cannot use the fixed-source histories/s mode in this implementation.
- No field can contain negative or nonfinite means/standard errors; no local solid+hydrogen heat sum can exceed the total mean without throwing an exception. This strict fail-closed requirement may reject statistically noisy, non-converged tallies; the remedy is better input evidence, NOT clipping.
- An explicitly configured minimum number of tally realizations and maximum relative standard error apply to significant heating bins (cutoff is relative to each tally's largest mean bin). Zero mean with nonzero standard error is rejected. Suppressed low-mean bins are counted and exposed, not silently qualified.

## Data provenance
- The reader SHA-256 hashes the ACTUAL statepoint file and an externally supplied nuclear-data *manifest* file.
- It also stores a deterministic fingerprint of the externally supplied Stage 7 multigroup material/operator inputs. **This does not establish that OpenMC used those same materials**; the user must verify actual core geometry/physics inputs, evaluated cross-section HDF5 files, temperature interpolation, and the neutron/photon heating score definition.
- Reported source category `extracted-openmc-arrays` indicates actual OpenMC Python API file import, not certified physics. Test-double evidence remains `synthetic`.

## Independent seeds and uncertainty
- `compare_independent_statepoints` accepts at least four **distinct seed IDs and distinct statepoint SHA hashes** with identical tally IDs, mesh, source and material fingerprint.
- Each run is separately mapped through the same Stage 9 conservative thermal projection, and the **sample standard error of the full thermal-cell power across runs** is reported. This preserves per-run spatial-bin correlations in the *observed across-run* variation.
- Tally `std_dev` is the standard error of a single spatial bin. This module does **not** assume statistical independence between spatial bins and does NOT claim a total power SEM by summing individual-bin variances. Between-run estimates from four seeds remain imprecise and require convergence and broader confidence analysis before physical prediction.

## Tests
- `validate_fx2th_openmc_statepoint_heating.py` uses explicitly labeled OpenMC-compatible **test doubles** (not an OpenMC runtime) to test array expansion, 3D mesh and nuclide/score filter guards, run-mode normalization, energy conservation, high variance rejection, strict local heat closure, and independent-seed SEM.
- CI also reruns Stage 9 energy projection tests.
- No real StatePoint file is present in the repo for physical verification in this PR. No neutron/photon tally from evaluated cross-section data has yet been executed.

## Next physical tasks
1. Acquire reproducible real OpenMC statepoints from NUKE's existing hot/cold nuclear models and independently confirm mesh bin order and tally units, first through the official API.
2. Verify heating score composition (including photons and delayed contributions) and that material partitions are disjoint for the selected tally estimator.
3. Audit source-rate/power normalization, independent seed convergence, active-batch/source convergence and spatial distributions; then compare against transport benchmarks with nuclear-data hashes.
4. Connect validated deposition to resolved fuel conduction and hydrogen enthalpy-based channel flow rather than assuming Stage 8 constant-cp two-node surrogate.
5. Do not infer NERVA reactor power or physical stability solely from eigenvalue tally normalization or tests based on test doubles.

Reference APIs: OpenMC documentation `openmc.StatePoint`, `openmc.Tally.get_reshaped_data` and official statepoint format; see docs.openmc.org.
