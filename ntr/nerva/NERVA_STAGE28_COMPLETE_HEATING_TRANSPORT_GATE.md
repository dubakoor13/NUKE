# NUKE Stage 28 — reject neutron-only partial `heating` in complete core thermal coupling

**Date:** 2026-10-26. **Parent:** Stage 27 PR #140. **Target:** real reactor-core physics coupling correctness, NOT another independent synthetic reactor solver.

## Concrete physics problem corrected

The existing Stage 10 `fx2th_openmc_statepoint_heating.py` extracts the `heating` score for three disjoint thermal partitions and can normalize this score to externally prescribed reactor watts in Stage 9. Previously it did **not** require `StatePoint.photon_transport=True`, even when the data was tagged as genuinely extracted OpenMC arrays.

According to the [official OpenMC energy-deposition method](https://docs.openmc.org/en/stable/methods/energy_deposition.html) and [tally documentation](https://docs.openmc.org/en/stable/usersguide/tallies.html):

- **Coupled neutron+photon transport:** `heating` (neutron MT=301 component plus transported photon energy depositions) is the appropriate score to represent spatial energy deposition. Its evaluated nuclear/photon libraries and material state still need independent physical qualification.
- **Neutron-only transport:** `heating-local` (MT=901 when available) approximates secondary photons as locally deposited, and **must independently pass evaluated KERMA coverage**. The Stage 10 mesh reader does NOT implement this alternate score/mode.
- Neutron-only ordinary `heating` can be **nonzero** while missing spatial photon deposition and hence is not a validated complete fuel/H₂/reflector thermal power field.

This is not a theoretical edge case. In **actual original archived OpenMC 0.15.3 neutron-only NUKE HDF5** from [run 39933618638](https://github.com/dubakoor13/NUKE/actions/runs/39933618638), Stage 24 found ordinary `heating` nonzero but `heating-local` first and second moments identically zero. A generic score-presence check risks promoting partial fission heat into complete thermal wattage.

## Implementation

The Stage 10 real-data coupling code is hardened by:

1. `require_total_heating_transport_mode(photon_transport, score)`: accept complete `heating` tally transport mode **only when coupled neutron-photon mode is recorded**. Fail on neutron-only positive ordinary `heating`; this guard does not synthesize missing MT=901, photon spectra, or fission power.
2. Call it during `extract_statepoint_objects` and again in `StatepointHeatingEvidence.project()`, preventing bypass via a modified dataclass.
3. Require `actual_statepoint_reader` to match `source_kind='extracted-openmc-arrays'` rather than letting a synthetic reader label mock arrays as non-synthetic real evidence.
4. For independent seeded ensemble comparisons, require matching photon-transport mode and actual/synthetic reader provenance. Do not calculate seed uncertainty across incompatible physical estimators.

The Stage 9 generic matrix projection remains a numerical tool and does not become an independently qualified OpenMC source reader. The 4 original cold source cases remain intentionally BLOCKED for thermal power.

## Testing

Existing Stage 10 unit/regression tests are retained and five adversarial cases added: neutron-only score rejection, mutated dataclass projection rejection, false extracted-data source-kind rejection, mixed-reader ensemble rejection, and mixed-photon-mode ensemble rejection.

A dedicated genuine-HDF5 audit `audit_original_neutron_only_thermal_gate.py` reopens exact original four statepoints through the SHA-checked Stage 21/24 source evidence API and verifies each original neutron-only mode is rejected despite positive ordinary heating. `validate_stage28_original_neutron_only_heating.py` requires the archive and has **no mock fallback**. Workflow `nerva-stage28-real-heating-gate.yml` tests both mathematical adapter and genuine original archive.

**Before CI execution:** staged source code is not a passed test. Record job counts and original archive outcome only after successful workflow execution.

## What remains physically missing after successful gate

This change prevents a **false heating promotion**; it cannot create correct hot reactor heating.

- Evaluated neutron/photon data for the as-modeled hot fuel, hydrogen, coating, reflector and other physically present materials. Original transport had only ~293.6 K evaluated neutron temperature and no archived isotope binary data for independent requalification.
- Genuine coupled neutron+photon core transport with source/stationarity checks, nonzero and materially separated deposited heat, leakage accounting and uncertainty for one explicitly documented core.
- Consistent transport geometry, material and isotope identity to thermal fuel/channel/tie geometry. Present spatial thermal feedback also needs density, pressure, full hydrogen h(P,T) and appropriate nuclear feedback.
- Physical normalizing power/source rate supported by independent reactor evidence; never convert k-effective, arbitrary eigenflux, `kappa-fission`, or neutron-only partial ordinary heating to thermal watts without distinct physical assumptions and evidence.
- Independent NASA historical measurement comparisons; numerical pass alone cannot justify 100% physics completion.

**No core geometry, uranium loading, control drums, criticality targeting or nuclear reaction coefficients were modified.**
