# FX2-TH / NUKE Stage 9 — transport-heating energy normalization and conservative transfer
Date: 2026-10-24
Parent: PR #120, Stage 8 thermal feedback. No NUKE main source changed.

## Physics gap being closed
The NUKE OpenMC transient pipeline currently distributes per-voxel heating across components while prescribing reactor power from outside. The Stage 8 neutron/thermal callback requires power in W on the exact thermal cells, but Stage 7 flux has arbitrary normalization. This module provides a conservative, independently testable transfer of externally extracted tally energies into Stage 8 thermal deposition without asserting an unearned neutron-to-watt normalization.

## Score interpretation: tally energy is NOT a power density
Input total, solid and hydrogen scores are dimensioned **eV per source history per [x,y,z] mesh bin**. They must be disjoint partitions from the same tally normalization, energy-deposition score (heating, including the selected neutron/photon treatment), estimator and mesh. They are not W/cm3 and must not be divided by voxel volume before being transferred. All bins must have nonnegative finite values, and solid+hydrogen must not exceed total in ANY bin.

Unassigned heat is the remainder in each bin, not a silently discarded/renormalized channel. It may include reflector, vessel, tie, coating or other regions not represented by the simple Stage 8 fuel/hydrogen thermal cells. Real-source partitions require geometric/material verification upstream: the module cannot discover region identities from a generic mesh tally.

## Power normalization
- Fixed-source mode: Q_bin [W] = heating_bin [eV per source] * source histories per second * 1.602176634e-19 J/eV.
- Externally prescribed power: Q_bin [W] = P_external [W] * heating_bin / SUM(all heating_bin). P_external must come from an independent, traceable normalization.
- No k_eff, arbitrary-normalized eigenvector, fission matrix result or assumed joules/fission is used to derive P. The user has to supply the normalization.

## Exact axial remapping, x/y integration
For a transport-bin axial interval [a,b] and a thermal-bin interval [c,d], weight equals max(0, min(b,d)-max(a,c))/(b-a). Weights sum to one per transport z bin. Heating in each transport bin is integrated over its full x/y mesh bins and distributed to thermal bins with these exact overlap fractions. This is conservative for arbitrary nonmatching axial cell boundaries. Thermal edges must span the full transport z extent with no gaps. The solver does not reconstruct subvoxel power shapes or model mixed material overlap without component-specific source tallies.

Projected solids, hydrogen and unmodeled compartments are provided as a Stage 8 ThermalDeposition, with an explicit synthetic flag, source identifier and global energy residual.

## Independent matching of transport snapshot to a nuclear material state
The callback used by Stage 8 requires an external HeatingMeshScore on each Picard trial. A deterministic SHA-256 fingerprint includes the Stage 7 integrated loss matrix, nuSigmaF, beta-family data, prompt/delayed spectra, volumes, group speeds and decay constants. A supplied transport snapshot must claim this same fingerprint or is rejected as stale. The fingerprint verifies array-state matching *only*: a matching string is NOT proof that someone ran OpenMC with those materials, geometry or cross-section libraries. The declared statepoint/data SHA-256 fields are traceability tokens, not independently verified files.

Files:
- ntr/nerva/fx2th_transport_heat_projection.py — unit normalization, exact axial overlap and checked material/provenance callback.
- ntr/nerva/validate_fx2th_transport_heat_projection.py — synthetic analytical, unit and regression checks; includes one Stage 8 coupled thermal step.
- .github/workflows/fx2th-transport-heat-projection.yml — independent GitHub test execution.

## Strict missing physical qualifications
1. No actual OpenMC StatePoint reader is implemented here. Tally extraction, independent SHA checks, mesh filter axis order, score history normalization, stochastic uncertainty and neutron/photon heating options must be verified upstream.
2. No physical reactor power from criticality or transient amplitude; a fixed-source primary rate or independently prescribed power is required, with correct source weighting.
3. No resolved fuel/tie-element geometrical overlap, variable material heat capacity, cryogenic-to-hot hydrogen EOS, conduction PDE, pressure-density feedback or qualified temperature-dependent multigroup nuclear data.
4. No independently reproduced NASA NERVA hot-core benchmark or calibrated NERVA geometry is claimed. Synthetic unit tests establish only the transfer algorithm and energy conservation.

Next: authenticated direct StatePoint/mesh/heating score import with source normalization, statistical uncertainties and provenance; then independent OpenMC benchmark and coupling to resolved NUKE fuel/hydrogen physics. Do not promote synthetic tests to historical NERVA validation gates.
