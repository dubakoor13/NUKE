# FX2-TH / NUKE Stage 8: conservative thermal boundary

Date: 2026-10-24. Parent: PR #119 (spatial delayed-family neutronics). Status: synthetic numerical checks only; no NERVA historical or transport validation.

## Equations and source contracts

Two-node-per-cell model: one solid fuel node (Ts), one well-mixed hydrogen node (Th). All thermal parameters are required SI inputs. Solid/hydrogen heat capacities Cs/Ch in J/K; H in W/K; m_dot in kg/s; cp in J/(kg K); inlet temperature in K; externally normalized deposition Qs, Qh and Qother in W.

Backward Euler per cell:
  Cs*(Ts1-Ts0)/dt = Qs - H*(Ts1-Th1)
  Ch*(Th1-Th0)/dt = Qh + H*(Ts1-Th1) - m_dot*cp*(Th1-Tin)

Global energy closure:
  deltaU_s + deltaU_h + net_hydrogen_export + Qother*dt = declared_power*dt.

Deposition outside the two modeled nodes is explicitly tracked; no silent repartitioning. All three per-cell deposition channels must match a declared total. Stage 7 flux is arbitrary-normalized and **never converted into W** by this module.

## Stage 7 fixed-point connection

Two external callback providers are mandatory. The first maps a trial temperature state to a fully specified Stage 7 multigroup material state; the second maps solved neutron state and material state to explicitly normalized W deposition with provenance ID and a synthetic flag. The provider supplies ALL temperature sensitivity; this code invents no reactivity/Doppler coefficients or cross sections.

Every fixed-point trial advances the *same old neutron and thermal states* for one actual dt. Iterative trials only change the guessed temperature; there is no accidental repeated time advancement. Nonconvergence raises, and changes in cell volumes, group speeds, group/family dimensions or decay constants across trial states are refused.

Files:
- ntr/nerva/fx2th_thermal_energy.py
- ntr/nerva/fx2th_thermal_feedback.py
- ntr/nerva/validate_fx2th_thermal_feedback.py
- .github/workflows/fx2th-thermal-boundary.yml

Validation: analytical two-node heat exchange and enthalpy export, full energy inventory audit, changing synthetic feedback callbacks, time-step consistency, independent Stage 7 neutron conservation, and fail-closed mismatch/invalid-input tests.

## Model limitations

This is a mathematical coupling surrogate, NOT actual NERVA heat-transfer or cold-to-hot hydrogen core thermodynamics. Hydrogen heat capacity cp is constant and cells are well mixed; no resolved axial flow advection, fluid compressibility, pressure work, radiation, variable material heat capacity, multi-dimensional fuel conduction, nuclear decay heat, coating deposition, or metal reflector thermal response is modeled.

No genuine neutron or photon heating tally, OpenMC transport run, measured reactor power, fuel enrichment, physical benchmark comparison, nuclear-data covariance or temperature-dependent evaluated cross sections has been supplied or verified. Replacing external providers with actual physics requires independently benchmarked energy-normalization and state-consistent nuclear/thermophysical data. Conservation alone does not establish reactor prediction accuracy. Existing NUKE resolved thermal and OpenMC models are unchanged.

Do not merge this stacked PR without parent PR #119 and its dependencies.
