# Four Alternative-Propellant NTP Cases

Source basis: *Methane-based Nuclear Thermal Propulsion Engine Considerations*
(Daria Nikitaeva, AMA; NASA STMD/SNP-supported presentation).

The source supports four useful reconstruction cases in this repository:

1. **H-NTP baseline (LH2)** — reference architecture/performance data.
2. **A-NTP (LNH3)** — reference architecture/performance data.
3. **M-NTP Case 1** — complete frozen-chemistry PBM sizing case.
4. **M-NTP Case 2** — complete frozen-chemistry PBM sizing case.

The presentation does **not** provide full numbered state tables for H-NTP or
A-NTP, so this implementation intentionally does not invent them.

## H-NTP source-backed reference

- propellant: LH2
- Isp at 2700 K: approximately 900 s
- storage range: 14-33 K
- density: 71 kg/m^3
- baseline expander cycle
- boost pump required
- hydrogen exits the pump train already supercritical
- risk focus: hot-hydrogen attack / embrittlement
- stated best-fit mission: high-delta-v crewed Mars transfer

## A-NTP source-backed reference

- propellant: LNH3
- Isp at 2700 K: approximately 370 s
- storage range: 195-405 K
- density: 730 kg/m^3
- no boost pump
- serial preheat loop / extra pumping
- reaches supercritical conditions only near state 12
- risk focus: pressure drop and preheater stress
- stated best-fit mission: long-dwell depots / low-Isp cargo

## M-NTP Case 1

Frozen-chemistry sizing baseline.

- chamber-temperature constraint: approximately 2100 K
- reactor power: 177.65 MW
- Isp: 353.71 s
- total mass flow: 19.55 kg/s
- derived thrust from source mdot and Isp: approximately 67.81 kN
- maximum fuel temperature: 2270.30 K
- chamber temperature: 2100.79 K
- chamber pressure at state 21: 4.60 MPa
- maximum system pressure: 11.03 MPa
- pump power: 482.47 kW
- turbine power: 580.61 kW
- turbopump speed: 36,602.72 rpm

## M-NTP Case 2

Frozen-chemistry sizing baseline.

- fuel-temperature constraint: approximately 2850 K
- reactor power: 214.66 MW
- Isp: 400.16 s
- total mass flow: 17.43 kg/s
- derived thrust from source mdot and Isp: approximately 68.40 kN
- maximum fuel temperature: 2850.83 K
- chamber temperature: 2630.28 K
- chamber pressure at state 21: 4.60 MPa
- maximum system pressure: 11.28 MPa
- pump power: 440.09 kW
- turbine power: 529.72 kW
- turbopump speed: 39,451.63 rpm

## Important source limitation

The methane table is explicitly identified as a frozen-chemistry baseline.
The presentation states that methane cracking can increase Isp and reduce mass
flow, while high-pressure short-residence-time kinetics remain uncertain.
Therefore these two cases are sizing anchors, not final reacting-flow
predictions.

## Run

From the repository root:

```bash
python -m ntr.nerva.validate_alternative_propellant_cases
python -m ntr.nerva.run_alternative_propellant_cases
```

Generated outputs:

```text
build/alternative_ntp_cases/
├── alternative_ntp_cases.json
├── m_ntp_case_1_states.csv
└── m_ntp_case_2_states.csv
```
