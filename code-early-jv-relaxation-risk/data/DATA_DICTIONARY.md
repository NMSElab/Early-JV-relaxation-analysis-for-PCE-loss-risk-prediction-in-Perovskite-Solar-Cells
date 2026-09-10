# Data dictionary

## `jv_scans.csv` — one row per J–V scan (8,650 rows, 1,451 devices)

| Column | Unit | Description |
|---|---|---|
| `device_id` | — | Coded device identifier (`PSC-0001` …). |
| `subseries_id` | — | Coded fabrication-subseries identifier (`SS-001` …). Devices within a subseries were fabricated together and are statistically correlated — use it as the grouping unit for cross-validation. |
| `storage_time_h` | h | Time since the device's first measurement. |
| `session_number` | — | Derived measurement-session index; `1` is the initial session used for early features. |
| `iteration_in_session` | — | Scan index within the session (1, 2, 3, …). |
| `pce_fwd_pct`, `pce_rev_pct`, `pce_qss_pct` | % | Power conversion efficiency: forward scan, reverse scan, quasi-steady-state. |
| `isc_fwd_ma_cm2`, `isc_rev_ma_cm2`, `isc_qss_ma_cm2` | mA/cm² | Short-circuit current density. |
| `voc_fwd_v`, `voc_rev_v`, `voc_qss_v` | V | Open-circuit voltage. |
| `ff_fwd_pct`, `ff_rev_pct`, `ff_qss_pct` | % | Fill factor. |
| `rs_fwd_ohm_cm2`, `rs_rev_ohm_cm2` | Ω·cm² | Series resistance. |
| `rsh_fwd_ohm_cm2`, `rsh_rev_ohm_cm2` | Ω·cm² | Shunt resistance. |
| `hysteresis_index` | — | Scan hysteresis index. |

Missing values indicate that the quantity could not be extracted for that scan.

## `device_metadata.csv` — one row per device (1,451 rows)

| Column | Description |
|---|---|
| `device_id`, `subseries_id` | As above. |
| `passivator_condition` | Exact treatment label (e.g. `control`, `PDAI2`, `PbCl2, MAI`). Chemically distinct PDAI2-family labels (`PDAI2`, `PDAI2_n`, `PDAI2new`, `PDAI2(HK)`) are never merged. |
| `is_control` | True if untreated control. |
| `n_components` | Number of treatment components. |
| `mixture_type` | `control` / `monotherapy` / `combination`. |
| `passivator_concentration_mg_ml` | Subseries-level treatment concentration, mg/mL (as recorded; may be missing). |
| `annealing_temp_c_perovskite` | Perovskite annealing temperature(s), °C (as recorded). |
| `solvent_type` | Treatment solvent (as recorded; may be missing). |
| `pce0` | Pristine (initial) reverse-scan PCE, %. |
| `pce0_physical` | False for artifact-range initial values; such devices are excluded from model training. |
| `pce_tier` | Initial-performance tier from 1-D K-means (k=4) on `pce0`: `G1_nonfunctional` / `G2_low` / `G3_moderate` / `G4_high` / `ARTIFACT`. |
| `timing_sensitive` | True if Session-1 timing flags made the device's early features unreliable; excluded from training. |
| `partition` | Frozen paper split: `development` / `locked_test` (assigned by subseries); empty if the device is not in the modelling population. |

## `device_features_endpoints.csv` — ML-ready table (423 devices with ≥4 Session-1 scans)

| Column | Description |
|---|---|
| `device_id`, `subseries_id` | As above. |
| `n_s1_scans` | Number of Session-1 scans (≥4 by construction). |
| `s1_end_h` | End of Session 1 (landmark for all endpoints), h. |
| `seq__<D>__iter<1..4>` | The 28 frozen features. For descriptors `PCE_rev`, `Voc_rev`, `Isc_rev`, `FF_rev`, `HI`: relative change from the first Session-1 scan, `(x_i − x_1)/|x_1|`. For `Rs_rev`, `Rsh_rev`: log-ratio `ln(x_i/x_1)`. `iter1` is 0 by construction where valid. |
| `<ENDPOINT>__known` | True if the endpoint label is determinable for this device (confirmed event, or follow-up covering the full horizon). |
| `<ENDPOINT>__label` | 1 = confirmed persistent loss within the horizon after the Session-1 landmark; 0 = confirmed survival past the horizon; empty = unlabeled (conservative censoring). |
| `pce0`, `pce_tier`, `pce0_physical`, `timing_sensitive`, `partition` | Copied from device metadata for convenience. |

Endpoints: `T90_500h`, `T80_500h`, `T80_1000h`, `T70_1000h`, `T70_3000h`,
`T50_3000h` — confirmed crossing of 90/80/70/50% of the final valid Session-1
PCE within 500/1,000/3,000 h after the Session-1 landmark, requiring two
consecutive below-threshold observations.

## Endpoint and eligibility rules used in the paper

- Training eligibility per endpoint: label known, complete 28 features,
  `timing_sensitive == False`, `pce0_physical == True`.
- Development metrics: grouped cross-validation by `subseries_id`.
- Locked metrics: single frozen evaluation on `partition == locked_test`.
