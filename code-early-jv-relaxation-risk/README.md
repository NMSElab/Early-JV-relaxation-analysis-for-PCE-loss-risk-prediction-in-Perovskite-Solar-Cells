# Early J–V Relaxation Predicts Persistent PCE Loss in Hybrid Perovskite Solar Cells

Data and code release for the paper:

> **Early J–V Relaxation Predicts Persistent PCE Loss in Hybrid Perovskite Solar Cells.**
> E.I. Marchenko, N.A. Belich, N.N. Udalova, A. Khakimullin, C. Wang, Y. Li, P.A. Ivlev, S. Tyutina, A.A. Ordinartsev, M.A. Schwartzberg, E.A. Goodilin, A.B. Tarasov.
> Lomonosov Moscow State University. *(journal reference to be added upon publication)*

This repository contains (i) a longitudinal J–V archive, with coded device identifiers, of encapsulated
p–i–n perovskite solar cells under an ISOS-D-aligned dark-storage protocol,
(ii) the frozen machine-learning models that rank long-term degradation risk from
the first four Session-1 J–V scans, and (iii) code to score new devices and to
reproduce model training from the released data.

## What the model does

From **four early J–V scans** of a fresh device (acquired at intervals of tens of
minutes during the first measurement session), the model computes 28 trajectory
features (relative scan-to-scan changes of reverse-scan PCE, V<sub>OC</sub>,
J<sub>SC</sub>, FF, hysteresis index, and log-ratio changes of R<sub>s</sub>,
R<sub>sh</sub>) and outputs a **risk score for confirmed persistent PCE loss**
at six pre-specified severity/horizon endpoints. Scores are intended for
**ranking and triage** of devices within a cohort measured under a comparable
protocol — not as calibrated lifetime forecasts.

Locked (unseen fabrication subseries) test performance of the frozen models:

| Endpoint | Locked ROC-AUC [95% CI] | Locked PR-AUC | Event prevalence |
|---|---|---|---|
| T90 within 500 h (primary) | 0.80 [0.63–0.93] | 0.69 | 0.36 |
| T80 within 500 h | 0.88 [0.70–1.00] | 0.83 | 0.24 |
| T80 within 1,000 h | 0.77 [0.59–0.92] | 0.64 | 0.36 |
| T70 within 1,000 h | 0.74 [0.54–0.91] | 0.59 | 0.33 |
| T70 within 3,000 h | 0.71 [0.52–0.88] | 0.76 | 0.52 |
| T50 within 3,000 h | 0.79 [0.62–0.94] | 0.80 | 0.48 |

CIs: 5,000-resample stratified bootstrap on the locked set (33/31 devices).
One-sided label-permutation p = 0.002 (100,000 permutations) for the primary endpoint.

## Repository contents

```
data/
  jv_scans.csv        8,650 J-V scans, 1,451 devices, 124 subseries,
                                 follow-up up to ~19,000 h (long format)
  device_metadata.csv device-level metadata: passivation condition,
                                 initial PCE, tiers, QC flags, dev/locked split
  device_features_endpoints.csv  ML-ready table: 28 frozen features + labels for
                                 all six endpoints (423 devices with >=4 scans)
  DATA_DICTIONARY.md             column-by-column description
model/
  early_JV_risk_grid_bundle.joblib   frozen models (one per endpoint) + training
                                     tertiles + feature ranges for QC
  early_JV_risk_grid_model_card.json model card
  early_JV_risk_grid_training_manifest.csv
  early_JV_risk_grid_supported_endpoints.csv
src/
  build_features.py    frozen feature engineering + endpoint definitions
  rank_devices.py      score & rank new devices from >=4 Session-1 scans
  train_risk_models.py retrain the bundle from the released data
examples/
  example_new_cohort.csv       12 devices x 4 Session-1 scans (input format)
  example_ranked_output.csv    corresponding ranked output
```

## Quickstart: rank your own devices

```bash
pip install -r requirements.txt
python src/rank_devices.py --input examples/example_new_cohort.csv --output ranked.csv
```

Your input CSV must contain one row per J–V scan with columns
`device_id, storage_time_h, session_number, iteration_in_session,
pce_rev_pct, voc_rev_v, isc_rev_ma_cm2, ff_rev_pct, rs_rev_ohm_cm2,
rsh_rev_ohm_cm2, hysteresis_index` (see `examples/example_new_cohort.csv`).
Each device needs **at least four Session-1 scans**; only the first four
(ordered by time) are used, and no later information enters the score.

The output is one row per device, sorted by descending risk for the primary
endpoint (T90 within 500 h), with a probability and a Low/Intermediate/High
tier (frozen training tertiles) for each supported endpoint. Devices with
missing or non-finite features are marked `REVIEW_NOT_SCORED`; devices with
features outside the training range are scored but flagged `OK_OOD_REVIEW`.

**Measurement protocol matters.** The model was trained on encapsulated p–i–n
devices measured under a solar simulator, with Session-1 scans acquired at
intervals of tens of minutes (not immediately repeated sweeps) and dark storage
between measurement occasions. Scores for devices measured under a very
different protocol should be treated as exploratory.

## Reproduce training

```bash
python src/train_risk_models.py            # writes model_retrained/
```

With the pinned scikit-learn version this reproduces the released frozen
bundle **bit-for-bit** (we verified identical predictions for all 363
fully-featured devices across all six endpoints). Endpoint labels use
persistent-crossing events (two consecutive below-threshold observations of
the fraction × final valid Session-1 PCE), a landmark at the end of Session 1,
and conservative censoring: devices without a confirmed event and without
sufficient follow-up are unlabeled, never treated as non-events.

## Trying your own models

`data/device_features_endpoints.csv` is a ready-to-use supervised-learning
table. To avoid information leakage, respect two rules from the paper:

1. **Group-aware validation** — always split by `subseries_id` (fabrication
   subseries), never by device; devices within a subseries are correlated.
2. **Honest partitioning** — the `partition` column preserves the paper's
   frozen split (`development` / `locked_test`). If you compare against our
   reported numbers, evaluate on `locked_test` only, without refitting on it.

The raw scan table `jv_scans.csv` additionally supports custom
feature engineering, alternative endpoint definitions, and survival analysis
(full follow-up trajectories are included).

## Intended use and limitations

- Research use only. The score ranks relative risk within a cohort; it is not
  a certified acceptance criterion and not a universal lifetime model.
- Risk scores must not be interpreted as causal effects of passivation or
  composition; treatment comparisons in the archive are observational.
- The archive covers one laboratory's device platform and storage protocol;
  transport to other platforms requires prospective validation (a template
  protocol is given in the paper's Supporting Information).

## Identifier coding note

Device, subseries, and file identifiers were replaced by neutral codes
(`PSC-xxxx`, `SS-xxx`); absolute timestamps and source file names were removed.
All measurement values, relative timing, chemistry condition labels, and the
development/locked partition are unchanged, and the frozen models reproduce
exactly on the released data.

## License

- Code: MIT License (see `LICENSE`).
- Data (`data/`): Creative Commons Attribution 4.0 International (CC BY 4.0).

## Citation

If you use the data or models, please cite the paper (see `CITATION.cff`).

## Funding

This work was supported by the Russian Science Foundation, project No. 25-73-20095.

## Contact

Corresponding author: Prof. Alexey B. Tarasov (alexey.bor.tarasov@yandex.ru),
Laboratory of New Materials for Solar Energetics, Lomonosov Moscow State University.
