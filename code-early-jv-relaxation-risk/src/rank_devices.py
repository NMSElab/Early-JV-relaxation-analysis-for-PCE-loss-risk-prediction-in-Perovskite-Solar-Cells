"""
rank_devices.py
===============

Rank new perovskite solar cells by long-term degradation risk from their
first four Session-1 J-V scans, using the frozen published model bundle.

The script never uses follow-up data: only the first four ordered Session-1
scans per device enter the features. Devices failing quality control are
reported as REVIEW and never silently scored.

Usage
-----
    python src/rank_devices.py --input my_scans.csv --output ranked.csv
    python src/rank_devices.py --input my_scans.csv --endpoint T90_500h

Required input columns (long format, one row per J-V scan)
-----------------------------------------------------------
    device_id            unique device identifier
    storage_time_h       time since first measurement of the device, hours
    session_number       measurement session index (1 = initial session)
    iteration_in_session scan index within the session
    pce_rev_pct, voc_rev_v, isc_rev_ma_cm2, ff_rev_pct,
    rs_rev_ohm_cm2, rsh_rev_ohm_cm2, hysteresis_index

    subseries_id is optional (carried through to the output if present).

Measurement protocol expectations (see the paper's Methods)
------------------------------------------------------------
    - Encapsulated p-i-n devices measured under a solar simulator.
    - At least four Session-1 J-V scans acquired at intervals of tens of
      minutes (not immediately repeated sweeps), reverse-scan descriptors.
    - Dark storage between measurement occasions (ISOS-D-aligned protocol).

Output
------
    One row per device, sorted by descending risk for the primary endpoint
    (T90 within 500 h), with for each supported endpoint:
      risk__<endpoint>        model probability (use for ranking/triage)
      tier__<endpoint>        Low / Intermediate / High (training tertiles)
    plus quality-control columns:
      qc_status               OK | OK_OOD_REVIEW | REVIEW_NOT_SCORED
      qc_detail               reason(s) for review
      n_s1_scans              number of Session-1 scans found

Interpretation caveats (from the model card)
--------------------------------------------
    - Scores are risk rankings for the supported endpoints under the
      training measurement protocol; they are NOT universal lifetime
      forecasts and not validated acceptance criteria.
    - Out-of-distribution inputs (features outside the training range)
      are flagged and must be reviewed by the user.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import (DEVICE, SUBSERIES, FEATURES, MIN_SCANS,
                            build_sequence_features)

DEFAULT_BUNDLE = Path(__file__).resolve().parent.parent / "model" / "early_JV_risk_grid_bundle.joblib"
PRIMARY_ENDPOINT = "T90_500h"


def qc_device(row, feature_ranges):
    """Return (status, detail) for one device's feature vector."""
    missing = [f for f in FEATURES if not np.isfinite(row.get(f, np.nan))]
    if missing:
        return "REVIEW_NOT_SCORED", f"missing/non-finite features: {len(missing)} (e.g. {missing[:3]})"
    out_of_range = []
    for f in FEATURES:
        r = feature_ranges.get(f)
        if r is None:
            continue
        if row[f] < r["min"] or row[f] > r["max"]:
            out_of_range.append(f)
    if out_of_range:
        return "OK_OOD_REVIEW", f"outside training range: {len(out_of_range)} feature(s) (e.g. {out_of_range[:3]})"
    return "OK", ""


def risk_tier(probability, cutpoints):
    if probability < cutpoints["low_to_intermediate"]:
        return "Low"
    if probability < cutpoints["intermediate_to_high"]:
        return "Intermediate"
    return "High"


def main():
    parser = argparse.ArgumentParser(description="Rank devices by early J-V long-term degradation risk.")
    parser.add_argument("--input", required=True, help="CSV with Session-1 J-V scans (long format).")
    parser.add_argument("--output", default="device_risk_ranking.csv", help="Output CSV path.")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE), help="Path to the frozen model bundle (.joblib).")
    parser.add_argument("--endpoint", default="all",
                        help="Endpoint to score: all (default) or one of the supported endpoint names.")
    args = parser.parse_args()

    bundle = joblib.load(args.bundle)
    models = bundle["models"]
    endpoints = list(models) if args.endpoint == "all" else [args.endpoint]
    for endpoint in endpoints:
        if endpoint not in models:
            sys.exit(f"Unsupported endpoint '{endpoint}'. Supported: {list(models)}")

    scans = pd.read_csv(args.input)
    scans.columns = scans.columns.str.strip()
    sequence = build_sequence_features(scans)
    if sequence.empty:
        sys.exit(f"No device has >= {MIN_SCANS} Session-1 scans; nothing to rank.")

    all_devices = scans[DEVICE].astype(str).str.strip().dropna().unique()
    skipped = sorted(set(all_devices) - set(sequence[DEVICE].astype(str)))

    # Quality control uses the primary endpoint's training feature ranges
    # (ranges are near-identical across endpoint models).
    feature_ranges = models[PRIMARY_ENDPOINT if PRIMARY_ENDPOINT in models else endpoints[0]]["feature_ranges"]

    records = []
    for _, row in sequence.iterrows():
        status, detail = qc_device(row, feature_ranges)
        record = {
            DEVICE: row[DEVICE],
            "n_s1_scans": row["n_s1_scans"],
            "qc_status": status,
            "qc_detail": detail,
        }
        if SUBSERIES in sequence.columns and pd.notna(row.get(SUBSERIES)):
            record[SUBSERIES] = row[SUBSERIES]
        if status != "REVIEW_NOT_SCORED":
            x = row[FEATURES].to_frame().T.astype(float)
            for endpoint in endpoints:
                model = models[endpoint]
                probability = float(model["model"].predict_proba(x[FEATURES])[:, 1][0])
                record[f"risk__{endpoint}"] = probability
                record[f"tier__{endpoint}"] = risk_tier(probability, model["risk_cutpoints"])
        else:
            for endpoint in endpoints:
                record[f"risk__{endpoint}"] = np.nan
                record[f"tier__{endpoint}"] = "REVIEW"
        records.append(record)

    result = pd.DataFrame(records)
    sort_key = f"risk__{PRIMARY_ENDPOINT}" if f"risk__{PRIMARY_ENDPOINT}" in result.columns else f"risk__{endpoints[0]}"
    result = result.sort_values(sort_key, ascending=False, na_position="last").reset_index(drop=True)
    result.insert(0, "risk_rank", np.where(result[sort_key].notna(), np.arange(1, len(result) + 1), np.nan))
    result.to_csv(args.output, index=False)

    scored = int(result[sort_key].notna().sum())
    print(f"Ranked {scored}/{len(result)} devices "
          f"({len(skipped)} device(s) skipped: fewer than {MIN_SCANS} Session-1 scans).")
    if skipped:
        print(f"  Skipped: {skipped[:10]}{' ...' if len(skipped) > 10 else ''}")
    review = result.loc[result["qc_status"] != "OK", [DEVICE, "qc_status", "qc_detail"]]
    if not review.empty:
        print(f"QC review needed for {len(review)} device(s):")
        print(review.to_string(index=False, max_rows=20))
    print(f"Output written to {args.output} (sorted by {sort_key}, highest risk first).")


if __name__ == "__main__":
    main()
