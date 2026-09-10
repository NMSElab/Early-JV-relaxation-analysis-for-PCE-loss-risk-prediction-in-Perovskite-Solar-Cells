"""
train_risk_models.py
====================

Retrain the early J-V risk models from the released dataset and
export a model bundle identical in structure to the published frozen bundle.

This reproduces the training half of the published workflow:
  1. Build the four-scan Session-1 feature table (28 features).
  2. Construct confirmed-loss endpoint labels (six supported endpoints,
     persistent crossing with two-scan confirmation, conservative censoring).
  3. Exclude timing-sensitive and non-physical devices (released metadata).
  4. Train one HistGradientBoosting model per endpoint on all eligible
     devices and export the bundle with training tertiles and feature ranges.

The published frozen bundle in model/ was produced by exactly this
procedure; rerunning this script on the released data reproduces its
predictions bit-for-bit with the pinned scikit-learn version.

Usage
-----
    python src/train_risk_models.py                 # writes model_retrained/
    python src/train_risk_models.py --outdir model  # overwrite released bundle
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import (DESCRIPTORS, DEVICE, FEATURES, SUPPORTED_ENDPOINTS,
                            build_endpoint_labels, build_sequence_features)

REPO = Path(__file__).resolve().parent.parent
SCANS_FILE = REPO / "data" / "jv_scans.csv"
METADATA_FILE = REPO / "data" / "device_metadata.csv"
RANDOM_SEED = 42


def hgb_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.04, max_iter=250, max_leaf_nodes=10,
            min_samples_leaf=15, l2_regularization=2.0,
            random_state=RANDOM_SEED)),
    ])


def to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--outdir", default=str(REPO / "model_retrained"))
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scans = pd.read_csv(SCANS_FILE)
    metadata = pd.read_csv(METADATA_FILE)
    metadata["timing_sensitive"] = to_bool(metadata["timing_sensitive"])
    metadata["pce0_physical"] = to_bool(metadata["pce0_physical"])

    sequence = build_sequence_features(scans)
    labels = build_endpoint_labels(scans, sequence)
    base = (sequence.merge(labels, on=DEVICE, how="inner")
                    .merge(metadata[[DEVICE, "timing_sensitive", "pce0_physical"]],
                           on=DEVICE, how="left"))
    base = base.loc[~base["timing_sensitive"].fillna(False)
                    & base["pce0_physical"].fillna(True)].copy()

    models, manifest = {}, []
    for endpoint, config in SUPPORTED_ENDPOINTS.items():
        known = base[f"{endpoint}__known"].fillna(False)
        training = base.loc[known & base[f"{endpoint}__label"].notna()].dropna(subset=FEATURES)
        y = training[f"{endpoint}__label"].astype(int)
        if y.nunique() < 2:
            raise RuntimeError(f"Cannot train {endpoint}: one class only.")
        fitted = hgb_model()
        fitted.fit(training[FEATURES], y)
        in_archive = fitted.predict_proba(training[FEATURES])[:, 1]
        q33, q67 = np.quantile(in_archive, [1 / 3, 2 / 3])
        feature_ranges = {f: {"min": float(training[f].min()), "max": float(training[f].max()),
                              "q01": float(training[f].quantile(.01)),
                              "q99": float(training[f].quantile(.99))} for f in FEATURES}
        models[endpoint] = {"model": fitted,
                            "risk_cutpoints": {"low_to_intermediate": float(q33),
                                               "intermediate_to_high": float(q67)},
                            "feature_ranges": feature_ranges,
                            "training_n": int(len(training)),
                            "training_events": int(y.sum()),
                            "endpoint": config}
        manifest.append({"endpoint": endpoint, "label": config["label"],
                         "threshold_fraction": config["fraction"], "horizon_h": config["horizon_h"],
                         "training_n": len(training), "training_events": int(y.sum()),
                         "training_non_events": int((1 - y).sum()),
                         "risk_cut_low_to_intermediate": q33, "risk_cut_intermediate_to_high": q67})

    bundle = {
        "bundle_version": "1.0",
        "purpose": "research-use early J-V long-horizon PCE-loss risk stratification",
        "required_features": FEATURES,
        "descriptor_columns": DESCRIPTORS,
        "preprocessing": {"K_actual_session1_scans": 4,
                          "normalization": "relative change from first Session-1 scan; Rs/Rsh log ratio",
                          "Session1_only": True},
        "models": models,
        "warnings": ["Not a universal lifetime model.",
                     "Do not interpret as causal treatment/passivator effect.",
                     "Scores are supported only for endpoint names in this bundle.",
                     "Out-of-distribution inputs must be flagged and reviewed."],
    }
    joblib.dump(bundle, outdir / "early_JV_risk_grid_bundle.joblib")
    pd.DataFrame(manifest).to_csv(outdir / "early_JV_risk_grid_training_manifest.csv", index=False)
    (outdir / "early_JV_risk_grid_model_card.json").write_text(json.dumps({
        "bundle_version": "1.0",
        "input_requirement": "At least four actual early Session-1 scans per device with "
                             "reverse PCE, Voc, Isc, FF, Rs, Rsh and HI.",
        "supported_endpoints": [{"endpoint": m["endpoint"], "label": m["label"],
                                 "training_n": int(m["training_n"]),
                                 "training_events": int(m["training_events"])} for m in manifest],
        "quality_control": "The inference script flags missing, non-finite and "
                           "outside-training-range features.",
        "interpretation": "Risk labels are training-distribution tertiles; probability is model "
                          "output for ranking/triage, not a guaranteed lifetime estimate.",
        "provenance": {"raw_file": "data/jv_scans.csv", "random_seed": RANDOM_SEED},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(pd.DataFrame(manifest).to_string(index=False))
    print(f"\nBundle exported to {outdir}")


if __name__ == "__main__":
    main()
