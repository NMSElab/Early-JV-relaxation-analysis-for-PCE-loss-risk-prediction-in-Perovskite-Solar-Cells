"""
build_features.py
=================

Shared feature-engineering and endpoint-definition functions for the
early J-V relaxation risk model.

Input data model (long format, one row per J-V scan):
    device_id            unique device identifier
    subseries_id         fabrication subseries identifier (grouping unit)
    storage_time_h       time since the first measurement of the device, hours
    session_number       measurement session index (1 = initial session)
    iteration_in_session scan index within the session (1, 2, 3, ...)
    pce_rev_pct          reverse-scan power conversion efficiency, %
    voc_rev_v            reverse-scan open-circuit voltage, V
    isc_rev_ma_cm2       reverse-scan short-circuit current density, mA/cm2
    ff_rev_pct           reverse-scan fill factor, %
    rs_rev_ohm_cm2       reverse-scan series resistance, Ohm*cm2
    rsh_rev_ohm_cm2      reverse-scan shunt resistance, Ohm*cm2
    hysteresis_index     scan hysteresis index (dimensionless)

Feature definition (frozen, identical to the published model):
    For each descriptor d and Session-1 scans i = 1..4 (ordered by
    storage_time_h, then iteration_in_session):
        - PCE, Voc, Isc, FF, HI:  seq__d__iter_i = (x_i - x_1) / |x_1|
        - Rs, Rsh (log-ratio):    seq__d__iter_i = ln(x_i / x_1)
    Only devices with at least MIN_SCANS (4) actual Session-1 scans are
    represented. No information later than the fourth Session-1 scan enters
    the features.

Endpoint definition (frozen):
    Confirmed crossing of fraction x (final valid Session-1 PCE) requiring
    CONFIRM_N = 2 consecutive below-threshold observations; landmark at the
    end of Session 1; devices without a confirmed event and with insufficient
    follow-up are unlabeled (never treated as non-events).
"""

import numpy as np
import pandas as pd

DEVICE = "device_id"
SUBSERIES = "subseries_id"
TIME = "storage_time_h"
SESSION = "session_number"
ITERATION = "iteration_in_session"
PCE = "pce_rev_pct"

DESCRIPTORS = {
    "PCE_rev": "pce_rev_pct",
    "Voc_rev": "voc_rev_v",
    "Isc_rev": "isc_rev_ma_cm2",
    "FF_rev": "ff_rev_pct",
    "Rs_rev": "rs_rev_ohm_cm2",
    "Rsh_rev": "rsh_rev_ohm_cm2",
    "HI": "hysteresis_index",
}
MIN_SCANS = 4
CONFIRM_N = 2
FEATURES = [f"seq__{d}__iter{i}" for d in DESCRIPTORS for i in range(1, MIN_SCANS + 1)]

SUPPORTED_ENDPOINTS = {
    "T90_500h": {"fraction": 0.90, "horizon_h": 500, "label": "T90 within 500 h"},
    "T80_500h": {"fraction": 0.80, "horizon_h": 500, "label": "T80 within 500 h"},
    "T80_1000h": {"fraction": 0.80, "horizon_h": 1000, "label": "T80 within 1000 h"},
    "T70_1000h": {"fraction": 0.70, "horizon_h": 1000, "label": "T70 within 1000 h"},
    "T70_3000h": {"fraction": 0.70, "horizon_h": 3000, "label": "T70 within 3000 h"},
    "T50_3000h": {"fraction": 0.50, "horizon_h": 3000, "label": "T50 within 3000 h"},
}


def require_columns(frame, columns, label="input"):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}\n"
                       f"Available: {frame.columns.tolist()}")


def normalize_descriptor(descriptor, values):
    """Relative change from the first valid scan; log-ratio for Rs/Rsh."""
    values = np.asarray(values, dtype=float)
    output = np.full_like(values, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return output
    first = values[np.where(valid)[0][0]]
    if descriptor in {"Rs_rev", "Rsh_rev"}:
        if first <= 0:
            return output
        positive = valid & (values > 0)
        output[positive] = np.log(values[positive] / first)
        return output
    if first == 0:
        return output
    output[valid] = (values[valid] - first) / abs(first)
    return output


def build_sequence_features(scans):
    """Per-device four-scan Session-1 feature table.

    Parameters
    ----------
    scans : DataFrame in the long input data model (see module docstring).

    Returns
    -------
    DataFrame with one row per device having >= MIN_SCANS Session-1 scans:
    device_id, subseries_id, n_s1_scans, s1_end_h, and the 28 seq__ features.
    """
    require_columns(scans, [DEVICE, TIME, SESSION, ITERATION] + list(DESCRIPTORS.values()))
    frame = scans.copy()
    for column in [TIME, SESSION, ITERATION] + list(DESCRIPTORS.values()):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[DEVICE, TIME, SESSION])
    frame = frame.loc[frame[TIME].ge(0)]
    s1 = frame.loc[frame[SESSION].eq(1)]

    rows = []
    for key, group in s1.groupby(DEVICE, sort=False):
        group = group.sort_values([TIME, ITERATION]).reset_index(drop=True)
        if len(group) < MIN_SCANS:
            continue
        row = {
            DEVICE: key,
            SUBSERIES: group[SUBSERIES].iloc[0] if SUBSERIES in group else np.nan,
            "n_s1_scans": int(len(group)),
            "s1_end_h": float(group[TIME].max()),
        }
        for descriptor, source in DESCRIPTORS.items():
            transformed = normalize_descriptor(descriptor, group[source].to_numpy(float))
            for i in range(1, MIN_SCANS + 1):
                row[f"seq__{descriptor}__iter{i}"] = transformed[i - 1]
        rows.append(row)
    return pd.DataFrame(rows)


def build_trajectories(scans):
    """Full PCE(t) trajectories per device (median PCE per unique time point)."""
    require_columns(scans, [DEVICE, TIME, SESSION, PCE])
    frame = scans.copy()
    for column in [TIME, SESSION, PCE]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    trajectories = {}
    for key, group in frame.groupby(DEVICE, sort=False):
        values = group[[TIME, PCE, SESSION]].dropna(subset=[TIME, PCE])
        values = values.loc[(values[TIME] >= 0) & (values[PCE] > 0)]
        if len(values) < 2:
            continue
        trajectory = (values.groupby(TIME, as_index=False)
                      .agg(pce=(PCE, "median"), session_number=(SESSION, "min"))
                      .sort_values(TIME).reset_index(drop=True))
        if len(trajectory) >= 2:
            trajectories[key] = trajectory
    return trajectories


def confirmed_crossing(times, pces, threshold):
    """First persistent (CONFIRM_N consecutive) crossing below threshold."""
    below = pces <= threshold
    for index in range(1, len(times)):
        if below[index] and pces[index - 1] > threshold:
            confirmation = below[index:min(index + CONFIRM_N, len(times))]
            if len(confirmation) >= CONFIRM_N and confirmation.all():
                t0, t1 = times[index - 1], times[index]
                p0, p1 = pces[index - 1], pces[index]
                event_time = t1 if p0 == p1 else t0 + (threshold - p0) * (t1 - t0) / (p1 - p0)
                return float(event_time), True
    return np.nan, False


def endpoint_label(trajectory, landmark, fraction, horizon):
    """Confirmed-loss endpoint label with conservative censoring.

    Returns (known, label): known=False means unlabeled at this horizon.
    """
    times = trajectory[TIME].to_numpy(float)
    pces = trajectory["pce"].to_numpy(float)
    s1 = trajectory.loc[trajectory["session_number"].eq(1)]
    if s1.empty:  # no valid (positive-PCE) Session-1 point -> no reference
        return False, np.nan
    reference = float(s1.loc[s1[TIME].idxmax(), "pce"])
    event_time, observed = confirmed_crossing(times, pces, fraction * reference)
    if not np.any(times > landmark) or (observed and event_time <= landmark):
        return False, np.nan
    if observed and event_time <= landmark + horizon:
        return True, 1
    if (not observed and times[-1] >= landmark + horizon) or (observed and event_time > landmark + horizon):
        return True, 0
    return False, np.nan


def build_endpoint_labels(scans, sequence):
    """Endpoint label table for all supported endpoints.

    Parameters
    ----------
    scans    : long-format scan table (all sessions, full follow-up).
    sequence : output of build_sequence_features (defines landmark s1_end_h).
    """
    trajectories = build_trajectories(scans)
    rows = []
    for record in sequence.itertuples(index=False):
        key = getattr(record, DEVICE)
        if key not in trajectories:
            continue
        landmark = float(getattr(record, "s1_end_h"))
        result = {DEVICE: key}
        for endpoint, config in SUPPORTED_ENDPOINTS.items():
            known, label = endpoint_label(trajectories[key], landmark,
                                          config["fraction"], config["horizon_h"])
            result[f"{endpoint}__known"] = known
            result[f"{endpoint}__label"] = label
        rows.append(result)
    return pd.DataFrame(rows)
