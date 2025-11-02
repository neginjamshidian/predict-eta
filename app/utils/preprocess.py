
# utils/preprocess.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

def convert_weight_to_kg(weight):
    """Best-effort conversion for strings like '2.2LBS', '1,234 KG', '12oz', or numeric."""
    if isinstance(weight, (int, float)) and not pd.isna(weight):
        return float(weight)
    if not isinstance(weight, str) or not weight:
        return np.nan
    s = weight.strip().upper().replace(",", "")
    if s.endswith(("S", "Z")) and not s.endswith("LBS"):
        s = s[:-1].strip()
    try:
        if "LBS" in s:
            return float(s.replace("LBS", "").strip()) * 0.453592
        if "KG" in s:
            return float(s.replace("KG", "").strip())
        if "OZ" in s:
            return float(s.replace("OZ", "").strip()) * 0.0283495
        return float(s)
    except Exception:
        return np.nan

def process_travel_history(travel_history: Any) -> Tuple[float, float, pd.Timestamp, pd.Timestamp, float]:
    """Compute features from a travel history list.

    Returns:
        num_events, time_in_transit_hours, last_event_time, first_event_time, avg_diff_hours
    """
    if not isinstance(travel_history, list) or len(travel_history) == 0:
        return np.nan, np.nan, pd.NaT, pd.NaT, np.nan

    # Normalize timestamps
    events = []
    for ev in travel_history:
        if isinstance(ev, dict) and "datetime" in ev:
            dt = ev["datetime"]
            if isinstance(dt, dict) and "$date" in dt:
                ts = pd.to_datetime(dt["$date"], errors="coerce")
            else:
                ts = pd.to_datetime(dt, errors="coerce")
            if pd.notna(ts):
                events.append({"datetime": ts})
    if not events:
        return np.nan, np.nan, pd.NaT, pd.NaT, np.nan

    # sort ascending
    events = sorted(events, key=lambda e: e["datetime"])
    first = events[0]["datetime"]
    last = events[-1]["datetime"]
    time_in_transit = (last - first).total_seconds() / 3600.0

    # diffs
    diffs = []
    for i in range(len(events) - 1):
        diff_h = (events[i + 1]["datetime"] - events[i]["datetime"]).total_seconds() / 3600.0
        diffs.append(diff_h)
    avg_diff = float(np.mean(diffs)) if diffs else np.nan

    return float(len(events)), float(time_in_transit), last, first, avg_diff

def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep and type-cast the modeling columns safely."""
    required = ["weight_courier_raw", "travel_diff_times", "time_diff_label_till_shippingdate", "time_in_transit"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    # type conversions
    df["weight_courier_raw"] = df["weight_courier_raw"].apply(convert_weight_to_kg).astype(float)
    for c in ["travel_diff_times", "time_diff_label_till_shippingdate", "time_in_transit"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[required].dropna()
