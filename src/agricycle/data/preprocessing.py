"""Data validation, -999 NASA POWER filtering, RS485 NPK removal, and normalization."""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from agricycle.schema import FORBIDDEN_NPK_COLUMNS


def clean_and_validate_dataframe(
    df: pd.DataFrame,
    sensor_bounds: Dict[str, Any],
    provenance_tag: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean, validate, normalize timestamps, and strip invalid RS485 NPK columns.

    Never silently replaces missing real-world observations with synthetic values.
    Returns (cleaned_df, audit_report).
    """
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    out = df.copy()
    audit: Dict[str, Any] = {
        "initial_rows": len(out),
        "dropped_npk_columns": [],
        "duplicates_removed": 0,
        "nasa_power_999_replaced": 0,
        "out_of_bounds_flagged": 0,
        "provenance": provenance_tag,
    }

    # 1. Strip physically unverified RS485 Modbus NPK columns (SRC-29 audit)
    cols_to_drop = [
        col for col in out.columns if col.strip().lower() in FORBIDDEN_NPK_COLUMNS
    ]
    if cols_to_drop:
        out.drop(columns=cols_to_drop, inplace=True)
        audit["dropped_npk_columns"] = cols_to_drop

    # 2. Normalize timestamp_utc to UTC and remove exact duplicate timestamps
    if "timestamp_utc" not in out.columns:
        raise KeyError("DataFrame must include a 'timestamp_utc' column.")
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    before_dedup = len(out)
    out = out.drop_duplicates(subset=["timestamp_utc", "field_id"], keep="first")
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    audit["duplicates_removed"] = int(before_dedup - len(out))

    # 3. Replace NASA POWER -999 / -999.0 missing value flags with NaN (SRC-01)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    mask_999 = out[numeric_cols].isin([-999, -999.0])
    audit["nasa_power_999_replaced"] = int(mask_999.sum().sum())
    out[numeric_cols] = out[numeric_cols].replace({-999: np.nan, -999.0: np.nan})

    # 4. Physical range outlier checks (flag rather than silently overwrite)
    bounds_map = {
        "vwc_root_zone_m3m3": (
            sensor_bounds["vwc_min_m3m3"],
            sensor_bounds["vwc_max_m3m3"],
        ),
        "temp_air_2m_c": (
            sensor_bounds["air_temp_min_c"],
            sensor_bounds["air_temp_max_c"],
        ),
        "soil_temp_c": (
            sensor_bounds["soil_temp_min_c"],
            sensor_bounds["soil_temp_max_c"],
        ),
        "rh_2m_pct": (sensor_bounds["rh_min_pct"], sensor_bounds["rh_max_pct"]),
        "soil_ec_dsm": (sensor_bounds["ec_min_dsm"], sensor_bounds["ec_max_dsm"]),
    }

    outlier_mask = pd.Series(False, index=out.index)
    for col, (low, high) in bounds_map.items():
        if col in out.columns:
            invalid = (out[col].notna()) & ((out[col] < low) | (out[col] > high))
            outlier_mask = outlier_mask | invalid

    out["sensor_quality_flag"] = np.where(
        outlier_mask,
        "OUT_OF_BOUNDS",
        np.where(out[numeric_cols].isna().any(axis=1), "MISSING_VALUE", "VALID"),
    )
    audit["out_of_bounds_flagged"] = int(outlier_mask.sum())

    # 5. Enforce provenance tag
    out["data_provenance"] = provenance_tag
    return out, audit