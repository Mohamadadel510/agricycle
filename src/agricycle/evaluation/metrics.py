"""Time-series evaluation pipeline for irrigation classification and regression."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def build_irrigation_labels(
    df: pd.DataFrame,
    field_area_m2: float = 4200.0,
    horizon_hours: int = 6,
) -> pd.DataFrame:
    """Create documented agronomic ground-truth targets from underlying physical states.

    Note: The six-hour classification horizon is an experimental design choice.
    Labels represent the noiseless FAO-56 / MWRI physical water balance state.
    """
    out = df.copy().reset_index(drop=True)
    n = len(out)
    pump_arr = out["pump_active"].astype(bool).to_numpy()
    depletion_true = out["depletion_root_zone_mm"].to_numpy(dtype=float)
    raw_arr = out["raw_threshold_mm"].to_numpy(dtype=float)
    ponded_arr = out["ponded_water_depth_mm"].to_numpy(dtype=float)
    canal_open_arr = out["canal_turn_open"].astype(bool).to_numpy()
    is_rice = (out["crop_type"] == "RICE").to_numpy()

    target_6h = np.zeros(n, dtype=int)
    target_hours_until = np.full(n, 168.0, dtype=float)
    target_vol_m3 = np.zeros(n, dtype=float)

    for i in range(n):
        # 1. Look-ahead window for pump_active or physical threshold crossing within 6h
        end_idx = min(n, i + horizon_hours + 1)
        window_pump = pump_arr[i:end_idx].any()

        if is_rice[i]:
            phys_need_6h = canal_open_arr[i] and (ponded_arr[i:end_idx].min() <= 15.0)
            net_need_mm = max(0.0, 50.0 - ponded_arr[i]) if (window_pump or phys_need_6h) else 0.0
            eta = 0.75
        else:
            phys_need_6h = canal_open_arr[i] and (depletion_true[i:end_idx].max() >= raw_arr[i])
            net_need_mm = (
                min(22.0, max(6.0, depletion_true[i] * 0.55))
                if (window_pump or phys_need_6h)
                else 0.0
            )
            eta = 0.90

        target_6h[i] = 1 if (window_pump or phys_need_6h) else 0
        gross_mm = net_need_mm / eta if net_need_mm > 0.0 else 0.0
        target_vol_m3[i] = (gross_mm * field_area_m2) / 1000.0

        # 2. Hours until next irrigation need
        future_indices = np.where(pump_arr[i:] | (depletion_true[i:] >= raw_arr[i]))[0]
        if len(future_indices) > 0:
            target_hours_until[i] = min(168.0, float(future_indices[0]))

    out["target_irrigate_within_6h"] = target_6h
    out["target_hours_until_irrigation"] = np.round(target_hours_until, 2)
    out["target_recommended_volume_m3"] = np.round(target_vol_m3, 3)
    return out


def chronological_train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame strictly chronologically without shuffling."""
    if not (0.1 < train_frac < 0.9 and 0.05 <= val_frac < 0.5):
        raise ValueError("Invalid train_frac or val_frac bounds.")
    if train_frac + val_frac >= 0.95:
        raise ValueError("train_frac + val_frac must leave at least 5% for test split.")

    sorted_df = df.sort_values("timestamp_utc").reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = sorted_df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = sorted_df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = sorted_df.iloc[val_end:].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def evaluate_irrigation_predictions(
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
    y_prob_class: Optional[np.ndarray],
    y_true_hours: np.ndarray,
    y_pred_hours: np.ndarray,
    y_true_vol_m3: np.ndarray,
    y_pred_vol_m3: np.ndarray,
    model_name: str,
    crop_type: str,
    data_provenance: str,
) -> Dict[str, Any]:
    """Compute classification and physical-unit regression metrics."""
    y_t_c = np.asarray(y_true_class, dtype=int)
    y_p_c = np.asarray(y_pred_class, dtype=int)

    precision = float(precision_score(y_t_c, y_p_c, zero_division=0))
    recall = float(recall_score(y_t_c, y_p_c, zero_division=0))
    f1 = float(f1_score(y_t_c, y_p_c, zero_division=0))

    cm = confusion_matrix(y_t_c, y_p_c, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    scores = y_prob_class if y_prob_class is not None else y_p_c
    if len(np.unique(y_t_c)) > 1:
        pr_auc = float(average_precision_score(y_t_c, scores))
    else:
        pr_auc = 1.0 if int(y_t_c[0]) == int(y_p_c[0]) else 0.0

    # Regression metrics in physical units (hours, m^3, and Liters)
    hours_mae = float(mean_absolute_error(y_true_hours, y_pred_hours))
    hours_rmse = float(np.sqrt(mean_squared_error(y_true_hours, y_pred_hours)))

    vol_mae_m3 = float(mean_absolute_error(y_true_vol_m3, y_pred_vol_m3))
    vol_rmse_m3 = float(np.sqrt(mean_squared_error(y_true_vol_m3, y_pred_vol_m3)))

    return {
        "model_name": model_name,
        "crop_type": crop_type,
        "data_provenance": data_provenance,
        "samples": len(y_t_c),
        "positive_labels": int(y_t_c.sum()),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "hours_until_irr_mae_h": round(hours_mae, 2),
        "hours_until_irr_rmse_h": round(hours_rmse, 2),
        "volume_mae_m3": round(vol_mae_m3, 3),
        "volume_rmse_m3": round(vol_rmse_m3, 3),
        "volume_mae_liters": round(vol_mae_m3 * 1000.0, 1),
        "volume_rmse_liters": round(vol_rmse_m3 * 1000.0, 1),
    }