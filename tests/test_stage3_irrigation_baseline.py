"""Stage 3 automated tests: FAO-56 baseline rules, canal lockouts, fallbacks, and metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from agricycle.crops.profiles import ConfigManager
from agricycle.evaluation.metrics import (
    build_irrigation_labels,
    chronological_train_val_test_split,
    evaluate_irrigation_predictions,
)
from agricycle.irrigation.baseline import RuleBasedIrrigationBaseline
from agricycle.schema import CropType
from agricycle.simulation.simulator import AgriCycleSimulator


def test_chronological_split_prevents_data_leakage() -> None:
    cfg = ConfigManager("config")
    sim = AgriCycleSimulator(cfg, seed=42)
    df = sim.simulate_season(CropType.TOMATO, soil_key="NEW_LANDS_SANDY_LOAM", days=10)

    train_df, val_df, test_df = chronological_train_val_test_split(df, 0.60, 0.20)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert train_df["timestamp_utc"].max() < val_df["timestamp_utc"].min()
    assert val_df["timestamp_utc"].max() < test_df["timestamp_utc"].min()


def test_baseline_canal_closure_lockout_and_pre_closure_refill() -> None:
    cfg = ConfigManager("config")
    baseline = RuleBasedIrrigationBaseline(cfg, field_area_m2=4200.0)

    # Construct 2 synthetic timesteps for Corn in Nile Delta Clay:
    # Row 0: Canal CLOSED (Batala) despite low moisture -> must lockout irrigation
    # Row 1: Canal OPEN with 10h until closure and high forecast ETo -> must trigger pre-closure refill
    test_rows = pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-06-05T12:00:00Z",
                "crop_type": "CORN",
                "days_after_planting": 60,
                "et0_fao_mm": 0.45,
                "wind_speed_2m_ms": 2.5,
                "forecast_et0_next_72h_mm": 18.0,
                "forecast_max_temp_next_72h_c": 34.0,
                "canal_turn_open": False,
                "hours_until_canal_close": 0.0,
                "vwc_root_zone_m3m3": 0.25,  # Very dry (theta_fc = 0.38)
                "ponded_water_depth_mm": 0.0,
                "sensor_quality_flag": "VALID",
                "data_provenance": "SYNTHETIC_FAO56",
            },
            {
                "timestamp_utc": "2026-06-04T14:00:00Z",
                "crop_type": "CORN",
                "days_after_planting": 59,
                "et0_fao_mm": 0.45,
                "wind_speed_2m_ms": 2.5,
                "forecast_et0_next_72h_mm": 22.0,
                "forecast_max_temp_next_72h_c": 34.0,
                "canal_turn_open": True,
                "hours_until_canal_close": 10.0,
                "vwc_root_zone_m3m3": 0.31,  # Approaching RAW before 10-day closure
                "ponded_water_depth_mm": 0.0,
                "sensor_quality_flag": "VALID",
                "data_provenance": "SYNTHETIC_FAO56",
            },
        ]
    )

    preds = baseline.predict_dataframe(test_rows, soil_key="NILE_DELTA_CLAY")
    assert preds.loc[0, "irrigate_now"] == False
    assert preds.loc[0, "rule_triggered"] == "CANAL_CLOSED_BATALA_LOCKOUT"
    assert preds.loc[0, "recommended_volume_m3"] == 0.0

    assert preds.loc[1, "irrigate_now"] == True
    assert preds.loc[1, "rule_triggered"] == "MWRI_PRE_CLOSURE_CANAL_REFILL"
    assert preds.loc[1, "recommended_volume_m3"] > 20.0


def test_baseline_sensor_disconnect_fallback() -> None:
    cfg = ConfigManager("config")
    baseline = RuleBasedIrrigationBaseline(cfg)

    # Row 0: Valid moist reading (0.36 m^3/m^3 in clay)
    # Row 1: Disconnected sensor (-0.05 m^3/m^3, OUT_OF_BOUNDS) -> must NOT trigger false alarm irrigation
    rows = pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-06-10T10:00:00Z",
                "crop_type": "CORN",
                "days_after_planting": 50,
                "et0_fao_mm": 0.35,
                "wind_speed_2m_ms": 2.0,
                "forecast_et0_next_72h_mm": 15.0,
                "forecast_max_temp_next_72h_c": 33.0,
                "canal_turn_open": True,
                "hours_until_canal_close": 48.0,
                "vwc_root_zone_m3m3": 0.36,
                "ponded_water_depth_mm": 0.0,
                "sensor_quality_flag": "VALID",
                "data_provenance": "SYNTHETIC_FAO56",
            },
            {
                "timestamp_utc": "2026-06-10T11:00:00Z",
                "crop_type": "CORN",
                "days_after_planting": 50,
                "et0_fao_mm": 0.35,
                "wind_speed_2m_ms": 2.0,
                "forecast_et0_next_72h_mm": 15.0,
                "forecast_max_temp_next_72h_c": 33.0,
                "canal_turn_open": True,
                "hours_until_canal_close": 47.0,
                "vwc_root_zone_m3m3": -0.05,
                "ponded_water_depth_mm": 0.0,
                "sensor_quality_flag": "OUT_OF_BOUNDS",
                "data_provenance": "SYNTHETIC_FAO56",
            },
        ]
    )

    preds = baseline.predict_dataframe(rows, soil_key="NILE_DELTA_CLAY")
    assert preds.loc[1, "sensor_fallback_used"] == True
    assert preds.loc[1, "irrigate_now"] == False
    assert preds.loc[1, "confidence_score"] < 0.70


def test_rice_paddy_baseline_and_evaluation_metrics() -> None:
    cfg = ConfigManager("config")
    sim = AgriCycleSimulator(cfg, seed=42)
    df_rice = sim.simulate_season(CropType.RICE, soil_key="NILE_DELTA_CLAY", days=15)
    labeled = build_irrigation_labels(df_rice, field_area_m2=4200.0, horizon_hours=6)

    baseline = RuleBasedIrrigationBaseline(cfg)
    preds = baseline.predict_dataframe(labeled, soil_key="NILE_DELTA_CLAY")

    metrics = evaluate_irrigation_predictions(
        y_true_class=labeled["target_irrigate_within_6h"].to_numpy(),
        y_pred_class=preds["irrigate_within_6h"].to_numpy(),
        y_prob_class=preds["irrigate_within_6h"].to_numpy(),
        y_true_hours=labeled["target_hours_until_irrigation"].to_numpy(),
        y_pred_hours=preds["hours_until_next_irrigation"].to_numpy(),
        y_true_vol_m3=labeled["target_recommended_volume_m3"].to_numpy(),
        y_pred_vol_m3=preds["recommended_volume_m3"].to_numpy(),
        model_name="FAO56_Rule_Baseline",
        crop_type="RICE",
        data_provenance="SYNTHETIC_FAO56",
    )
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert metrics["volume_mae_liters"] == pytest.approx(metrics["volume_mae_m3"] * 1000.0, rel=1e-2)