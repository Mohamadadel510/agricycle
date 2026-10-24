"""Stage 2 automated unit tests: crop profiles, units, preprocessing, and simulator."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from agricycle.crops.evapotrans import (
    compute_dual_kc_etc,
    compute_fao56_hourly_et0,
    compute_hargreaves_daily_et0,
)
from agricycle.crops.profiles import ConfigManager
from agricycle.data.preprocessing import clean_and_validate_dataframe
from agricycle.schema import (
    CropType,
    DataProvenance,
    GrowthStage,
    HydrologicalRegime,
    pressure_head_m_to_bar,
    us_cm_to_ds_m,
    wh_m2_to_mj_m2,
    wind_10m_to_2m,
)
from agricycle.simulation.simulator import AgriCycleSimulator


def test_unit_conversions() -> None:
    # FAO-56 Eq. 47 wind conversion at 10m -> 2m factor is ~0.748
    u2 = wind_10m_to_2m(10.0, z_m=10.0)
    assert math.isclose(u2, 7.4797, rel_tol=1e-3)
    assert math.isclose(wh_m2_to_mj_m2(1000.0), 3.6, rel_tol=1e-5)
    assert math.isclose(us_cm_to_ds_m(2500.0), 2.5, rel_tol=1e-5)
    assert math.isclose(pressure_head_m_to_bar(20.0), 1.96133, rel_tol=1e-4)

    with pytest.raises(ValueError):
        wind_10m_to_2m(-1.0)


def test_crop_profiles_and_rice_transplanting_stratification() -> None:
    cfg = ConfigManager("config")

    # Verify all 4 crops resolve properly
    corn_mid = cfg.resolve_crop_stage(CropType.CORN, days_after_planting=80)
    assert corn_mid.stage == GrowthStage.MID
    assert math.isclose(corn_mid.kc_single, 1.20)
    assert math.isclose(corn_mid.root_depth_m, 0.60)
    assert not corn_mid.is_paddy

    # Verify Rice May vs June transplanting Kc shift (SRC-13)
    rice_may = cfg.resolve_crop_stage(
        CropType.RICE, days_after_planting=85, transplanting_window="MAY_EARLY"
    )
    rice_june = cfg.resolve_crop_stage(
        CropType.RICE, days_after_planting=85, transplanting_window="JUNE_LATE"
    )
    assert rice_may.is_paddy
    assert math.isclose(rice_may.kc_single, 1.36)
    assert math.isclose(rice_june.kc_single, 1.26)


def test_mwri_canal_rotation_schedule() -> None:
    cfg = ConfigManager("config")
    # Rice operates on 4 days ON (Amala) / 6 days OFF (Batala) = 10-day cycle (SRC-11)
    open_day4, h_left = cfg.get_canal_status(
        CropType.RICE, HydrologicalRegime.OLD_LANDS_CANAL, day_of_season=4, hour_of_day=18
    )
    closed_day5, _ = cfg.get_canal_status(
        CropType.RICE, HydrologicalRegime.OLD_LANDS_CANAL, day_of_season=5, hour_of_day=8
    )
    assert open_day4 is True
    assert math.isclose(h_left, 6.0)
    assert closed_day5 is False


def test_dual_kc_wetted_fraction_reduces_evaporation() -> None:
    et0 = compute_fao56_hourly_et0(34.0, 35.0, 2.5, 2.8)
    assert et0 > 0.3

    # Drip (fw = 0.35) vs Flood (fw = 1.0) during initial stage (low Kcb, wet topsoil)
    etc_drip, _, ke_drip = compute_dual_kc_etc(
        et0_mm=et0,
        kcb_basal=0.15,
        kc_max=1.20,
        wetted_fraction_fw=0.35,
        topsoil_depletion_mm=2.0,
        rew_mm=8.0,
        tew_mm=22.0,
        root_depletion_dr_mm=10.0,
        taw_mm=80.0,
        raw_mm=40.0,
    )
    etc_flood, _, ke_flood = compute_dual_kc_etc(
        et0_mm=et0,
        kcb_basal=0.15,
        kc_max=1.20,
        wetted_fraction_fw=1.00,
        topsoil_depletion_mm=2.0,
        rew_mm=8.0,
        tew_mm=22.0,
        root_depletion_dr_mm=10.0,
        taw_mm=80.0,
        raw_mm=40.0,
    )
    assert ke_drip < ke_flood
    assert etc_drip < etc_flood

    # Hargreaves fallback sanity check
    et0_harg = compute_hargreaves_daily_et0(20.0, 36.0, 28.0)
    assert 4.0 <= et0_harg <= 10.0


def test_preprocessing_strips_rs485_npk_and_flags_999() -> None:
    cfg = ConfigManager("config")
    raw = pd.DataFrame(
        {
            "timestamp_utc": [
                "2026-06-01T00:00:00Z",
                "2026-06-01T00:00:00Z",  # Duplicate
                "2026-06-01T01:00:00Z",
            ],
            "field_id": ["F1", "F1", "F1"],
            "temp_air_2m_c": [28.5, 28.5, -999.0],  # NASA POWER -999 sentinel
            "vwc_root_zone_m3m3": [0.32, 0.32, -0.05],  # Out-of-bounds disconnect
            "Soil_N": [140, 140, 145],  # Unverified RS485 NPK register (SRC-29)
            "Soil_P": [45, 45, 48],
            "Soil_K": [190, 190, 195],
        }
    )

    cleaned, audit = clean_and_validate_dataframe(
        raw,
        sensor_bounds=cfg.thresholds_cfg["sensor_bounds"],
        provenance_tag=DataProvenance.REAL_INSITU_IOT.value,
    )

    assert len(cleaned) == 2
    assert audit["duplicates_removed"] == 1
    assert set(audit["dropped_npk_columns"]) == {"Soil_N", "Soil_P", "Soil_K"}
    assert "Soil_N" not in cleaned.columns
    assert audit["nasa_power_999_replaced"] == 1
    assert np.isnan(cleaned.loc[1, "temp_air_2m_c"])
    assert cleaned.loc[1, "sensor_quality_flag"] == "OUT_OF_BOUNDS"


def test_simulator_determinism_and_upland_vs_rice_physics() -> None:
    cfg = ConfigManager("config")
    sim_a = AgriCycleSimulator(cfg, seed=123)
    sim_b = AgriCycleSimulator(cfg, seed=123)

    df_corn_a = sim_a.simulate_season(CropType.CORN, days=15)
    df_corn_b = sim_b.simulate_season(CropType.CORN, days=15)
    pd.testing.assert_frame_equal(df_corn_a, df_corn_b)

    df_rice = sim_a.simulate_season(CropType.RICE, days=15)
    # Rice must track standing ponded water depth > 0, whereas Corn ponded depth is always 0
    assert (df_corn_a["ponded_water_depth_mm"] == 0.0).all()
    assert (df_rice["ponded_water_depth_mm"] > 0.0).any()
    assert (df_corn_a["data_provenance"] == DataProvenance.SYNTHETIC_FAO56.value).all()