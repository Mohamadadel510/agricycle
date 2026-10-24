"""Unified data schemas, enums, provenance contracts, and physical unit conversions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
from typing import Optional


class CropType(str, Enum):
    CORN = "CORN"
    ONION = "ONION"
    TOMATO = "TOMATO"
    RICE = "RICE"


class GrowthStage(str, Enum):
    INI = "INI"
    DEV = "DEV"
    MID = "MID"
    END = "END"


class HydrologicalRegime(str, Enum):
    OLD_LANDS_CANAL = "OLD_LANDS_CANAL"
    NEW_LANDS_PRESSURIZED = "NEW_LANDS_PRESSURIZED"


class DataProvenance(str, Enum):
    REAL_INSITU_IOT = "REAL_INSITU_IOT"
    REAL_FIELD_RCBD = "REAL_FIELD_RCBD"
    REAL_LAB_VIBRATION = "REAL_LAB_VIBRATION"
    REANALYSIS_ERA5_LAND = "REANALYSIS_ERA5_LAND"
    REANALYSIS_NASA_POWER = "REANALYSIS_NASA_POWER"
    NWP_FORECAST_LIVE = "NWP_FORECAST_LIVE"
    NWP_ARCHIVE_RUNS = "NWP_ARCHIVE_RUNS"
    SYNTHETIC_FAO56 = "SYNTHETIC_FAO56"
    SYNTHETIC_EPANET = "SYNTHETIC_EPANET"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# Verified uncalibrated RS485 NPK columns that must be stripped (SRC-29)
FORBIDDEN_NPK_COLUMNS: frozenset[str] = frozenset(
    {
        "soil_n",
        "soil_p",
        "soil_k",
        "nitrogen_mg_kg",
        "phosphorus_mg_kg",
        "potassium_mg_kg",
        "npk_n",
        "npk_p",
        "npk_k",
        "soil n",
        "soil p",
        "soil k",
    }
)


def wind_10m_to_2m(u_z_ms: float, z_m: float = 10.0) -> float:
    """Convert wind speed at height z_m to standard 2m height using FAO-56 Eq. 47."""
    if u_z_ms < 0.0:
        raise ValueError(f"Wind speed cannot be negative: {u_z_ms}")
    if z_m == 2.0:
        return float(u_z_ms)
    return float(u_z_ms * (4.87 / math.log(67.8 * z_m - 5.42)))


def wh_m2_to_mj_m2(wh_m2: float) -> float:
    """Convert hourly solar radiation from Wh/m^2 (NASA POWER) to MJ/m^2."""
    if wh_m2 < 0.0:
        raise ValueError(f"Solar radiation cannot be negative: {wh_m2}")
    return float(wh_m2 * 0.0036)


def wm2_to_mj_m2_hour(wm2: float) -> float:
    """Convert average hourly irradiance in W/m^2 to MJ/m^2/hour."""
    if wm2 < 0.0:
        raise ValueError(f"Irradiance cannot be negative: {wm2}")
    return float(wm2 * 0.0036)


def us_cm_to_ds_m(ec_us_cm: float) -> float:
    """Convert Electrical Conductivity from uS/cm (RS485 probe) to dS/m."""
    if ec_us_cm < 0.0:
        raise ValueError(f"Electrical conductivity cannot be negative: {ec_us_cm}")
    return float(ec_us_cm / 1000.0)


def pressure_head_m_to_bar(head_m: float) -> float:
    """Convert hydraulic head in meters (BattLeDIM SCADA) to pressure in bar."""
    if head_m < 0.0:
        return 0.0
    return float(head_m * 0.0980665)


@dataclass(frozen=True)
class TelemetryRecord:
    """Validated single-timestep record for AgriCycle inference and simulation."""

    timestamp_utc: datetime
    field_id: str
    crop_type: CropType
    days_after_planting: int
    growth_stage: GrowthStage
    hydrological_regime: HydrologicalRegime
    data_provenance: DataProvenance

    # Weather observations
    temp_air_2m_c: float
    rh_2m_pct: float
    wind_speed_2m_ms: float
    solar_rad_mj_m2_h: float
    precip_mm: float
    et0_fao_mm: float

    # Soil / Root-zone state
    vwc_root_zone_m3m3: float
    soil_temp_c: float
    soil_ec_dsm: float
    soil_ph: Optional[float]
    root_depth_m: float
    depletion_root_zone_mm: float
    ponded_water_depth_mm: float  # Used for Rice; 0.0 for upland crops

    # Canal & Forecast context
    canal_turn_open: bool
    hours_until_canal_close: float
    forecast_et0_next_72h_mm: float
    forecast_max_temp_next_72h_c: float

    # Hydraulic & Pump telemetry
    pump_active: bool
    irrigation_applied_mm: float
    flow_rate_lpm: float
    manifold_pressure_bar: float
    pump_current_amps: float
    vib_rms_mms: float
    anomaly_label: str = "NORMAL"