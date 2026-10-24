"""Live and cached ingestion clients for NASA POWER, Open-Meteo ERA5-Land, and Mendeley IoT CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import requests

from agricycle.crops.evapotrans import compute_fao56_hourly_et0
from agricycle.data.preprocessing import clean_and_validate_dataframe
from agricycle.schema import (
    DataProvenance,
    wh_m2_to_mj_m2,
    wind_10m_to_2m,
    wm2_to_mj_m2_hour,
)


def fetch_nasa_power_hourly(
    latitude: float = 31.0409,
    longitude: float = 31.3785,
    start_yyyymmdd: str = "20250501",
    end_yyyymmdd: str = "20250510",
    save_path: Optional[Path] = None,
    timeout_s: int = 15,
) -> pd.DataFrame:
    """Fetch verified hourly agroclimatology data from NASA POWER Point API (SRC-01).

    Verified parameters: ALLSKY_SFC_SW_DWN (Wh/m^2), T2M (°C), RH2M (%),
    T2MDEW (°C), PRECTOTCORR (mm), WS2M (m/s). Missing values are -999.
    """
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,RH2M,T2MDEW,PRECTOTCORR,WS2M",
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_yyyymmdd,
        "end": end_yyyymmdd,
        "format": "JSON",
        "time-standard": "UTC",
    }
    response = requests.get(url, params=params, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()

    param_dict: Dict[str, Dict[str, float]] = payload["properties"]["parameter"]
    timestamps = sorted(param_dict["T2M"].keys())

    rows = []
    for ts_key in timestamps:
        # NASA POWER hourly timestamp format: YYYYMMDDHH
        dt_utc = pd.to_datetime(ts_key, format="%Y%m%d%H", utc=True)
        sw_wh = float(param_dict["ALLSKY_SFC_SW_DWN"].get(ts_key, -999.0))
        t2m = float(param_dict["T2M"].get(ts_key, -999.0))
        rh2m = float(param_dict["RH2M"].get(ts_key, -999.0))
        tdew = float(param_dict["T2MDEW"].get(ts_key, -999.0))
        prec = float(param_dict["PRECTOTCORR"].get(ts_key, -999.0))
        ws2m = float(param_dict["WS2M"].get(ts_key, -999.0))

        solar_mj = wh_m2_to_mj_m2(sw_wh) if sw_wh >= 0.0 else np.nan
        et0_mm = (
            compute_fao56_hourly_et0(t2m, rh2m, max(0.1, ws2m), solar_mj)
            if (t2m > -900 and rh2m > -900 and ws2m >= 0 and not np.isnan(solar_mj))
            else np.nan
        )

        rows.append(
            {
                "timestamp_utc": dt_utc.isoformat(),
                "field_id": "NASA_POWER_MANSOURA",
                "temp_air_2m_c": t2m,
                "rh_2m_pct": rh2m,
                "dew_point_2m_c": tdew,
                "wind_speed_2m_ms": ws2m,
                "solar_rad_mj_m2_h": solar_mj,
                "precip_mm": prec,
                "et0_fao_mm": et0_mm,
                "data_provenance": DataProvenance.REANALYSIS_NASA_POWER.value,
            }
        )

    df = pd.DataFrame(rows)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


def fetch_open_meteo_era5_land(
    latitude: float = 31.0409,
    longitude: float = 31.3785,
    start_date: str = "2025-05-01",
    end_date: str = "2025-05-10",
    save_path: Optional[Path] = None,
    timeout_s: int = 15,
) -> pd.DataFrame:
    """Fetch verified hourly reanalysis from Open-Meteo ERA5-Land API (SRC-06)."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "models": "era5_land",
        "hourly": (
            "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,"
            "wind_speed_10m,shortwave_radiation,et0_fao_evapotranspiration,"
            "soil_temperature_0_to_7cm,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm"
        ),
        "timezone": "UTC",
    }
    response = requests.get(url, params=params, timeout=timeout_s)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    rows = []
    for i, ts_str in enumerate(hourly["time"]):
        u10 = float(hourly["wind_speed_10m"][i] or 0.0) / 3.6  # km/h -> m/s
        u2 = wind_10m_to_2m(u10, z_m=10.0)
        sw_wm2 = float(hourly["shortwave_radiation"][i] or 0.0)
        solar_mj = wm2_to_mj_m2_hour(sw_wm2)
        t2m = float(hourly["temperature_2m"][i])
        rh = float(hourly["relative_humidity_2m"][i])
        et0_api = hourly.get("et0_fao_evapotranspiration", [None])[i]
        et0_val = (
            float(et0_api)
            if et0_api is not None
            else compute_fao56_hourly_et0(t2m, rh, u2, solar_mj)
        )

        rows.append(
            {
                "timestamp_utc": pd.to_datetime(ts_str, utc=True).isoformat(),
                "field_id": "ERA5_LAND_MANSOURA",
                "temp_air_2m_c": t2m,
                "rh_2m_pct": rh,
                "dew_point_2m_c": float(hourly["dew_point_2m"][i]),
                "wind_speed_2m_ms": round(u2, 3),
                "solar_rad_mj_m2_h": round(solar_mj, 4),
                "precip_mm": float(hourly["precipitation"][i] or 0.0),
                "et0_fao_mm": round(et0_val, 4),
                "soil_temp_c": float(hourly["soil_temperature_0_to_7cm"][i] or t2m),
                "vwc_root_zone_m3m3": float(hourly["soil_moisture_7_to_28cm"][i] or 0.25),
                "data_provenance": DataProvenance.REANALYSIS_ERA5_LAND.value,
            }
        )

    df = pd.DataFrame(rows)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


def ingest_mendeley_tomato_csv(
    csv_path: Path | str,
    sensor_bounds: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Ingest and clean Mendeley Tomato IoT Dataset (SRC-14, DOI: 10.17632/33cngpcrmx.3).

    Explicitly strips unverified 5V RS485 Soil N, P, K columns per hardware audit SRC-29.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Mendeley Tomato dataset file not found at {path}. "
            "Download 'tomato irrigation dataset.csv' from https://data.mendeley.com/datasets/33cngpcrmx"
        )

    raw_df = pd.read_csv(path)
    col_map = {
        "Timestamp": "timestamp_utc",
        "Date": "timestamp_utc",
        "Soil Moisture": "vwc_root_zone_m3m3",
        "Temp": "temp_air_2m_c",
        "Temperature": "temp_air_2m_c",
        "Humidity": "rh_2m_pct",
        "Wind Speed": "wind_speed_2m_ms",
        "Solar Radiation": "solar_rad_mj_m2_h",
    }
    renamed = raw_df.rename(columns={k: v for k, v in col_map.items() if k in raw_df.columns})
    if "field_id" not in renamed.columns:
        renamed["field_id"] = "MENDELEY_TOMATO_IOT"
    if "timestamp_utc" not in renamed.columns:
        renamed["timestamp_utc"] = pd.date_range(
            "2024-11-01", periods=len(renamed), freq="h", tz="UTC"
        )

    # Convert percentage soil moisture (0-100%) to volumetric m^3/m^3 if scaled > 1.0
    if "vwc_root_zone_m3m3" in renamed.columns and renamed["vwc_root_zone_m3m3"].max() > 1.0:
        renamed["vwc_root_zone_m3m3"] = renamed["vwc_root_zone_m3m3"] / 100.0

    return clean_and_validate_dataframe(
        renamed,
        sensor_bounds=sensor_bounds,
        provenance_tag=DataProvenance.REAL_INSITU_IOT.value,
    )