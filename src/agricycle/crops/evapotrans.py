"""FAO-56 Penman-Monteith ETo, Hargreaves fallback, and Dual-Kc (Kcb + Ke) equations."""

from __future__ import annotations

import math
from typing import Optional


def compute_fao56_hourly_et0(
    temp_c: float,
    rh_pct: float,
    wind_2m_ms: float,
    solar_rad_mj_m2_h: float,
    elevation_m: float = 15.0,
) -> float:
    """Compute hourly reference evapotranspiration ETo (mm/hour) per FAO-56 Eq. 53.

    Assumes Mansoura, Nile Delta default elevation = 15.0 m.
    """
    # Atmospheric pressure P (kPa) and psychrometric constant gamma (kPa/°C)
    pressure_kpa = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    gamma = 0.000665 * pressure_kpa

    # Saturation vapor pressure e_s (kPa) and slope of vapor pressure curve delta
    e_s = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    delta = (4098.0 * e_s) / ((temp_c + 237.3) ** 2)

    # Actual vapor pressure e_a (kPa)
    rh_clamped = max(1.0, min(100.0, rh_pct))
    e_a = e_s * (rh_clamped / 100.0)
    vpd = max(0.0, e_s - e_a)

    # Net radiation approximation R_n (MJ/m^2/h) with albedo = 0.23
    r_ns = (1.0 - 0.23) * max(0.0, solar_rad_mj_m2_h)
    r_nl = 0.15 * r_ns if solar_rad_mj_m2_h > 0.05 else 0.08
    r_n = max(0.0, r_ns - r_nl)

    # Soil heat flux G (MJ/m^2/h): 0.1 * R_n during daylight, 0.5 * R_n at night (FAO-56)
    g_flux = 0.10 * r_n if solar_rad_mj_m2_h > 0.05 else 0.50 * r_n

    # Daytime vs nighttime surface resistance coefficient Cn/Cd
    cd = 0.24 if solar_rad_mj_m2_h > 0.05 else 0.96
    u2 = max(0.1, wind_2m_ms)

    numerator = (
        0.408 * delta * (r_n - g_flux)
        + gamma * (37.0 / (temp_c + 273.0)) * u2 * vpd
    )
    denominator = delta + gamma * (1.0 + cd * u2)
    et0 = numerator / denominator
    return float(max(0.0, et0))


def compute_hargreaves_daily_et0(
    t_min_c: float,
    t_max_c: float,
    t_mean_c: float,
    extraterrestrial_ra_mj_m2_d: float = 38.5,
) -> float:
    """Documented FAO-56 fallback (Eq. 52) when RH, solar, or wind telemetry is missing."""
    temp_range = max(0.0, t_max_c - t_min_c)
    et0_daily = 0.0023 * (t_mean_c + 17.8) * math.sqrt(temp_range) * (
        0.408 * extraterrestrial_ra_mj_m2_d
    )
    return float(max(0.0, et0_daily))


def adjust_kc_for_arid_climate(
    kc_table: float,
    wind_2m_ms: float,
    rh_min_pct: float,
    crop_height_m: float,
) -> float:
    """Apply FAO-56 Eq. 62 arid/semi-arid climate adjustment when Kc > 0.45."""
    if kc_table <= 0.45:
        return float(kc_table)
    u2_clamped = max(1.0, min(6.0, wind_2m_ms))
    rh_clamped = max(20.0, min(80.0, rh_min_pct))
    h_clamped = max(0.1, min(10.0, crop_height_m))
    adjustment = (
        0.04 * (u2_clamped - 2.0) - 0.004 * (rh_clamped - 45.0)
    ) * ((h_clamped / 3.0) ** 0.3)
    return float(max(0.15, kc_table + adjustment))


def compute_dual_kc_etc(
    et0_mm: float,
    kcb_basal: float,
    kc_max: float,
    wetted_fraction_fw: float,
    topsoil_depletion_mm: float,
    rew_mm: float,
    tew_mm: float,
    root_depletion_dr_mm: float,
    taw_mm: float,
    raw_mm: float,
) -> tuple[float, float, float]:
    """Compute crop evapotranspiration ETc = (Ks * Kcb + Ke) * ETo per FAO-56 Dual-Kc.

    Explicitly accounts for wetted surface fraction fw under drip/furrow irrigation
    to prevent the 11.7%-25.2% single-Kc overestimation verified in SRC-31.
    Returns (etc_mm, ks_stress, ke_evap).
    """
    # Water stress coefficient Ks (FAO-56 Eq. 84)
    if root_depletion_dr_mm <= raw_mm or taw_mm <= raw_mm:
        ks = 1.0
    elif root_depletion_dr_mm >= taw_mm:
        ks = 0.0
    else:
        ks = (taw_mm - root_depletion_dr_mm) / max(1e-6, (taw_mm - raw_mm))

    # Topsoil evaporation reduction coefficient Kr (FAO-56 Eq. 74)
    if topsoil_depletion_mm <= rew_mm or tew_mm <= rew_mm:
        kr = 1.0
    elif topsoil_depletion_mm >= tew_mm:
        kr = 0.0
    else:
        kr = (tew_mm - topsoil_depletion_mm) / max(1e-6, (tew_mm - rew_mm))

    # Exposed and wetted soil fraction few
    few = max(0.05, min(1.0, wetted_fraction_fw))
    ke_unbounded = kr * max(0.0, kc_max - ks * kcb_basal)
    ke = min(ke_unbounded, few * kc_max)

    etc_mm = max(0.0, (ks * kcb_basal + ke) * et0_mm)
    return float(etc_mm), float(ks), float(ke)