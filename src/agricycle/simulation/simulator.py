"""Deterministic hourly simulator for Corn, Onions, Tomatoes, and Paddy Rice."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from agricycle.crops.evapotrans import (
    adjust_kc_for_arid_climate,
    compute_dual_kc_etc,
    compute_fao56_hourly_et0,
)
from agricycle.crops.profiles import ConfigManager
from agricycle.schema import CropType, DataProvenance, HydrologicalRegime


class AgriCycleSimulator:
    """Generates deterministic hourly time-series with explicit FAO-56 & BattLeDIM physics.

    All outputs are explicitly tagged as SYNTHETIC_FAO56.
    """

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        seed: int = 42,
    ) -> None:
        self.cfg = config_manager or ConfigManager()
        self.seed = int(seed)

    def simulate_season(
        self,
        crop_type: CropType,
        soil_key: str = "NILE_DELTA_CLAY",
        days: int = 30,
        start_dap: int = 20,
        inject_anomalies: bool = True,
        transplanting_window: str = "MAY_EARLY",
    ) -> pd.DataFrame:
        """Simulate an hourly agro-hydrological and equipment telemetry dataset."""
        rng = np.random.default_rng(self.seed)
        soil = self.cfg.get_soil_profile(soil_key)
        crop_cfg = self.cfg.crops_cfg[crop_type.value]
        thresh = self.cfg.thresholds_cfg
        regime = HydrologicalRegime(soil["regime"])
        is_paddy = bool(crop_cfg["is_paddy"])

        irr_method = crop_cfg["default_irrigation_method"]
        fw = float(crop_cfg["wetted_fraction_fw"][irr_method])
        eta_irr = float(crop_cfg["irrigation_efficiency"][irr_method])

        theta_fc = float(soil["theta_fc_m3m3"])
        theta_pwp = float(soil["theta_pwp_m3m3"])
        theta_sat = float(soil["theta_sat_m3m3"])
        rew_mm = float(soil["rew_mm"])
        tew_mm = float(soil["tew_mm"])

        # Initialize state variables
        depletion_dr_mm = 12.0
        topsoil_depletion_mm = 4.0
        ponded_depth_mm = 45.0 if is_paddy else 0.0
        hours_since_last_irr = 36.0
        base_time = datetime(2026, 5, 15, 0, 0, tzinfo=timezone.utc)

        records: List[Dict[str, object]] = []
        total_hours = days * 24

        for step_h in range(total_hours):
            ts = base_time + timedelta(hours=step_h)
            day_idx = (step_h // 24) + 1
            hour_of_day = step_h % 24
            dap = start_dap + day_idx - 1

            resolved = self.cfg.resolve_crop_stage(
                crop_type=crop_type,
                days_after_planting=dap,
                transplanting_window=transplanting_window,
            )

            # 1. Diurnal Mansoura/Nile Delta meteorological simulation
            # Inject a 3-day CLAC heat wave on days 12-14 if inject_anomalies is True
            in_heat_wave = inject_anomalies and (12 <= day_idx <= 14)
            heat_boost_c = 7.5 if in_heat_wave else 0.0

            diurnal_phase = math.sin(math.pi * (hour_of_day - 6) / 12.0)
            t_mean = 27.5 + 0.04 * day_idx + heat_boost_c
            temp_air_c = (
                t_mean
                + 7.0 * diurnal_phase
                + float(rng.normal(0.0, 0.45))
            )
            rh_pct = float(
                np.clip(
                    58.0 - 22.0 * diurnal_phase - (12.0 if in_heat_wave else 0.0)
                    + rng.normal(0.0, 2.5),
                    12.0,
                    94.0,
                )
            )
            wind_2m_ms = float(
                np.clip(2.1 + 0.9 * max(0.0, diurnal_phase) + rng.normal(0.0, 0.25), 0.4, 7.5)
            )

            if 6 <= hour_of_day <= 18:
                solar_rad_mj = max(
                    0.0,
                    3.1 * math.sin(math.pi * (hour_of_day - 6) / 12.0)
                    * float(rng.uniform(0.92, 1.03)),
                )
            else:
                solar_rad_mj = 0.0

            # Occasional light rainfall event on day 6
            precip_mm = 2.5 if (day_idx == 6 and hour_of_day in (3, 4)) else 0.0
            eff_precip_mm = 0.80 * precip_mm

            # 2. FAO-56 ETo and Dual-Kc ETc calculation
            et0_h = compute_fao56_hourly_et0(
                temp_c=temp_air_c,
                rh_pct=rh_pct,
                wind_2m_ms=wind_2m_ms,
                solar_rad_mj_m2_h=solar_rad_mj,
            )

            taw_mm = 1000.0 * (theta_fc - theta_pwp) * resolved.root_depth_m
            raw_mm = resolved.depletion_fraction_p * taw_mm

            canal_open, hours_until_close = self.cfg.get_canal_status(
                crop_type=crop_type,
                regime=regime,
                day_of_season=day_idx,
                hour_of_day=hour_of_day,
            )

            # Look-ahead 72h forecast estimates (with small forecast noise)
            forecast_et0_72h = float(max(5.0, et0_h * 72.0 * rng.uniform(0.95, 1.08) + 14.5))
            forecast_max_t_72h = float(
                t_mean + 7.5 + (6.5 if (inject_anomalies and 9 <= day_idx <= 13) else 0.0)
            )

            # 3. Determine whether irrigation is triggered and simulate water balance
            pump_active = False
            irr_net_mm = 0.0
            anomaly_label = "NORMAL"

            if is_paddy:
                # Paddy Rice water balance (ponded depth + deep percolation)
                kc_adj = adjust_kc_for_arid_climate(
                    resolved.kc_single, wind_2m_ms, 30.0, resolved.crop_height_m
                )
                etc_h = et0_h * kc_adj
                perc_h = float(
                    crop_cfg["paddy_management"]["deep_percolation_mm_day"] / 24.0
                )

                min_trigger = (
                    float(crop_cfg["paddy_management"]["flowering_min_depth_mm"])
                    if resolved.stage.value == "MID"
                    else float(crop_cfg["paddy_management"]["awd_trigger_depth_mm"])
                )
                # Trigger refill if below AWD threshold OR pre-closure refill on last day of Amala
                pre_close_refill = (
                    canal_open
                    and hours_until_close <= 12.0
                    and ponded_depth_mm < 45.0
                )
                if canal_open and (ponded_depth_mm <= min_trigger or pre_close_refill) and hour_of_day in (6, 7, 8):
                    pump_active = True
                    irr_net_mm = 18.0

                ponded_depth_mm = float(
                    np.clip(
                        ponded_depth_mm + eff_precip_mm + irr_net_mm - etc_h - perc_h,
                        -80.0,
                        float(crop_cfg["paddy_management"]["max_bund_height_mm"]),
                    )
                )
                # Map ponded depth to root-zone VWC
                if ponded_depth_mm >= 0.0:
                    true_vwc = theta_sat
                    depletion_dr_mm = 0.0
                else:
                    depletion_dr_mm = abs(ponded_depth_mm) * 0.6
                    true_vwc = max(
                        theta_pwp,
                        theta_fc - depletion_dr_mm / (1000.0 * resolved.root_depth_m),
                    )
            else:
                # Upland Crop FAO-56 Dual-Kc water balance (Corn, Onion, Tomato)
                kcb_adj = adjust_kc_for_arid_climate(
                    resolved.kcb_basal, wind_2m_ms, 30.0, resolved.crop_height_m
                )
                kc_max = max(kcb_adj + 0.05, 1.20)
                etc_h, _, ke_h = compute_dual_kc_etc(
                    et0_mm=et0_h,
                    kcb_basal=kcb_adj,
                    kc_max=kc_max,
                    wetted_fraction_fw=fw,
                    topsoil_depletion_mm=topsoil_depletion_mm,
                    rew_mm=rew_mm,
                    tew_mm=tew_mm,
                    root_depletion_dr_mm=depletion_dr_mm,
                    taw_mm=taw_mm,
                    raw_mm=raw_mm,
                )

                # Pre-closure refill if canal closes within 18h and forecast will exceed RAW
                pre_close_needed = (
                    canal_open
                    and hours_until_close <= 18.0
                    and (depletion_dr_mm + 0.6 * forecast_et0_72h) >= raw_mm
                )
                if canal_open and (depletion_dr_mm >= raw_mm or pre_close_needed) and hour_of_day in (6, 7):
                    pump_active = True
                    irr_net_mm = min(22.0, max(6.0, depletion_dr_mm * 0.55))

                # Seed Scenario: Ineffective irrigation on day 18 (water fails to reach root zone)
                effective_infiltration_mm = irr_net_mm
                if inject_anomalies and day_idx == 18 and pump_active:
                    effective_infiltration_mm = 0.5  # Runoff/bypass failure
                    anomaly_label = "INEFFECTIVE_IRRIGATION"

                depletion_dr_mm = float(
                    np.clip(
                        depletion_dr_mm - eff_precip_mm - effective_infiltration_mm + etc_h,
                        0.0,
                        taw_mm,
                    )
                )
                topsoil_depletion_mm = float(
                    np.clip(
                        topsoil_depletion_mm - eff_precip_mm - effective_infiltration_mm + ke_h * et0_h,
                        0.0,
                        tew_mm,
                    )
                )
                true_vwc = float(
                    np.clip(
                        theta_fc - (depletion_dr_mm / (1000.0 * resolved.root_depth_m)),
                        theta_pwp * 0.85,
                        theta_sat,
                    )
                )

            if pump_active:
                hours_since_last_irr = 0.0
            else:
                hours_since_last_irr += 1.0

            # 4. Capacitive sensor observation model (RMSE = 0.026 m^3/m^3 per SRC-28)
            sensor_rmse = float(thresh["sensor_bounds"]["capacitive_sensor_rmse_m3m3"])
            measured_vwc = float(true_vwc + rng.normal(0.0, sensor_rmse * 0.35))

            # Seed Scenario: Disconnected capacitive sensor on day 22 (hours 10-15)
            if inject_anomalies and day_idx == 22 and (10 <= hour_of_day <= 15):
                measured_vwc = -0.05  # Implausible reading / open wire
                anomaly_label = "SENSOR_DISCONNECTED"

            # 5. Hydraulic & Pump Telemetry (BattLeDIM orifice leak & blockage physics, SRC-24)
            nom_flow = (
                float(thresh["equipment_and_alerts"]["nominal_flood_flow_lpm"])
                if is_paddy
                else float(thresh["equipment_and_alerts"]["nominal_drip_flow_lpm"])
            )
            nom_press = (
                float(thresh["equipment_and_alerts"]["nominal_flood_pressure_bar"])
                if is_paddy
                else float(thresh["equipment_and_alerts"]["nominal_drip_pressure_bar"])
            )

            if pump_active:
                flow_lpm = float(nom_flow + rng.normal(0.0, 4.5))
                pressure_bar = float(nom_press + rng.normal(0.0, 0.05))
                current_amps = float(6.8 + rng.normal(0.0, 0.18))
                vib_rms = float(1.8 + rng.normal(0.0, 0.15))

                # Seed Scenario: Pipe burst/orifice leak on day 10
                if inject_anomalies and day_idx == 10:
                    cd = float(thresh["equipment_and_alerts"]["battledim_orifice_cd"])
                    alpha = float(
                        thresh["equipment_and_alerts"]["battledim_plastic_exponent_alpha"]
                    )
                    pressure_bar = nom_press * 0.72
                    head_m = pressure_bar / 0.0980665
                    leak_flow_lpm = cd * 6.2 * (head_m ** (alpha * 0.45))
                    flow_lpm = nom_flow + leak_flow_lpm
                    current_amps = 7.9
                    anomaly_label = "PIPE_LEAKAGE"

                # Seed Scenario: Pipe/emitter blockage + pump cavitation on day 15
                elif inject_anomalies and day_idx == 15:
                    flow_lpm = nom_flow * 0.62
                    pressure_bar = nom_press * 1.28
                    current_amps = 8.4
                    vib_rms = 5.6  # High vibration RMS (SRC-23 cavitation/impeller fault proxy)
                    anomaly_label = "PIPE_BLOCKAGE_OR_CAVITATION"
            else:
                flow_lpm = 0.0
                pressure_bar = 0.05
                current_amps = 0.0
                vib_rms = 0.05

            if in_heat_wave and anomaly_label == "NORMAL":
                anomaly_label = "CLAC_HEAT_WAVE"

            records.append(
                {
                    "timestamp_utc": ts.isoformat(),
                    "field_id": f"MANSOURA_{crop_type.value}_01",
                    "crop_type": crop_type.value,
                    "days_after_planting": dap,
                    "growth_stage": resolved.stage.value,
                    "hydrological_regime": regime.value,
                    "data_provenance": DataProvenance.SYNTHETIC_FAO56.value,
                    "temp_air_2m_c": round(temp_air_c, 2),
                    "rh_2m_pct": round(rh_pct, 2),
                    "wind_speed_2m_ms": round(wind_2m_ms, 2),
                    "solar_rad_mj_m2_h": round(solar_rad_mj, 4),
                    "precip_mm": round(precip_mm, 2),
                    "et0_fao_mm": round(et0_h, 4),
                    "vwc_true_m3m3": round(true_vwc, 4),
                    "vwc_root_zone_m3m3": round(measured_vwc, 4),
                    "soil_temp_c": round(temp_air_c * 0.88 + 2.1, 2),
                    "soil_ec_dsm": round(1.05 + 0.005 * day_idx + float(rng.normal(0.0, 0.03)), 3),
                    "soil_ph": 7.4,
                    "root_depth_m": resolved.root_depth_m,
                    "depletion_root_zone_mm": round(depletion_dr_mm, 3),
                    "raw_threshold_mm": round(raw_mm, 3),
                    "taw_mm": round(taw_mm, 3),
                    "ponded_water_depth_mm": round(ponded_depth_mm, 2),
                    "canal_turn_open": canal_open,
                    "hours_until_canal_close": round(hours_until_close, 1),
                    "hours_since_last_irrigation": round(hours_since_last_irr, 1),
                    "forecast_et0_next_72h_mm": round(forecast_et0_72h, 2),
                    "forecast_max_temp_next_72h_c": round(forecast_max_t_72h, 2),
                    "pump_active": pump_active,
                    "irrigation_applied_mm": round(irr_net_mm / eta_irr, 2) if pump_active else 0.0,
                    "flow_rate_lpm": round(flow_lpm, 2),
                    "manifold_pressure_bar": round(pressure_bar, 3),
                    "pump_current_amps": round(current_amps, 2),
                    "vib_rms_mms": round(vib_rms, 3),
                    "anomaly_label": anomaly_label,
                }
            )

        return pd.DataFrame(records)