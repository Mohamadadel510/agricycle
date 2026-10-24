"""Transparent rule-based irrigation baseline using FAO-56 Dual-Kc and MWRI canal rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

from agricycle.crops.evapotrans import adjust_kc_for_arid_climate, compute_dual_kc_etc
from agricycle.crops.profiles import ConfigManager
from agricycle.schema import CropType, HydrologicalRegime


@dataclass(frozen=True)
class IrrigationDecision:
    """Structured output from the Module A Rule-Based Baseline Scheduler."""

    timestamp_utc: str
    crop_type: str
    irrigate_now: bool
    irrigate_within_6h: int
    hours_until_next_irrigation: float
    expected_deficit_mm: float
    recommended_net_mm: float
    recommended_volume_m3: float
    recommended_volume_liters: float
    confidence_score: float
    rule_triggered: str
    sensor_fallback_used: bool
    data_provenance: str


class RuleBasedIrrigationBaseline:
    """Transparent FAO-56 Dual-Kc + MWRI Canal Rotation + CLAC Heat-Wave Baseline.

    Operates strictly on observed sensor telemetry and forecast inputs without peeking
    at ground-truth simulation states (vwc_true_m3m3 or depletion_root_zone_mm).
    """

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        field_area_m2: float = 4200.0,  # 1 Egyptian Feddan = 4,200 m^2
    ) -> None:
        self.cfg = config_manager or ConfigManager("config")
        self.field_area_m2 = float(field_area_m2)

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        soil_key: str = "NILE_DELTA_CLAY",
    ) -> pd.DataFrame:
        """Run sequential rule-based scheduling across a time-series DataFrame."""
        soil = self.cfg.get_soil_profile(soil_key)
        theta_fc = float(soil["theta_fc_m3m3"])
        theta_pwp = float(soil["theta_pwp_m3m3"])
        rew_mm = float(soil["rew_mm"])
        tew_mm = float(soil["tew_mm"])
        bounds = self.cfg.thresholds_cfg["sensor_bounds"]

        decisions: List[IrrigationDecision] = []
        last_valid_depletion_mm = 10.0

        for _, row in df.iterrows():
            crop_type = CropType(str(row["crop_type"]))
            crop_cfg = self.cfg.crops_cfg[crop_type.value]
            is_paddy = bool(crop_cfg["is_paddy"])
            irr_method = crop_cfg["default_irrigation_method"]
            fw = float(crop_cfg["wetted_fraction_fw"][irr_method])
            eta_irr = float(crop_cfg["irrigation_efficiency"][irr_method])

            dap = int(row["days_after_planting"])
            resolved = self.cfg.resolve_crop_stage(crop_type, dap)

            et0_h = float(row.get("et0_fao_mm", 0.25))
            if np.isnan(et0_h) or et0_h < 0.0:
                et0_h = 0.25

            wind_2m = float(row.get("wind_speed_2m_ms", 2.0))
            forecast_et0_72h = float(row.get("forecast_et0_next_72h_mm", 15.0))
            forecast_max_t_72h = float(row.get("forecast_max_temp_next_72h_c", 32.0))
            canal_open = bool(row.get("canal_turn_open", True))
            hours_until_close = float(row.get("hours_until_canal_close", 72.0))
            provenance = str(row.get("data_provenance", "UNKNOWN"))
            ts_str = str(row["timestamp_utc"])

            # Check sensor validity
            vwc_meas = float(row.get("vwc_root_zone_m3m3", np.nan))
            quality_flag = str(row.get("sensor_quality_flag", "VALID"))
            sensor_invalid = (
                np.isnan(vwc_meas)
                or quality_flag == "OUT_OF_BOUNDS"
                or vwc_meas < float(bounds["vwc_min_m3m3"])
                or vwc_meas > float(bounds["vwc_max_m3m3"])
            )

            if is_paddy:
                # Paddy Rice water management (standing water depth + MWRI 4d ON / 6d OFF)
                paddy_cfg = crop_cfg["paddy_management"]
                ponded_mm = float(row.get("ponded_water_depth_mm", 20.0))
                target_depth_mm = float(paddy_cfg["target_ponded_depth_mm"])
                pre_close_target_mm = float(paddy_cfg["pre_closure_refill_depth_mm"])
                min_trigger_mm = (
                    float(paddy_cfg["flowering_min_depth_mm"])
                    if resolved.stage.value == "MID"
                    else float(paddy_cfg["awd_trigger_depth_mm"])
                )

                kc_adj = adjust_kc_for_arid_climate(
                    resolved.kc_single, wind_2m, 30.0, resolved.crop_height_m
                )
                hourly_loss_mm = max(
                    0.12,
                    et0_h * kc_adj
                    + float(paddy_cfg["deep_percolation_mm_day"]) / 24.0
                    + (forecast_et0_72h / 72.0) * 0.35,
                )

                deficit_mm = max(0.0, target_depth_mm - ponded_mm)
                margin_to_trigger_mm = ponded_mm - min_trigger_mm
                est_hours_to_trigger = max(0.0, margin_to_trigger_mm / hourly_loss_mm)

                pre_close_needed = (
                    canal_open
                    and hours_until_close <= 12.0
                    and ponded_mm < 45.0
                )

                if not canal_open:
                    irr_now = False
                    irr_6h = 0
                    # Estimate hours until canal reopens (MWRI 6 days Batala max)
                    hours_until_irr = max(12.0, est_hours_to_trigger)
                    net_refill_mm = 0.0
                    rule_name = "CANAL_CLOSED_BATALA_LOCKOUT"
                    conf = 0.90
                elif ponded_mm <= min_trigger_mm:
                    irr_now = True
                    irr_6h = 1
                    hours_until_irr = 0.0
                    net_refill_mm = max(10.0, target_depth_mm - ponded_mm)
                    rule_name = "PADDY_AWD_OR_FLOWERING_THRESHOLD"
                    conf = 0.92
                elif pre_close_needed:
                    irr_now = True
                    irr_6h = 1
                    hours_until_irr = 0.0
                    net_refill_mm = max(12.0, pre_close_target_mm - ponded_mm)
                    rule_name = "MWRI_PRE_CLOSURE_PADDY_REFILL"
                    conf = 0.88
                elif est_hours_to_trigger <= 6.0:
                    irr_now = False
                    irr_6h = 1
                    hours_until_irr = round(est_hours_to_trigger, 2)
                    net_refill_mm = max(10.0, target_depth_mm - min_trigger_mm)
                    rule_name = "PADDY_6H_LOOKAHEAD_TRIGGER"
                    conf = 0.85
                else:
                    irr_now = False
                    irr_6h = 0
                    hours_until_irr = min(168.0, round(est_hours_to_trigger, 2))
                    net_refill_mm = 0.0
                    rule_name = "PADDY_DEPTH_ADEQUATE"
                    conf = 0.90

                gross_mm = net_refill_mm / eta_irr if net_refill_mm > 0.0 else 0.0
                vol_m3 = (gross_mm * self.field_area_m2) / 1000.0

                decisions.append(
                    IrrigationDecision(
                        timestamp_utc=ts_str,
                        crop_type=crop_type.value,
                        irrigate_now=irr_now,
                        irrigate_within_6h=irr_6h,
                        hours_until_next_irrigation=hours_until_irr,
                        expected_deficit_mm=round(deficit_mm, 2),
                        recommended_net_mm=round(net_refill_mm, 2),
                        recommended_volume_m3=round(vol_m3, 3),
                        recommended_volume_liters=round(vol_m3 * 1000.0, 1),
                        confidence_score=conf,
                        rule_triggered=rule_name,
                        sensor_fallback_used=sensor_invalid,
                        data_provenance=provenance,
                    )
                )
                continue

            # Upland Crop FAO-56 Dual-Kc Water Balance (Corn, Onion, Tomato)
            taw_mm = 1000.0 * (theta_fc - theta_pwp) * resolved.root_depth_m
            raw_mm = resolved.depletion_fraction_p * taw_mm

            kcb_adj = adjust_kc_for_arid_climate(
                resolved.kcb_basal, wind_2m, 30.0, resolved.crop_height_m
            )
            kc_max = max(kcb_adj + 0.05, 1.20)
            etc_h, _, _ = compute_dual_kc_etc(
                et0_mm=et0_h,
                kcb_basal=kcb_adj,
                kc_max=kc_max,
                wetted_fraction_fw=fw,
                topsoil_depletion_mm=0.35 * tew_mm,
                rew_mm=rew_mm,
                tew_mm=tew_mm,
                root_depletion_dr_mm=last_valid_depletion_mm,
                taw_mm=taw_mm,
                raw_mm=raw_mm,
            )

            if sensor_invalid:
                # Safe FAO-56 water-balance fallback when capacitive probe is disconnected
                est_dr_mm = min(taw_mm, last_valid_depletion_mm + etc_h)
                conf_base = 0.55
            else:
                est_dr_mm = float(
                    np.clip(
                        1000.0 * (theta_fc - vwc_meas) * resolved.root_depth_m,
                        0.0,
                        taw_mm,
                    )
                )
                last_valid_depletion_mm = est_dr_mm
                conf_base = 0.90

            # Blended hourly depletion rate (current Dual-Kc ETc + 72h forecast trajectory)
            forecast_hourly_etc = (forecast_et0_72h / 72.0) * (kcb_adj * 0.85 + fw * 0.15)
            hourly_depletion_rate = max(0.08, 0.5 * etc_h + 0.5 * forecast_hourly_etc)

            remaining_raw_mm = max(0.0, raw_mm - est_dr_mm)
            est_hours_to_raw = remaining_raw_mm / hourly_depletion_rate

            # Check special Egyptian agronomic rules (MWRI canal closure & CLAC heat wave)
            pre_close_needed = (
                canal_open
                and hours_until_close <= 18.0
                and (est_dr_mm + 0.6 * forecast_et0_72h) >= raw_mm
            )
            clac_heat_wave_risk = (
                canal_open
                and forecast_max_t_72h >= resolved.heat_wave_threshold_c
                and est_dr_mm >= (0.75 * raw_mm)
            )

            if not canal_open:
                irr_now = False
                irr_6h = 0
                hours_until_irr = max(12.0, round(est_hours_to_raw, 2))
                net_refill_mm = 0.0
                rule_name = "CANAL_CLOSED_BATALA_LOCKOUT"
            elif sensor_invalid and est_dr_mm < raw_mm:
                irr_now = False
                irr_6h = 0
                hours_until_irr = min(168.0, round(est_hours_to_raw, 2))
                net_refill_mm = 0.0
                rule_name = "SENSOR_FALLBACK_CONSERVATIVE_HOLD"
            elif est_dr_mm >= raw_mm:
                irr_now = True
                irr_6h = 1
                hours_until_irr = 0.0
                net_refill_mm = min(22.0, max(6.0, est_dr_mm * 0.55))
                rule_name = "FAO56_RAW_THRESHOLD_EXCEEDED"
            elif pre_close_needed:
                irr_now = True
                irr_6h = 1
                hours_until_irr = 0.0
                net_refill_mm = min(22.0, max(6.0, est_dr_mm * 0.55))
                rule_name = "MWRI_PRE_CLOSURE_CANAL_REFILL"
            elif est_hours_to_raw <= 6.0:
                irr_now = False
                irr_6h = 1
                hours_until_irr = round(est_hours_to_raw, 2)
                net_refill_mm = min(22.0, max(6.0, raw_mm * 0.55))
                rule_name = "FAO56_6H_LOOKAHEAD_DEPLETION"
            elif clac_heat_wave_risk and est_hours_to_raw <= 12.0:
                irr_now = False
                irr_6h = 1
                hours_until_irr = round(min(6.0, est_hours_to_raw), 2)
                net_refill_mm = min(20.0, max(6.0, est_dr_mm * 0.55))
                rule_name = "CLAC_HEAT_WAVE_PRE_IRRIGATION"
            else:
                irr_now = False
                irr_6h = 0
                hours_until_irr = min(168.0, round(est_hours_to_raw, 2))
                net_refill_mm = 0.0
                rule_name = "ROOT_ZONE_MOISTURE_ADEQUATE"

            gross_mm = net_refill_mm / eta_irr if net_refill_mm > 0.0 else 0.0
            vol_m3 = (gross_mm * self.field_area_m2) / 1000.0

            decisions.append(
                IrrigationDecision(
                    timestamp_utc=ts_str,
                    crop_type=crop_type.value,
                    irrigate_now=irr_now,
                    irrigate_within_6h=irr_6h,
                    hours_until_next_irrigation=hours_until_irr,
                    expected_deficit_mm=round(est_dr_mm, 2),
                    recommended_net_mm=round(net_refill_mm, 2),
                    recommended_volume_m3=round(vol_m3, 3),
                    recommended_volume_liters=round(vol_m3 * 1000.0, 1),
                    confidence_score=round(conf_base, 2),
                    rule_triggered=rule_name,
                    sensor_fallback_used=sensor_invalid,
                    data_provenance=provenance,
                )
            )

        return pd.DataFrame([d.__dict__ for d in decisions])