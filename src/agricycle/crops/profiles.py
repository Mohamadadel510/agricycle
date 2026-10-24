"""Crop and soil configuration loaders and dynamic phenology resolvers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import yaml

from agricycle.schema import CropType, GrowthStage, HydrologicalRegime


@dataclass(frozen=True)
class ResolvedCropStage:
    """Resolved crop parameters for a specific Day After Planting (DAP)."""

    crop_type: CropType
    dap: int
    stage: GrowthStage
    kc_single: float
    kcb_basal: float
    root_depth_m: float
    depletion_fraction_p: float
    crop_height_m: float
    salinity_threshold_ec_dsm: float
    heat_wave_threshold_c: float
    is_paddy: bool


class ConfigManager:
    """Loads and validates YAML configurations for crops, soils, and thresholds."""

    def __init__(self, config_dir: Path | str = "config") -> None:
        self.config_dir = Path(config_dir)
        self.crops_cfg = self._load_yaml("crop_profiles.yaml")["crops"]
        self.soils_cfg = self._load_yaml("soil_profiles.yaml")["soils"]
        self.thresholds_cfg = self._load_yaml("thresholds.yaml")
        self._validate_configs()

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing configuration file: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _validate_configs(self) -> None:
        for crop in CropType:
            if crop.value not in self.crops_cfg:
                raise ValueError(f"Missing required crop profile: {crop.value}")
            c_data = self.crops_cfg[crop.value]
            for stg in ("INI", "DEV", "MID", "END"):
                if c_data["stage_days"].get(stg, 0) <= 0:
                    raise ValueError(f"Invalid stage length for {crop.value}.{stg}")
            if not (0.05 <= c_data["depletion_fraction_p"] <= 0.85):
                raise ValueError(f"Invalid depletion fraction p for {crop.value}")

    def resolve_crop_stage(
        self,
        crop_type: CropType,
        days_after_planting: int,
        transplanting_window: str = "MAY_EARLY",
    ) -> ResolvedCropStage:
        """Resolve stage, interpolated Kc/Kcb, and root depth for a given DAP."""
        if days_after_planting < 1:
            raise ValueError("days_after_planting must be >= 1")

        cfg = self.crops_cfg[crop_type.value]
        if crop_type == CropType.RICE and "transplanting_profiles" in cfg:
            t_prof = cfg["transplanting_profiles"].get(
                transplanting_window, cfg["transplanting_profiles"]["MAY_EARLY"]
            )
            s_days = t_prof["stage_days"]
            kc_map = t_prof["kc_single"]
        else:
            s_days = cfg["stage_days"]
            kc_map = cfg["kc_single"]

        kcb_map = cfg["kcb_basal"]
        l_ini = int(s_days["INI"])
        l_dev = int(s_days["DEV"])
        l_mid = int(s_days["MID"])
        l_end = int(s_days["END"])
        total_days = l_ini + l_dev + l_mid + l_end

        dap_clamped = min(days_after_planting, total_days)

        # Determine growth stage and linear interpolation during DEV and END (FAO-56)
        if dap_clamped <= l_ini:
            stage = GrowthStage.INI
            kc = float(kc_map["INI"])
            kcb = float(kcb_map["INI"])
        elif dap_clamped <= l_ini + l_dev:
            stage = GrowthStage.DEV
            frac = (dap_clamped - l_ini) / float(l_dev)
            kc = float(kc_map["INI"] + frac * (kc_map["MID"] - kc_map["INI"]))
            kcb = float(kcb_map["INI"] + frac * (kcb_map["MID"] - kcb_map["INI"]))
        elif dap_clamped <= l_ini + l_dev + l_mid:
            stage = GrowthStage.MID
            kc = float(kc_map["MID"])
            kcb = float(kcb_map["MID"])
        else:
            stage = GrowthStage.END
            frac = (dap_clamped - (l_ini + l_dev + l_mid)) / float(l_end)
            kc = float(kc_map["MID"] + frac * (kc_map["END"] - kc_map["MID"]))
            kcb = float(kcb_map["MID"] + frac * (kcb_map["END"] - kcb_map["MID"]))

        # Root depth grows linearly from min to max across INI + DEV stages
        r_min = float(cfg["root_depth_m"]["min"])
        r_max = float(cfg["root_depth_m"]["max"])
        if dap_clamped <= (l_ini + l_dev):
            r_frac = dap_clamped / float(l_ini + l_dev)
            root_depth = r_min + r_frac * (r_max - r_min)
        else:
            root_depth = r_max

        h_max = float(cfg["crop_height_max_m"])
        crop_height = max(0.10, h_max * min(1.0, dap_clamped / float(l_ini + l_dev)))

        return ResolvedCropStage(
            crop_type=crop_type,
            dap=days_after_planting,
            stage=stage,
            kc_single=round(kc, 4),
            kcb_basal=round(kcb, 4),
            root_depth_m=round(root_depth, 3),
            depletion_fraction_p=float(cfg["depletion_fraction_p"]),
            crop_height_m=round(crop_height, 2),
            salinity_threshold_ec_dsm=float(cfg["salinity_threshold_ec_dsm"]),
            heat_wave_threshold_c=float(cfg["heat_wave_threshold_c"][stage.value]),
            is_paddy=bool(cfg["is_paddy"]),
        )

    def get_soil_profile(self, soil_key: str = "NILE_DELTA_CLAY") -> Dict[str, Any]:
        if soil_key not in self.soils_cfg:
            raise KeyError(f"Unknown soil profile: {soil_key}")
        return self.soils_cfg[soil_key]

    def get_canal_status(
        self,
        crop_type: CropType,
        regime: HydrologicalRegime,
        day_of_season: int,
        hour_of_day: int = 12,
    ) -> Tuple[bool, float]:
        """Return (canal_turn_open, hours_until_canal_close) under MWRI rotation rules."""
        if regime == HydrologicalRegime.NEW_LANDS_PRESSURIZED:
            return True, 168.0  # Groundwater / pressurized manifold available continuously

        if crop_type == CropType.RICE:
            r_cfg = self.crops_cfg["RICE"]["paddy_management"]["mwri_canal_rotation"]
            on_days = int(r_cfg["amala_open_days"])      # 4 days ON
            off_days = int(r_cfg["batala_closed_days"])  # 6 days OFF
        else:
            s_cfg = self.soils_cfg["NILE_DELTA_CLAY"]["default_canal_rotation"]
            on_days = int(s_cfg["general_amala_days"])   # 5 days ON
            off_days = int(s_cfg["general_batala_days"]) # 10 days OFF

        cycle_days = on_days + off_days
        day_in_cycle = (day_of_season - 1) % cycle_days
        is_open = day_in_cycle < on_days

        if is_open:
            remaining_days = on_days - day_in_cycle - 1
            hours_left = float(remaining_days * 24 + (24 - hour_of_day))
        else:
            hours_left = 0.0
        return is_open, hours_left