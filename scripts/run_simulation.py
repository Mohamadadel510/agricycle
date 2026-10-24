"""Generate deterministic synthetic datasets for all four target crops."""

from __future__ import annotations

from pathlib import Path
from agricycle.crops.profiles import ConfigManager
from agricycle.data.preprocessing import clean_and_validate_dataframe
from agricycle.schema import CropType, DataProvenance
from agricycle.simulation.simulator import AgriCycleSimulator


def main() -> None:
    cfg = ConfigManager("config")
    sim = AgriCycleSimulator(config_manager=cfg, seed=42)
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("AgriCycle Stage 2 — Deterministic Multi-Crop Simulator (Provenance: SYNTHETIC_FAO56)")
    print("=" * 78)

    for crop in CropType:
        soil_key = (
            "NILE_DELTA_CLAY"
            if crop in (CropType.RICE, CropType.CORN)
            else "NEW_LANDS_SANDY_LOAM"
        )
        raw_df = sim.simulate_season(
            crop_type=crop,
            soil_key=soil_key,
            days=30,
            start_dap=25,
            inject_anomalies=True,
        )
        clean_df, audit = clean_and_validate_dataframe(
            df=raw_df,
            sensor_bounds=cfg.thresholds_cfg["sensor_bounds"],
            provenance_tag=DataProvenance.SYNTHETIC_FAO56.value,
        )

        file_path = out_dir / f"sim_{crop.value.lower()}_30d.csv"
        clean_df.to_csv(file_path, index=False)

        irr_events = int(clean_df["pump_active"].sum())
        anomalies = int((clean_df["anomaly_label"] != "NORMAL").sum())
        flagged = int(audit["out_of_bounds_flagged"])
        print(
            f"[{crop.value:<6}] Soil: {soil_key:<20} | Rows: {len(clean_df):4d} | "
            f"Irr Hours: {irr_events:2d} | Anomalies: {anomalies:3d} | "
            f"Sensor Outliers Flagged: {flagged:2d} -> Saved: {file_path}"
        )


if __name__ == "__main__":
    main()