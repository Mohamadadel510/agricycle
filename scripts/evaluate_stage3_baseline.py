"""Run and evaluate the Stage 3 FAO-56 Rule-Based Baseline on all four crops."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from agricycle.crops.profiles import ConfigManager
from agricycle.evaluation.metrics import (
    build_irrigation_labels,
    chronological_train_val_test_split,
    evaluate_irrigation_predictions,
)
from agricycle.irrigation.baseline import RuleBasedIrrigationBaseline
from agricycle.schema import CropType


def main() -> None:
    cfg = ConfigManager("config")
    baseline = RuleBasedIrrigationBaseline(config_manager=cfg, field_area_m2=4200.0)
    syn_dir = Path("data/synthetic")
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("AgriCycle Stage 3 — Rule-Based Irrigation Baseline Evaluation (Chronological Test Split)")
    print("NOTE: Evaluated on noisy capacitive sensor observations (SYNTHETIC_FAO56 provenance)")
    print("=" * 92)
    header = (
        f"{'CROP':<8} | {'PROV':<16} | {'PREC':>6} | {'REC':>6} | {'F1':>6} | "
        f"{'PR-AUC':>6} | {'TIME MAE(h)':>11} | {'VOL MAE(m³)':>11} | {'FALLBACKS':>9}"
    )
    print(header)
    print("-" * 92)

    all_metrics = []

    for crop in CropType:
        soil_key = (
            "NILE_DELTA_CLAY"
            if crop in (CropType.RICE, CropType.CORN)
            else "NEW_LANDS_SANDY_LOAM"
        )
        csv_path = syn_dir / f"sim_{crop.value.lower()}_30d.csv"
        df = pd.read_csv(csv_path)

        # Build documented agronomic ground-truth targets and save labeled dataset
        labeled_df = build_irrigation_labels(df, field_area_m2=4200.0, horizon_hours=6)
        labeled_path = proc_dir / f"labeled_{crop.value.lower()}_30d.csv"
        labeled_df.to_csv(labeled_path, index=False)

        # Chronological 60/20/20 split -> evaluate on unseen 20% Test horizon (last 6 days)
        _, _, test_df = chronological_train_val_test_split(labeled_df, 0.60, 0.20)
        preds_df = baseline.predict_dataframe(test_df, soil_key=soil_key)

        metrics = evaluate_irrigation_predictions(
            y_true_class=test_df["target_irrigate_within_6h"].to_numpy(),
            y_pred_class=preds_df["irrigate_within_6h"].to_numpy(),
            y_prob_class=preds_df["irrigate_within_6h"].to_numpy(),
            y_true_hours=test_df["target_hours_until_irrigation"].to_numpy(),
            y_pred_hours=preds_df["hours_until_next_irrigation"].to_numpy(),
            y_true_vol_m3=test_df["target_recommended_volume_m3"].to_numpy(),
            y_pred_vol_m3=preds_df["recommended_volume_m3"].to_numpy(),
            model_name="FAO56_Rule_Baseline",
            crop_type=crop.value,
            data_provenance=str(test_df["data_provenance"].iloc[0]),
        )
        fallbacks_used = int(preds_df["sensor_fallback_used"].sum())
        metrics["sensor_fallbacks_in_test"] = fallbacks_used
        all_metrics.append(metrics)

        print(
            f"{crop.value:<8} | {metrics['data_provenance']:<16} | "
            f"{metrics['precision']:6.3f} | {metrics['recall']:6.3f} | "
            f"{metrics['f1_score']:6.3f} | {metrics['pr_auc']:6.3f} | "
            f"{metrics['hours_until_irr_mae_h']:11.2f} | "
            f"{metrics['volume_mae_m3']:11.3f} | {fallbacks_used:9d}"
        )

    out_json = proc_dir / "stage3_baseline_metrics.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2)
    print("-" * 92)
    print(f"Saved labeled datasets and baseline evaluation metrics to: {out_json}")


if __name__ == "__main__":
    main()