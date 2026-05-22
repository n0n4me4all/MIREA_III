"""Ablation: partial correlation погоды и сравнение наборов признаков."""
from __future__ import annotations

import argparse

import pandas as pd

from src.analysis.ablation import run_ablation, save_ablation
from src.features.build_features import build_feature_matrix
from src.utils.config import load_config, resolve_path
from src.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation и partial correlation")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_parquet(resolve_path(cfg, "processed_path"))
    featured = build_feature_matrix(df)

    results = run_ablation(
        featured,
        test_days=cfg["model"]["test_days"],
        val_days=cfg["model"]["val_days"],
    )
    results["test_period"] = {
        "test_days": cfg["model"]["test_days"],
        "val_days": cfg["model"]["val_days"],
    }

    artifacts_dir = resolve_path(cfg, "artifact_path").parent
    save_ablation(results, artifacts_dir)

    print("\n=== Частная корреляция (погода | календарь) ===")
    for row in results["partial_correlations"]:
        print(
            f"  {row['feature']:22s}  raw={row['pearson_raw']:+.3f}  "
            f"partial={row['partial_vs_calendar']:+.3f}"
        )

    print("\n=== MAE на test (меньше — лучше) ===")
    for name, m in sorted(results["models"].items(), key=lambda x: x[1]["mae"]):
        print(f"  {name:28s}  MAE={m['mae']:.1f}")


if __name__ == "__main__":
    main()
