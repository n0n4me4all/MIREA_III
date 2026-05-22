from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.synthetic import generate_synthetic_hourly_demand
from src.utils.config import load_config, project_root, resolve_path


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Обучает модель на синтетике во временной директории."""
    root = project_root()
    cfg = load_config(root / "configs" / "config.yaml")
    tmp = tmp_path_factory.mktemp("artifacts")

    proc = tmp / "hourly_demand.parquet"
    df = generate_synthetic_hourly_demand(seed=42)
    df.to_parquet(proc, index=False)

    cfg["data"]["processed_path"] = str(proc.relative_to(root)) if proc.is_relative_to(root) else str(proc)
    cfg["model"]["artifact_path"] = str((tmp / "model.joblib"))
    cfg["model"]["feature_schema_path"] = str((tmp / "feature_schema.json"))
    cfg["model"]["metrics_path"] = str((tmp / "metrics.json"))

    # Пишем временный конфиг
    import yaml

    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.dump({k: v for k, v in cfg.items() if k != "_paths"}), encoding="utf-8")

    from src.models.train import train_pipeline

    train_pipeline(load_config(cfg_path))
    return tmp


def test_metrics_file_exists(trained_artifacts: Path) -> None:
    metrics = trained_artifacts / "metrics.json"
    assert metrics.exists()
    data = json.loads(metrics.read_text(encoding="utf-8"))
    assert "models" in data
    assert "final_model" in data
