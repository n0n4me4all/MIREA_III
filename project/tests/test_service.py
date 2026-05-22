from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.data.synthetic import generate_synthetic_hourly_demand
from src.utils.config import load_config, project_root


@pytest.fixture(scope="module")
def client_with_model(tmp_path_factory: pytest.TempPathFactory) -> tuple[TestClient, Path]:
    root = project_root()
    tmp = tmp_path_factory.mktemp("svc")

    proc = tmp / "hourly_demand.parquet"
    generate_synthetic_hourly_demand(seed=99).to_parquet(proc, index=False)

    import yaml

    cfg = load_config(root / "configs" / "config.yaml")
    cfg["data"]["processed_path"] = str(proc)
    cfg["model"]["artifact_path"] = str(tmp / "model.joblib")
    cfg["model"]["feature_schema_path"] = str(tmp / "feature_schema.json")
    cfg["model"]["metrics_path"] = str(tmp / "metrics.json")
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.dump({k: v for k, v in cfg.items() if k != "_paths"}), encoding="utf-8")

    from src.models.train import _build_sample_request, train_pipeline

    loaded_cfg = load_config(cfg_path)
    train_pipeline(loaded_cfg)
    _build_sample_request(loaded_cfg)
    sample_path = Path(loaded_cfg["model"]["artifact_path"]).parent / "sample_request.json"

    import os

    os.environ["CONFIG_PATH"] = str(cfg_path)

    from src.service import app as app_module
    from src.service.model_loader import store

    store.load(str(cfg_path))
    return TestClient(app_module.app), sample_path


def test_health(client_with_model: tuple[TestClient, Path]) -> None:
    client, _ = client_with_model
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_sample(client_with_model: tuple[TestClient, Path]) -> None:
    client, sample_path = client_with_model
    if not sample_path.exists():
        pytest.skip("sample_request.json ещё не создан")

    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == payload["horizon"]


def test_predict_short_history(client_with_model: tuple[TestClient, Path]) -> None:
    client, _ = client_with_model
    payload = {
        "history": [{"timestamp": "2024-08-01T00:00:00-04:00", "trip_count": 100}],
        "future_weather": [
            {
                "timestamp": "2024-08-02T00:00:00-04:00",
                "temperature_2m": 20.0,
                "precipitation": 0.0,
                "wind_speed_10m": 5.0,
                "relative_humidity_2m": 60.0,
                "weather_code": 1,
            }
        ],
        "horizon": 1,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
