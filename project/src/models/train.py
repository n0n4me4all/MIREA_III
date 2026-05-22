from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features.build_features import (
    build_feature_matrix,
    get_feature_columns,
    save_feature_schema,
)
from src.models.baselines import predict_seasonal_naive
from src.models.evaluate import MetricsResult, regression_metrics, time_split
from src.utils.config import load_config, resolve_path
from src.utils.logging import setup_logging

logger = setup_logging()

SEED = 42


def _create_model(backend: str) -> Any:
    if backend == "lightgbm":
        try:
            import lightgbm as lgb

            return lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                random_state=SEED,
                verbose=-1,
            )
        except ImportError:
            logger.warning("lightgbm недоступен, используем HistGradientBoosting")
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=8,
        random_state=SEED,
    )


def _evaluate_split(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    m = regression_metrics(y_true, y_pred)
    logger.info("%s: MAE=%.2f RMSE=%.2f sMAPE=%.4f", name, m.mae, m.rmse, m.smape)
    return {"mae": m.mae, "rmse": m.rmse, "smape": m.smape}


def train_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    processed_path = resolve_path(cfg, "processed_path")
    df = pd.read_parquet(processed_path)
    featured = build_feature_matrix(df)

    train_df, val_df, test_df = time_split(
        featured,
        test_days=cfg["model"]["test_days"],
        val_days=cfg["model"]["val_days"],
    )

    feature_cols = get_feature_columns()
    train_fit = pd.concat([train_df, val_df], ignore_index=True)
    train_fit = train_fit.dropna(subset=feature_cols + ["trip_count"])

    X_train = train_fit[feature_cols]
    y_train = train_fit["trip_count"]
    if len(X_train) == 0:
        all_na = featured[feature_cols].isna().all()
        bad = [c for c in feature_cols if all_na.get(c, False)]
        raise ValueError(
            "Нет строк для обучения после dropna. "
            f"Полностью пустые признаки: {bad}. "
            "Проверьте период погоды: python -m src.data.build_dataset --weather-only"
        )

    model = _create_model(cfg["model"].get("backend", "sklearn"))
    model.fit(X_train, y_train)

    results: dict[str, Any] = {"models": {}}

    for lag, label in [(24, "seasonal_naive_24h"), (168, "seasonal_naive_168h")]:
        preds = predict_seasonal_naive(test_df, lag_hours=lag)
        results["models"][label] = _evaluate_split(
            label, test_df["trip_count"].to_numpy(), preds
        )

    test_clean = test_df.dropna(subset=feature_cols)
    ml_preds = model.predict(test_clean[feature_cols])
    model_name = cfg["model"].get("backend", "sklearn") + "_boosting"
    results["models"][model_name] = _evaluate_split(
        model_name,
        test_clean["trip_count"].to_numpy(),
        ml_preds,
    )

    artifact_path = resolve_path(cfg, "artifact_path")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "model_name": model_name,
        },
        artifact_path,
    )

    schema_path = resolve_path(cfg, "feature_schema_path")
    save_feature_schema(schema_path)

    test_clean = test_clean.copy()
    test_clean["y_pred"] = ml_preds
    pred_path = artifact_path.parent / "predictions_test.csv"
    test_clean[["timestamp", "trip_count", "y_pred"]].to_csv(pred_path, index=False)

    winner = min(
        results["models"].items(),
        key=lambda x: x[1]["mae"],
    )
    results["final_model"] = winner[0]
    results["test_period"] = {
        "start": str(test_df["timestamp"].min()),
        "end": str(test_df["timestamp"].max()),
    }

    metrics_path = resolve_path(cfg, "metrics_path")
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Финальная модель по MAE: %s", results["final_model"])
    return results


def _build_sample_request(cfg: dict[str, Any]) -> None:
    """Сохраняет sample_request.json для демо API."""
    df = pd.read_parquet(resolve_path(cfg, "processed_path"))
    df = df.sort_values("timestamp")
    horizon = cfg["model"]["horizon"]
    min_hist = cfg["model"]["min_history_hours"]

    tail = df.tail(min_hist + horizon)
    history = tail.head(min_hist)
    future = tail.tail(horizon)

    payload = {
        "history": [
            {
                "timestamp": row["timestamp"].isoformat(),
                "trip_count": int(row["trip_count"]),
            }
            for _, row in history.iterrows()
        ],
        "future_weather": [
            {
                "timestamp": row["timestamp"].isoformat(),
                "temperature_2m": float(row.get("temperature_2m", 20.0)),
                "precipitation": float(row.get("precipitation", 0.0)),
                "wind_speed_10m": float(row.get("wind_speed_10m", 10.0)),
                "relative_humidity_2m": float(row.get("relative_humidity_2m", 60.0)),
                "weather_code": int(row.get("weather_code", 1)),
            }
            for _, row in future.iterrows()
        ],
        "horizon": horizon,
    }
    out = resolve_path(cfg, "artifact_path").parent / "sample_request.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Sample request: %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение моделей прогноза спроса")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_pipeline(cfg)
    _build_sample_request(cfg)


if __name__ == "__main__":
    main()
