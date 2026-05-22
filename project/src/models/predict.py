from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import build_rows_for_forecast, load_feature_schema


def load_model_bundle(artifact_path: Path) -> dict[str, Any]:
    """Загружает joblib-пакет с моделью и списком признаков."""
    return joblib.load(artifact_path)


def predict_hourly(
    history: pd.DataFrame,
    future_weather: pd.DataFrame,
    artifact_path: Path,
    feature_schema_path: Path,
    horizon: int = 24,
) -> np.ndarray:
    """
    Возвращает прогноз trip_count на horizon часов вперёд.

    Использует рекурсивный прогноз: каждый следующий час опирается на уже
    предсказанные значения в lag_1h и rolling-признаках.
    """
    bundle = load_model_bundle(artifact_path)
    model = bundle["model"]
    feature_cols = bundle.get("feature_columns") or load_feature_schema(feature_schema_path)

    hist = history.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    weather = future_weather.copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather = weather.head(horizon)

    preds: list[float] = []
    for i in range(horizon):
        step_weather = weather.iloc[i : i + 1]
        rows = build_rows_for_forecast(hist, step_weather, horizon=1)
        if rows.empty or rows[feature_cols].isna().any(axis=None):
            raise ValueError("Недостаточно признаков для прогноза (проверьте длину history)")

        y_hat = float(model.predict(rows[feature_cols])[0])
        y_hat = max(0.0, y_hat)
        preds.append(y_hat)

        new_row = step_weather.copy()
        new_row["trip_count"] = y_hat
        hist = pd.concat([hist, new_row], ignore_index=True)

    return np.array(preds)
