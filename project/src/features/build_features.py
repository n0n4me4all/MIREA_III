from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import holidays
import numpy as np
import pandas as pd

CALENDAR_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_holiday",
    "month",
]

WEATHER_FEATURE_COLUMNS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
    "weather_code",
]

LAG_FEATURE_COLUMNS = [
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_mean_168h",
]

FEATURE_COLUMNS = CALENDAR_FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS + LAG_FEATURE_COLUMNS


def get_feature_columns() -> list[str]:
    """Список признаков для обучения и инференса."""
    return list(FEATURE_COLUMNS)


def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Добавляет календарные признаки и циклическое кодирование."""
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    us_holidays = holidays.country_holidays("US")
    out["is_holiday"] = ts.dt.date.map(lambda d: 1 if d in us_holidays else 0).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


def add_lag_features(
    df: pd.DataFrame,
    target: str = "trip_count",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Добавляет лаги и скользящие средние без утечки из будущего."""
    out = df.sort_values(timestamp_col).copy()
    y = out[target]
    out["lag_1h"] = y.shift(1)
    out["lag_24h"] = y.shift(24)
    out["lag_168h"] = y.shift(168)
    out["rolling_mean_24h"] = y.shift(1).rolling(24, min_periods=1).mean()
    out["rolling_mean_168h"] = y.shift(1).rolling(168, min_periods=1).mean()
    return out


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Полный пайплайн признаков для обучения."""
    out = add_calendar_features(df)
    out = add_lag_features(out)
    return out


def save_feature_schema(path: Path) -> None:
    """Сохраняет список признаков в JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"feature_columns": get_feature_columns()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_feature_schema(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["feature_columns"])


def build_rows_for_forecast(
    history: pd.DataFrame,
    future_weather: pd.DataFrame,
    horizon: int = 24,
) -> pd.DataFrame:
    """
    Строит матрицу признаков для прогноза на horizon часов.

    history: колонки timestamp, trip_count (+ опционально погода для прошлого)
    future_weather: timestamp + погодные поля на будущие часы
    """
    hist = history.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    fut = future_weather.copy()
    fut["timestamp"] = pd.to_datetime(fut["timestamp"])

    combined = pd.concat(
        [
            hist[["timestamp", "trip_count"]],
            fut.assign(trip_count=np.nan)[["timestamp", "trip_count"]],
        ],
        ignore_index=True,
    )

    if "temperature_2m" in hist.columns:
        weather_hist = hist[
            ["timestamp", "temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m", "weather_code"]
        ]
    else:
        weather_hist = pd.DataFrame()

    weather_cols = ["timestamp", "temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m", "weather_code"]
    weather_fut = fut[[c for c in weather_cols if c in fut.columns]]
    if not weather_hist.empty:
        weather = pd.concat([weather_hist, weather_fut], ignore_index=True)
    else:
        weather = weather_fut

    combined = combined.merge(weather, on="timestamp", how="left")
    featured = build_feature_matrix(combined)
    future_mask = featured["trip_count"].isna()
    return featured.loc[future_mask].head(horizon)
