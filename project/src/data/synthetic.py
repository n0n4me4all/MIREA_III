from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_synthetic_hourly_demand(
    start: str = "2024-06-01",
    end: str = "2024-08-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Генерирует синтетический почасовой ряд спроса с сезонностью.

    Используется для тестов и демо, если реальные Citi Bike данные недоступны.
    """
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start, tz="America/New_York")
    end_ts = pd.Timestamp(end, tz="America/New_York") + pd.Timedelta(hours=23)
    index = pd.date_range(start_ts, end_ts, freq="h", tz="America/New_York")

    hours = index.hour
    dow = index.dayofweek
    base = 200 + 80 * np.sin(2 * np.pi * hours / 24) + 40 * (dow < 5).astype(float)
    weekend_factor = np.where(dow >= 5, 1.15, 1.0)
    noise = rng.normal(0, 25, size=len(index))
    trip_count = np.maximum(0, (base * weekend_factor + noise).astype(int))

    temp = 18 + 10 * np.sin(2 * np.pi * (index.dayofyear) / 365) + rng.normal(0, 2, len(index))
    precip = np.where(rng.random(len(index)) < 0.12, rng.exponential(1.5, len(index)), 0.0)

    return pd.DataFrame(
        {
            "timestamp": index,
            "trip_count": trip_count,
            "temperature_2m": temp.round(1),
            "precipitation": precip.round(2),
            "rain": precip.round(2),
            "snowfall": 0.0,
            "wind_speed_10m": rng.uniform(3, 15, len(index)).round(1),
            "relative_humidity_2m": rng.integers(40, 90, len(index)),
            "weather_code": np.where(precip > 0, 61, 1),
        }
    )
