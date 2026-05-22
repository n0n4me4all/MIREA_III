from __future__ import annotations

import numpy as np
import pandas as pd


def predict_seasonal_naive(
    df: pd.DataFrame,
    lag_hours: int,
    target: str = "trip_count",
) -> np.ndarray:
    """
    Прогноз: значение target со сдвигом lag_hours.

    Для строк без достаточной истории возвращает NaN.
    """
    return df[target].shift(lag_hours).to_numpy()


def seasonal_naive_forecast_vector(
    history: pd.Series,
    future_index: pd.DatetimeIndex,
    lag_hours: int,
) -> np.ndarray:
    """Почасовой прогноз naive для будущих меток времени."""
    full_index = history.index.union(future_index)
    full = history.reindex(full_index)
    preds = full.shift(lag_hours).loc[future_index]
    return preds.fillna(full.median()).to_numpy()
