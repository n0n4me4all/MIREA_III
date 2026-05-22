from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MetricsResult:
    mae: float
    rmse: float
    smape: float


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsResult:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return MetricsResult(mae=float("nan"), rmse=float("nan"), smape=float("nan"))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return MetricsResult(mae=mae, rmse=rmse, smape=smape(y_true, y_pred))


def time_split(
    df: pd.DataFrame,
    test_days: int = 14,
    val_days: int = 14,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разбивает ряд по времени: train / val / test."""
    df = df.sort_values(timestamp_col)
    end = df[timestamp_col].max()
    test_start = end - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = df[df[timestamp_col] < val_start].copy()
    val = df[(df[timestamp_col] >= val_start) & (df[timestamp_col] < test_start)].copy()
    test = df[df[timestamp_col] >= test_start].copy()
    return train, val, test
