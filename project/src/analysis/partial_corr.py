from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _residualize(series: pd.Series, controls: pd.DataFrame) -> np.ndarray:
    """Остатки линейной регрессии series ~ controls (без перехвата — чистый partial)."""
    mask = series.notna() & controls.notna().all(axis=1)
    y = series.loc[mask].to_numpy(dtype=float)
    z = controls.loc[mask].to_numpy(dtype=float)
    if len(y) < 3:
        return np.array([])
    pred = LinearRegression().fit(z, y).predict(z)
    return y - pred


def partial_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    controls: list[str],
) -> float:
    """
    Частная корреляция corr(x, y | controls).

    Сначала убираем линейное влияние controls из x и y, затем считаем Pearson
    между остатками.
    """
    ctrl = df[controls]
    rx = _residualize(df[x], ctrl)
    ry = _residualize(df[y], ctrl)
    n = min(len(rx), len(ry))
    if n < 3:
        return float("nan")
    rx, ry = rx[:n], ry[:n]
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def weather_partial_correlations(
    df: pd.DataFrame,
    target: str = "trip_count",
    calendar_controls: list[str] | None = None,
) -> pd.DataFrame:
    """Частные корреляции погодных признаков со спросом при контроле календаря."""
    from src.features.build_features import CALENDAR_FEATURE_COLUMNS, WEATHER_FEATURE_COLUMNS

    controls = calendar_controls or CALENDAR_FEATURE_COLUMNS
    rows: list[dict[str, float | str]] = []
    for col in WEATHER_FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        raw = float(df[target].corr(df[col])) if df[col].notna().any() else float("nan")
        partial = partial_correlation(df, col, target, controls)
        rows.append(
            {
                "feature": col,
                "pearson_raw": raw,
                "partial_vs_calendar": partial,
            }
        )
    return pd.DataFrame(rows)
