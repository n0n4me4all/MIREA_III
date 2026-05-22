from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.partial_corr import weather_partial_correlations
from src.features.build_features import (
    CALENDAR_FEATURE_COLUMNS,
    LAG_FEATURE_COLUMNS,
    WEATHER_FEATURE_COLUMNS,
    build_feature_matrix,
    get_feature_columns,
)
from src.models.baselines import predict_seasonal_naive
from src.models.evaluate import regression_metrics, time_split
from src.models.train import SEED
from src.utils.logging import setup_logging

logger = setup_logging()

# Наборы признаков для ablation (прогноз на test с известными лагами в прошлом).
MODEL_FEATURE_SETS: dict[str, list[str]] = {
    "weather_only_boosting": list(WEATHER_FEATURE_COLUMNS),
    "calendar_only_boosting": list(CALENDAR_FEATURE_COLUMNS),
    "lags_calendar_boosting": CALENDAR_FEATURE_COLUMNS + LAG_FEATURE_COLUMNS,
    "full_boosting": get_feature_columns(),
}


def _fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    model_kind: str,
) -> tuple[pd.DataFrame, Any]:
    train_clean = train_df.dropna(subset=feature_cols + ["trip_count"])
    test_clean = test_df.dropna(subset=feature_cols)
    x_train = train_clean[feature_cols]
    y_train = train_clean["trip_count"]
    x_test = test_clean[feature_cols]

    if model_kind == "ridge":
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, random_state=SEED)),
            ]
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            random_state=SEED,
        )

    model.fit(x_train, y_train)
    test_clean = test_clean.copy()
    test_clean["y_pred"] = model.predict(x_test)
    return test_clean, model


def run_ablation(
    featured: pd.DataFrame,
    test_days: int,
    val_days: int,
) -> dict[str, Any]:
    train_df, val_df, test_df = time_split(featured, test_days=test_days, val_days=val_days)
    train_fit = pd.concat([train_df, val_df], ignore_index=True)

    results: dict[str, Any] = {"models": {}, "feature_sets": MODEL_FEATURE_SETS}

    for lag, label in [(24, "seasonal_naive_24h"), (168, "seasonal_naive_168h")]:
        preds = predict_seasonal_naive(test_df, lag_hours=lag)
        m = regression_metrics(test_df["trip_count"].to_numpy(), preds)
        results["models"][label] = {"mae": m.mae, "rmse": m.rmse, "smape": m.smape}

    for name, cols in MODEL_FEATURE_SETS.items():
        pred_df, _ = _fit_predict(train_fit, test_df, cols, "boosting")
        m = regression_metrics(
            pred_df["trip_count"].to_numpy(),
            pred_df["y_pred"].to_numpy(),
        )
        results["models"][name] = {"mae": m.mae, "rmse": m.rmse, "smape": m.smape}
        logger.info("%s: MAE=%.2f", name, m.mae)

    # Линейная модель на всех признаках — интерпретируемые коэффициенты.
    full_cols = get_feature_columns()
    _, ridge_model = _fit_predict(train_fit, test_df, full_cols, "ridge")
    ridge = ridge_model.named_steps["ridge"]
    coefs = pd.Series(ridge.coef_, index=full_cols).sort_values(key=abs, ascending=False)
    results["ridge_coefficients_top10"] = coefs.head(10).to_dict()

    pred_df, _ = _fit_predict(train_fit, test_df, full_cols, "ridge")
    m = regression_metrics(pred_df["trip_count"].to_numpy(), pred_df["y_pred"].to_numpy())
    results["models"]["ridge_full"] = {"mae": m.mae, "rmse": m.rmse, "smape": m.smape}

    clean = featured.dropna(subset=get_feature_columns())
    results["partial_correlations"] = weather_partial_correlations(clean).to_dict(orient="records")

    full_mae = results["models"]["full_boosting"]["mae"]
    lags_mae = results["models"]["lags_calendar_boosting"]["mae"]
    weather_mae = results["models"]["weather_only_boosting"]["mae"]
    results["interpretation"] = {
        "weather_only_mae_minus_full": weather_mae - full_mae,
        "lags_calendar_mae_minus_full": lags_mae - full_mae,
        "full_minus_naive_24h": full_mae - results["models"]["seasonal_naive_24h"]["mae"],
        "note": (
            "weather_only_mae_minus_full >> 0: без лагов погода не прогнозирует спрос. "
            "lags_calendar_mae_minus_full < 0: на test лаги+календарь могут быть чуть лучше полной "
            "(погода коррелирует, но не всегда улучшает MAE). "
            "partial_vs_calendar — связь погоды со спросом после учёта часа/дня недели."
        ),
    }

    return results


def save_ablation(results: dict[str, Any], artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifacts_dir / "ablation_metrics.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Ablation: %s", json_path)

    partial = pd.DataFrame(results["partial_correlations"])
    csv_path = artifacts_dir / "partial_correlations.csv"
    partial.to_csv(csv_path, index=False)
    logger.info("Partial corr: %s", csv_path)
