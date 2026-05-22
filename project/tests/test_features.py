from __future__ import annotations

import pandas as pd

from src.data.synthetic import generate_synthetic_hourly_demand
from src.features.build_features import add_lag_features, build_feature_matrix, get_feature_columns


def test_lag_features_no_future_leakage() -> None:
    df = generate_synthetic_hourly_demand(start="2024-06-01", end="2024-06-07", seed=1)
    featured = add_lag_features(df)
    # lag_1h в строке i должен равняться trip_count в i-1
    row = featured.iloc[50]
    prev = featured.iloc[49]["trip_count"]
    assert row["lag_1h"] == prev


def test_build_feature_matrix_has_expected_columns() -> None:
    df = generate_synthetic_hourly_demand(start="2024-06-01", end="2024-06-14", seed=2)
    featured = build_feature_matrix(df)
    cols = get_feature_columns()
    for c in cols:
        assert c in featured.columns


def test_train_rows_dropna_after_warmup() -> None:
    df = generate_synthetic_hourly_demand(start="2024-06-01", end="2024-08-31", seed=3)
    featured = build_feature_matrix(df)
    cols = get_feature_columns()
    clean = featured.dropna(subset=cols + ["trip_count"])
    assert len(clean) > 1000
