"""Экспорт PNG-графиков для отчёта и защиты."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import project_root


def main() -> None:
    root = project_root()
    artifacts = root / "artifacts"
    figures = artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    ablation_path = artifacts / "ablation_metrics.json"
    rows = [{"model": k, **v} for k, v in metrics["models"].items()]
    if ablation_path.exists():
        ab = json.loads(ablation_path.read_text(encoding="utf-8"))
        seen = {r["model"] for r in rows}
        for k, v in ab["models"].items():
            if k not in seen:
                rows.append({"model": k, **v})
    df_m = pd.DataFrame(rows).sort_values("mae")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(df_m["model"], df_m["mae"], color="steelblue")
    ax.set_xlabel("MAE (поездок/час)")
    ax.set_title("Сравнение моделей на test")
    ax.invert_yaxis()
    fig.tight_layout()
    p1 = figures / "metrics_comparison.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    print("Wrote", p1)

    pred = pd.read_csv(artifacts / "predictions_test.csv")
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    sub = pred.head(72)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(sub["timestamp"], sub["trip_count"], label="факт", linewidth=2)
    ax.plot(sub["timestamp"], sub["y_pred"], label="прогноз", linestyle="--")
    ax.legend()
    ax.set_title("Test: факт vs прогноз (первые 72 ч)")
    fig.autofmt_xdate()
    fig.tight_layout()
    p2 = figures / "forecast_test.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    print("Wrote", p2)

    partial_path = artifacts / "partial_correlations.csv"
    if partial_path.exists():
        partial = pd.read_csv(partial_path)
        fig, ax = plt.subplots(figsize=(8, 3))
        x = range(len(partial))
        w = 0.35
        ax.bar([i - w / 2 for i in x], partial["pearson_raw"], width=w, label="raw")
        ax.bar(
            [i + w / 2 for i in x],
            partial["partial_vs_calendar"],
            width=w,
            label="partial | calendar",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(partial["feature"], rotation=25, ha="right")
        ax.legend()
        ax.set_title("Погода vs спрос")
        fig.tight_layout()
        p3 = figures / "partial_corr.png"
        fig.savefig(p3, dpi=120)
        plt.close(fig)
        print("Wrote", p3)

    parquet = root / "data" / "processed" / "hourly_demand.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        monthly = df.set_index("timestamp")["trip_count"].resample("ME").mean()
        fig, ax = plt.subplots(figsize=(10, 3))
        monthly.plot(ax=ax)
        ax.set_title("Средний почасовой спрос по месяцам")
        ax.set_ylabel("trip_count")
        fig.tight_layout()
        p4 = figures / "demand_monthly.png"
        fig.savefig(p4, dpi=120)
        plt.close(fig)
        print("Wrote", p4)


if __name__ == "__main__":
    main()
