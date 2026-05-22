from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from time import perf_counter

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.models.predict import predict_hourly
from src.service.model_loader import store
from src.service.schemas import (
    ForecastPoint,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.utils.logging import setup_logging

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
    try:
        store.load(config_path)
        logger.info("Модель загружена: %s", store.artifact_path())
    except FileNotFoundError as exc:
        logger.error("%s", exc)
    yield


app = FastAPI(
    title="Citi Bike Hourly Demand Forecast API",
    version="1.0.0",
    description="Сервис почасового прогноза спроса Citi Bike с учётом погоды и календаря.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Проверка работоспособности сервиса и наличия модели."""
    return HealthResponse(
        status="ok",
        model_loaded=store.loaded,
        service="citibike-hourly-demand",
        version="1.0.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    """Метаданные и метрики финальной модели."""
    if not store.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    bundle = store.bundle
    metrics_block = None
    final_name = bundle.get("model_name", "unknown")
    if store.metrics:
        final_name = store.metrics.get("final_model", final_name)
        winner = store.metrics.get("models", {}).get(final_name, {})
        metrics_block = {k: float(v) for k, v in winner.items() if isinstance(v, (int, float))}

    return ModelInfoResponse(
        model_name=bundle.get("model_name", "unknown"),
        target="hourly_trip_count",
        metrics=metrics_block,
        final_model=final_name,
    )


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(req: PredictRequest) -> PredictResponse:
    """
    Прогноз числа поездок Citi Bike на следующие `horizon` часов.

    Требуется история не короче `min_history_hours` (по умолчанию 168).
    """
    if not store.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена. Сначала обучите модель.")

    min_hist = store.cfg["model"]["min_history_hours"]
    if len(req.history) < min_hist:
        raise HTTPException(
            status_code=400,
            detail=f"История слишком короткая: нужно >= {min_hist} часов, получено {len(req.history)}",
        )
    if len(req.future_weather) < req.horizon:
        raise HTTPException(
            status_code=400,
            detail=f"Нужно >= {req.horizon} точек future_weather, получено {len(req.future_weather)}",
        )

    start = perf_counter()

    history_df = pd.DataFrame([h.model_dump() for h in req.history])
    weather_df = pd.DataFrame([w.model_dump() for w in req.future_weather])

    try:
        preds = predict_hourly(
            history_df,
            weather_df,
            store.artifact_path(),
            store.feature_schema_path(),
            horizon=req.horizon,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    timestamps = weather_df["timestamp"].head(req.horizon)
    forecast = [
        ForecastPoint(
            timestamp=ts if isinstance(ts, str) else pd.Timestamp(ts).isoformat(),
            predicted_trip_count=round(float(p), 2),
        )
        for ts, p in zip(timestamps, preds)
    ]

    latency_ms = (perf_counter() - start) * 1000.0
    logger.info(
        "predict horizon=%d history_len=%d latency_ms=%.1f",
        req.horizon,
        len(req.history),
        latency_ms,
    )

    return PredictResponse(
        horizon=req.horizon,
        model_version=store.bundle.get("model_name", "v1"),
        forecast=forecast,
        latency_ms=latency_ms,
    )
