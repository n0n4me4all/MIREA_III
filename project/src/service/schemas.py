from __future__ import annotations

from pydantic import BaseModel, Field


class HistoryPoint(BaseModel):
    """Одна точка истории спроса."""

    timestamp: str = Field(..., description="ISO-8601, часовая метка (America/New_York)")
    trip_count: int = Field(..., ge=0, description="Число поездок за час")


class WeatherPoint(BaseModel):
    """Погодные условия на один час."""

    timestamp: str = Field(..., description="ISO-8601")
    temperature_2m: float = Field(..., description="Температура, °C")
    precipitation: float = Field(0.0, ge=0, description="Осадки, мм")
    wind_speed_10m: float = Field(0.0, ge=0, description="Скорость ветра, м/с")
    relative_humidity_2m: float = Field(50.0, ge=0, le=100, description="Влажность, %")
    weather_code: int = Field(1, description="Код погоды WMO")


class PredictRequest(BaseModel):
    """Запрос прогноза на horizon часов."""

    history: list[HistoryPoint] = Field(
        ...,
        min_length=1,
        description="История спроса (рекомендуется >= 168 часов)",
    )
    future_weather: list[WeatherPoint] = Field(
        ...,
        min_length=1,
        description="Погода на будущие часы",
    )
    horizon: int = Field(24, ge=1, le=48, description="Горизонт прогноза в часах")


class ForecastPoint(BaseModel):
    timestamp: str
    predicted_trip_count: float = Field(..., ge=0)


class PredictResponse(BaseModel):
    """Ответ с почасовым прогнозом."""

    horizon: int
    model_version: str
    forecast: list[ForecastPoint]
    latency_ms: float = Field(..., ge=0, description="Время обработки, мс")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    service: str
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    target: str
    metrics: dict[str, float] | None = None
    final_model: str | None = None
