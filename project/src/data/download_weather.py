from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import load_config, resolve_path
from src.utils.logging import setup_logging

logger = setup_logging()

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_hourly(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Загружает почасовую погоду из Open-Meteo Historical API."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "rain",
                "snowfall",
                "weather_code",
                "wind_speed_10m",
            ]
        ),
        "timezone": timezone,
    }
    logger.info(
        "Запрос погоды Open-Meteo: %s — %s (NYC, tz=%s)",
        start_date,
        end_date,
        timezone,
    )
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(
            timezone, ambiguous=False, nonexistent="shift_forward"
        )
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(timezone)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Загрузка почасовой погоды NYC")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = resolve_path(cfg, "weather_raw_path")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_weather_hourly(
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        latitude=cfg["weather"]["latitude"],
        longitude=cfg["weather"]["longitude"],
        timezone=cfg["project"]["timezone"],
    )
    df.to_csv(out_path, index=False)
    logger.info("Погода сохранена: %s (%d строк)", out_path, len(df))


if __name__ == "__main__":
    main()
