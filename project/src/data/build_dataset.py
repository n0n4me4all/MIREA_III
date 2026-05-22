from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.download_weather import fetch_weather_hourly
from src.data.synthetic import generate_synthetic_hourly_demand
from src.utils.config import load_config, resolve_path
from src.utils.logging import setup_logging

logger = setup_logging()

TZ = "America/New_York"
START_COL_CANDIDATES = ("started_at", "starttime", "Start Time")


def _find_start_column(columns: pd.Index) -> str:
    for c in START_COL_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(f"Не найдена колонка времени старта. Доступны: {list(columns)}")


def parse_trip_start_times(series: pd.Series) -> pd.Series:
    """Парсинг started_at: локальное America/New_York (v3 CSV часто без суффикса TZ)."""
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        # Без суффикса TZ: неоднозначный час при откате DST → стандартное время (EST).
        return ts.dt.tz_localize(TZ, ambiguous=False, nonexistent="shift_forward")
    return ts.dt.tz_convert(TZ)


def floor_to_local_hour(ts: pd.Series) -> pd.Series:
    """Округление до часа; через UTC, чтобы не падать на переходе DST (ноябрь)."""
    return ts.dt.tz_convert("UTC").dt.floor("h").dt.tz_convert(TZ)


def _iter_trip_csv_files(csv_dirs: list[Path]):
    for csv_dir in csv_dirs:
        for csv_path in sorted(csv_dir.rglob("*.csv")):
            if "__MACOSX" in csv_path.parts or csv_path.name.startswith("._"):
                continue
            yield csv_path


def aggregate_trips_from_csv_dir(csv_dirs: list[Path]) -> pd.DataFrame:
    """Агрегирует поездки из всех CSV в почасовой trip_count."""
    chunks: list[pd.DataFrame] = []
    for csv_path in _iter_trip_csv_files(csv_dirs):
        logger.info("Чтение %s", csv_path.name)
        start_col: str | None = None
        for chunk in pd.read_csv(csv_path, chunksize=200_000, low_memory=False):
            if start_col is None:
                start_col = _find_start_column(chunk.columns)
            ts = floor_to_local_hour(parse_trip_start_times(chunk[start_col]))
            hourly = (
                ts
                .to_frame("timestamp")
                .assign(trip_count=1)
                .groupby("timestamp", as_index=False)["trip_count"]
                .sum()
            )
            chunks.append(hourly)
    if not chunks:
        return pd.DataFrame(columns=["timestamp", "trip_count"])
    trips = pd.concat(chunks, ignore_index=True)
    trips = trips.groupby("timestamp", as_index=False)["trip_count"].sum()
    return trips


def _reindex_hourly(trips: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=TZ)
    end_ts = pd.Timestamp(end, tz=TZ) + pd.Timedelta(hours=23)
    full_index = pd.date_range(start_ts, end_ts, freq="h", tz=TZ)
    trips = trips.set_index("timestamp").reindex(full_index)
    trips["trip_count"] = trips["trip_count"].fillna(0).astype(int)
    trips = trips.reset_index().rename(columns={"index": "timestamp"})
    return trips


WEATHER_FILL_COLS = (
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
    "weather_code",
)


def parse_weather_timestamps(series: pd.Series) -> pd.Series:
    """Парсинг timestamp погоды из CSV/API (в т.ч. смешанные -04:00/-05:00 в одном файле)."""
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.tz_convert(TZ)


def weather_covers_period(weather: pd.DataFrame, start: str, end: str) -> bool:
    """Проверяет, что файл погоды покрывает [start, end] по часам."""
    ts = parse_weather_timestamps(weather["timestamp"])
    if ts.isna().all():
        return False
    start_ts = pd.Timestamp(start, tz=TZ)
    end_ts = pd.Timestamp(end, tz=TZ) + pd.Timedelta(hours=23)
    return ts.min() <= start_ts and ts.max() >= end_ts


def merge_weather(trips: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.copy()
    weather["timestamp"] = parse_weather_timestamps(weather["timestamp"])
    weather = weather.drop_duplicates(subset=["timestamp"])
    merged = trips.merge(weather, on="timestamp", how="left")
    for col in WEATHER_FILL_COLS:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill()
    return merged


def load_weather_for_period(cfg: dict) -> pd.DataFrame:
    """Загружает погоду из CSV или API, если период не совпадает с config."""
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"]
    weather_path = resolve_path(cfg, "weather_raw_path")

    if weather_path.exists():
        weather = pd.read_csv(weather_path)
        if weather_covers_period(weather, start, end):
            return weather
        w_ts = parse_weather_timestamps(weather["timestamp"])
        logger.warning(
            "Погода в %s не покрывает %s — %s (файл: %s — %s), перезагрузка из API",
            weather_path,
            start,
            end,
            w_ts.min(),
            w_ts.max(),
        )

    logger.info("Загрузка погоды из Open-Meteo: %s — %s", start, end)
    weather = fetch_weather_hourly(
        start,
        end,
        cfg["weather"]["latitude"],
        cfg["weather"]["longitude"],
        cfg["project"]["timezone"],
    )
    weather_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(weather_path, index=False)
    return weather


def _citibike_raw_dirs_for_period(raw_dir: Path, start: str, end: str) -> list[Path]:
    """Подпапки raw/citibike, попадающие в [start, end]: 2023 или 2024-01 …"""
    start_p = pd.Period(start[:7], freq="M")
    end_p = pd.Period(end[:7], freq="M")
    selected: list[Path] = []
    for p in sorted(raw_dir.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name == "__pycache__":
            continue
        if p.name.isdigit() and len(p.name) == 4:
            year_p = pd.Period(f"{p.name}-01", freq="M")
            if start_p.year <= int(p.name) <= end_p.year:
                selected.append(p)
            continue
        if len(p.name) == 7 and p.name[4] == "-":
            try:
                folder_p = pd.Period(p.name, freq="M")
            except ValueError:
                continue
            if start_p <= folder_p <= end_p:
                selected.append(p)
    return selected


def build_from_raw(cfg: dict) -> pd.DataFrame:
    raw_dir = resolve_path(cfg, "citibike_raw_dir")
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"]
    csv_dirs = _citibike_raw_dirs_for_period(raw_dir, start, end)
    logger.info(
        "Чтение CSV из %d папок raw (период %s — %s)",
        len(csv_dirs),
        start,
        end,
    )
    if not csv_dirs:
        raise FileNotFoundError(
            f"Нет CSV в {raw_dir}. Сначала:\n"
            "  python -m src.data.download_citibike --list\n"
            "  python -m src.data.download_citibike\n"
            "или: python -m src.data.download_and_build"
        )

    trips = aggregate_trips_from_csv_dir(csv_dirs)
    trips = _reindex_hourly(trips, cfg["data"]["start_date"], cfg["data"]["end_date"])

    weather = load_weather_for_period(cfg)
    return merge_weather(trips, weather)


def refresh_weather_in_processed(cfg: dict) -> pd.DataFrame:
    """Пересобирает только слияние погоды в существующем parquet (без CSV поездок)."""
    out_path = resolve_path(cfg, "processed_path")
    if not out_path.exists():
        raise FileNotFoundError(f"Нет {out_path}. Сначала: python -m src.data.build_dataset")
    trips = pd.read_parquet(out_path)[["timestamp", "trip_count"]]
    weather = load_weather_for_period(cfg)
    return merge_weather(trips, weather)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборка hourly_demand.parquet")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Синтетический датасет")
    parser.add_argument(
        "--weather-only",
        action="store_true",
        help="Только обновить погоду в hourly_demand.parquet (без чтения CSV поездок)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = resolve_path(cfg, "processed_path")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.weather_only:
        df = refresh_weather_in_processed(cfg)
    elif args.synthetic or cfg["data"].get("use_synthetic", False):
        logger.info("Генерация синтетического датасета")
        df = generate_synthetic_hourly_demand(
            cfg["data"]["start_date"],
            cfg["data"]["end_date"],
            seed=cfg["project"].get("seed", 42),
        )
    else:
        try:
            df = build_from_raw(cfg)
        except FileNotFoundError as exc:
            logger.warning("%s — переключение на synthetic", exc)
            df = generate_synthetic_hourly_demand(
                cfg["data"]["start_date"],
                cfg["data"]["end_date"],
                seed=cfg["project"].get("seed", 42),
            )

    df.to_parquet(out_path, index=False)
    logger.info("Датасет сохранён: %s (%d строк)", out_path, len(df))


if __name__ == "__main__":
    main()
