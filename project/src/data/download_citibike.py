from __future__ import annotations

import argparse
import json
import zipfile
from io import BytesIO
from pathlib import Path

import requests

from src.data.archive_utils import ensure_csv_extracted, has_csv_data
from src.data.citibike_urls import (
    CITIBIKE_LIST_URL,
    DownloadPlanItem,
    fetch_s3_keys,
    list_period_coverage,
    plan_downloads,
    zip_download_url,
)
from src.utils.config import load_config, resolve_path
from src.utils.logging import setup_logging

logger = setup_logging()


def download_archive(item: DownloadPlanItem, raw_dir: Path) -> Path | None:
    """Скачивает и распаковывает один архив (годовой или месячный)."""
    out_dir = raw_dir / item.label
    if out_dir.exists() and has_csv_data(out_dir):
        logger.info("Уже есть CSV: %s", out_dir)
        return out_dir

    if out_dir.exists() and any(out_dir.rglob("*.zip")):
        logger.info("Архивы на диске, распаковка вложенных ZIP: %s", out_dir)
        return out_dir if ensure_csv_extracted(out_dir) else None

    url = zip_download_url(item.s3_key)
    logger.info("Скачивание [%s] %s -> %s", item.kind, item.s3_key, out_dir)

    try:
        resp = requests.get(url, timeout=1200)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Ошибка загрузки %s: %s", url, exc)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / item.s3_key
    zip_path.write_bytes(resp.content)
    logger.info("ZIP сохранён: %s (%.1f МБ)", zip_path, len(resp.content) / 1e6)

    try:
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            zf.extractall(out_dir)
    except zipfile.BadZipFile as exc:
        logger.error("Повреждённый ZIP %s: %s", item.s3_key, exc)
        return None

    csv_count = ensure_csv_extracted(out_dir)
    logger.info("Распаковано CSV: %d в %s", csv_count, out_dir)
    return out_dir if csv_count else None


def write_manifest(raw_dir: Path, results: dict[str, str]) -> None:
    manifest_path = raw_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Манифест: %s", manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Загрузка Citi Bike trip data (S3 bucket tripdata)"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Показать план загрузки для периода из config (без скачивания)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"]

    if args.list:
        print(f"S3 API: {CITIBIKE_LIST_URL}")
        print(f"Период: {start} .. {end}\n")
        print(f"{'Месяц':<10} {'Источник':<28} {'S3 key'}")
        print("-" * 70)
        for row in list_period_coverage(start, end):
            print(f"{row['month']:<10} {row['source']:<28} {row['s3_key']}")
        print("\nПодсказка: 2023 — один файл 2023-citibike-tripdata.zip на весь год;")
        print("2024 — помесячные 202401-citibike-tripdata.zip … 202412-citibike-tripdata.zip")
        return

    raw_dir = resolve_path(cfg, "citibike_raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    keys = fetch_s3_keys()
    logger.info("В бакете tripdata объектов: %d", len(keys))
    items = plan_downloads(start, end, keys)
    logger.info("К скачиванию архивов: %d", len(items))

    manifest: dict[str, str] = {}
    ok = 0
    for item in items:
        path = download_archive(item, raw_dir)
        key = f"{item.kind}:{item.s3_key}"
        if path:
            manifest[key] = "ok"
            ok += 1
        else:
            manifest[key] = "failed"

    write_manifest(raw_dir, manifest)
    logger.info("Успешно: %d из %d архивов", ok, len(items))
    if ok == 0:
        raise SystemExit("Ничего не скачано. Запустите: python -m src.data.download_citibike --list")


if __name__ == "__main__":
    main()
