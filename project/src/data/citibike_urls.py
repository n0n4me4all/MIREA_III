from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import requests

CITIBIKE_LIST_URL = "https://s3.amazonaws.com/tripdata/?list-type=2"
CITIBIKE_BASE = "https://s3.amazonaws.com/tripdata"

# Помесячные архивы NYC (без Jersey City JC-)
NYC_MONTHLY_ZIP = re.compile(r"^(?P<yyyymm>20\d{2}(0[1-9]|1[0-2]))-citibike-tripdata\.zip$")
# Годовые архивы: 2022-citibike-tripdata.zip, 2023-citibike-tripdata.zip
NYC_YEARLY_ZIP = re.compile(r"^(?P<year>20\d{2})-citibike-tripdata\.zip$")


@dataclass(frozen=True)
class DownloadPlanItem:
    """Один файл для скачивания."""

    kind: str  # "yearly" | "monthly"
    s3_key: str
    label: str  # папка в raw_dir: "2023" или "2024-06"


@lru_cache(maxsize=1)
def fetch_s3_keys() -> tuple[str, ...]:
    """Список ключей объектов в бакете tripdata (S3 ListObjectsV2 XML)."""
    resp = requests.get(CITIBIKE_LIST_URL, timeout=120)
    resp.raise_for_status()
    keys = re.findall(r"<Key>([^<]+)</Key>", resp.text)
    return tuple(keys)


def zip_download_url(s3_key: str) -> str:
    return f"{CITIBIKE_BASE}/{s3_key}"


def plan_downloads(start_date: str, end_date: str, keys: tuple[str, ...] | None = None) -> list[DownloadPlanItem]:
    """
    План загрузки NYC Citi Bike за период [start_date, end_date].

    - Для целого года, если на S3 есть `YYYY-citibike-tripdata.zip`, качаем один годовой архив.
    - Иначе — помесячные `YYYYMM-citibike-tripdata.zip`.
  Jersey City (префикс JC-) не используем.
    """
    import pandas as pd

    all_keys = keys if keys is not None else fetch_s3_keys()
    key_set = set(all_keys)

    yearly: dict[str, str] = {}
    monthly: dict[str, str] = {}
    for k in all_keys:
        m_y = NYC_YEARLY_ZIP.match(k)
        if m_y:
            yearly[m_y.group("year")] = k
            continue
        m_m = NYC_MONTHLY_ZIP.match(k)
        if m_m:
            monthly[m_m.group("yyyymm")] = k

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    plan: list[DownloadPlanItem] = []

    for year in range(start.year, end.year + 1):
        year_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        if year_start > year_end:
            continue

        ystr = str(year)
        if ystr in yearly and year_start.month == 1 and year_end.month == 12 and year_start.day == 1:
            # весь календарный год в диапазоне — один годовой zip
            plan.append(DownloadPlanItem("yearly", yearly[ystr], ystr))
            continue

        # помесячно
        months = pd.period_range(year_start.strftime("%Y-%m"), year_end.strftime("%Y-%m"), freq="M")
        for p in months:
            yyyymm = p.strftime("%Y%m")
            ym = p.strftime("%Y-%m")
            if yyyymm in monthly:
                plan.append(DownloadPlanItem("monthly", monthly[yyyymm], ym))
            elif ystr in yearly:
                # частичный год, но есть годовой архив — качаем год один раз
                if not any(item.label == ystr and item.kind == "yearly" for item in plan):
                    plan.append(DownloadPlanItem("yearly", yearly[ystr], ystr))

    # дедупликация по s3_key
    seen: set[str] = set()
    unique: list[DownloadPlanItem] = []
    for item in plan:
        if item.s3_key not in seen:
            seen.add(item.s3_key)
            unique.append(item)
    return unique


def list_period_coverage(start_date: str, end_date: str) -> list[dict[str, str]]:
    """Таблица: месяц → что будет скачано."""
    import pandas as pd

    keys = fetch_s3_keys()
    plan = plan_downloads(start_date, end_date, keys)
    plan_by_label = {p.label: p for p in plan}

    rows: list[dict[str, str]] = []
    for ym in pd.period_range(start=start_date, end=end_date, freq="M").strftime("%Y-%m"):
        year = ym[:4]
        if ym in plan_by_label:
            p = plan_by_label[ym]
            rows.append({"month": ym, "source": p.kind, "s3_key": p.s3_key})
        elif year in plan_by_label:
            p = plan_by_label[year]
            rows.append({"month": ym, "source": f"yearly ({p.s3_key})", "s3_key": p.s3_key})
        else:
            rows.append({"month": ym, "source": "MISSING", "s3_key": "-"})
    return rows
