from __future__ import annotations

from src.data.citibike_urls import NYC_MONTHLY_ZIP, NYC_YEARLY_ZIP, plan_downloads


def test_regex_patterns() -> None:
    assert NYC_YEARLY_ZIP.match("2023-citibike-tripdata.zip")
    assert NYC_MONTHLY_ZIP.match("202406-citibike-tripdata.zip")
    assert not NYC_MONTHLY_ZIP.match("JC-202406-citibike-tripdata.csv.zip")


def test_plan_2023_yearly_and_2024_monthly() -> None:
    keys = (
        "2023-citibike-tripdata.zip",
        "202401-citibike-tripdata.zip",
        "202412-citibike-tripdata.zip",
    )
    plan = plan_downloads("2023-01-01", "2024-12-31", keys=keys)
    kinds = {p.s3_key: p.kind for p in plan}
    assert kinds["2023-citibike-tripdata.zip"] == "yearly"
    assert kinds["202401-citibike-tripdata.zip"] == "monthly"
    assert len([p for p in plan if p.kind == "monthly"]) >= 1
