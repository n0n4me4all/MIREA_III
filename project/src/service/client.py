from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx

from src.utils.config import project_root

BASE_URL = "http://localhost:8000"


def test_health(client: httpx.Client) -> dict[str, Any]:
    print("GET /health")
    r = client.get("/health")
    r.raise_for_status()
    data = r.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def test_predict(client: httpx.Client, sample_path: Path) -> dict[str, Any]:
    print("POST /predict")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    r = client.post("/predict", json=payload)
    r.raise_for_status()
    data = r.json()
    print(f"horizon={data['horizon']} latency_ms={data['latency_ms']:.1f}")
    print("Первые 3 точки прогноза:")
    for pt in data["forecast"][:3]:
        print(f"  {pt['timestamp']}: {pt['predicted_trip_count']}")
    return data


def main() -> None:
    sample = project_root() / "artifacts" / "sample_request.json"
    if not sample.exists():
        print(f"Не найден {sample}. Сначала: python -m src.models.train", file=sys.stderr)
        sys.exit(1)

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        test_health(client)
        test_predict(client, sample)


if __name__ == "__main__":
    main()
