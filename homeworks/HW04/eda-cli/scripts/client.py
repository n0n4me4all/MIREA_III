from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


BASE_URL = "http://localhost:8000"


def test_quality_endpoint(client: httpx.Client) -> dict[str, Any]:
    """Тестирует эндпоинт POST /quality с разными параметрами."""
    print("Тестирование POST /quality")

    test_cases = [
        {
            "name": "Хороший датасет",
            "data": {
                "n_rows": 5000,
                "n_cols": 20,
                "max_missing_share": 0.1,
                "numeric_cols": 10,
                "categorical_cols": 10,
            },
        },
        {
            "name": "Маленький датасет",
            "data": {
                "n_rows": 50,
                "n_cols": 5,
                "max_missing_share": 0.2,
                "numeric_cols": 3,
                "categorical_cols": 2,
            },
        },
        {
            "name": "Много пропусков",
            "data": {
                "n_rows": 2000,
                "n_cols": 15,
                "max_missing_share": 0.8,
                "numeric_cols": 8,
                "categorical_cols": 7,
            },
        },
        {
            "name": "Слишком много колонок",
            "data": {
                "n_rows": 3000,
                "n_cols": 150,
                "max_missing_share": 0.15,
                "numeric_cols": 75,
                "categorical_cols": 75,
            },
        },
    ]

    results = []
    for test_case in test_cases:
        try:
            response = client.post(
                f"{BASE_URL}/quality",
                json=test_case["data"],
                timeout=10.0,
            )
            response.raise_for_status()
            result = response.json()

            results.append(
                {
                    "name": test_case["name"],
                    "status": response.status_code,
                    "quality_score": result.get("quality_score", 0.0),
                    "ok_for_model": result.get("ok_for_model", False),
                    "latency_ms": result.get("latency_ms", 0.0),
                    "message": result.get("message", ""),
                }
            )

            print(f"\n {test_case['name']}")
            print(f"  Статус: {response.status_code}")
            print(f"  Quality Score: {result.get('quality_score', 0.0):.3f}")
            print(f"  OK for model: {result.get('ok_for_model', False)}")
            print(f"  Latency: {result.get('latency_ms', 0.0):.2f} ms")
            print(f"  Сообщение: {result.get('message', '')[:60]}...")

        except httpx.HTTPStatusError as e:
            print(f"\n {test_case['name']} - Ошибка: {e.response.status_code}")
            results.append(
                {
                    "name": test_case["name"],
                    "status": e.response.status_code,
                    "error": str(e),
                }
            )
        except Exception as e:
            print(f"\n {test_case['name']} - Ошибка: {e}")
            results.append(
                {
                    "name": test_case["name"],
                    "status": "error",
                    "error": str(e),
                }
            )

    return {"endpoint": "/quality", "results": results}


def test_quality_from_csv_endpoint(
    client: httpx.Client, csv_files: list[Path]
) -> dict[str, Any]:
    """Тестирует эндпоинт POST /quality-from-csv с разными CSV-файлами."""
    print("Тестирование POST /quality-from-csv")

    results = []
    for csv_file in csv_files:
        if not csv_file.exists():
            print(f"\n Файл не найден: {csv_file}")
            continue

        try:
            with open(csv_file, "rb") as f:
                files = {"file": (csv_file.name, f, "text/csv")}
                response = client.post(
                    f"{BASE_URL}/quality-from-csv",
                    files=files,
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()

                results.append(
                    {
                        "file": csv_file.name,
                        "status": response.status_code,
                        "quality_score": result.get("quality_score", 0.0),
                        "ok_for_model": result.get("ok_for_model", False),
                        "latency_ms": result.get("latency_ms", 0.0),
                        "n_rows": result.get("dataset_shape", {}).get("n_rows", 0),
                        "n_cols": result.get("dataset_shape", {}).get("n_cols", 0),
                    }
                )

                print(f"\n {csv_file.name}")
                print(f"  Статус: {response.status_code}")
                print(f"  Размер: {result.get('dataset_shape', {}).get('n_rows', 0)} строк × {result.get('dataset_shape', {}).get('n_cols', 0)} колонок")
                print(f"  Quality Score: {result.get('quality_score', 0.0):.3f}")
                print(f"  OK for model: {result.get('ok_for_model', False)}")
                print(f"  Latency: {result.get('latency_ms', 0.0):.2f} ms")

        except httpx.HTTPStatusError as e:
            print(f"\n {csv_file.name} - Ошибка: {e.response.status_code}")
            if e.response.status_code == 400:
                try:
                    error_detail = e.response.json().get("detail", "Unknown error")
                    print(f"  Детали: {error_detail}")
                except Exception:
                    print(f"  Детали: {e.response.text[:100]}")
            results.append(
                {
                    "file": csv_file.name,
                    "status": e.response.status_code,
                    "error": str(e),
                }
            )
        except Exception as e:
            print(f"\n {csv_file.name} - Ошибка: {e}")
            results.append(
                {
                    "file": csv_file.name,
                    "status": "error",
                    "error": str(e),
                }
            )

    return {"endpoint": "/quality-from-csv", "results": results}


def test_quality_flags_from_csv_endpoint(
    client: httpx.Client, csv_files: list[Path]
) -> dict[str, Any]:
    """Тестирует эндпоинт POST /quality-flags-from-csv с разными CSV-файлами."""
    print("Тестирование POST /quality-flags-from-csv")

    results = []
    for csv_file in csv_files:
        if not csv_file.exists():
            print(f"\n Файл не найден: {csv_file}")
            continue

        try:
            with open(csv_file, "rb") as f:
                files = {"file": (csv_file.name, f, "text/csv")}
                response = client.post(
                    f"{BASE_URL}/quality-flags-from-csv",
                    files=files,
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()

                flags = result.get("flags", {})
                results.append(
                    {
                        "file": csv_file.name,
                        "status": response.status_code,
                        "flags_count": len(flags),
                        "flags": flags,
                    }
                )

                print(f"\n {csv_file.name}")
                print(f"  Статус: {response.status_code}")
                print(f"  Количество флагов: {len(flags)}")
                print("  Флаги:")
                for flag_name, flag_value in sorted(flags.items()):
                    print(f"    {flag_name}: {flag_value}")

        except httpx.HTTPStatusError as e:
            print(f"\n {csv_file.name} - Ошибка: {e.response.status_code}")
            if e.response.status_code == 400:
                try:
                    error_detail = e.response.json().get("detail", "Unknown error")
                    print(f"  Детали: {error_detail}")
                except Exception:
                    print(f"  Детали: {e.response.text[:100]}")
            results.append(
                {
                    "file": csv_file.name,
                    "status": e.response.status_code,
                    "error": str(e),
                }
            )
        except Exception as e:
            print(f"\n {csv_file.name} - Ошибка: {e}")
            results.append(
                {
                    "file": csv_file.name,
                    "status": "error",
                    "error": str(e),
                }
            )

    return {"endpoint": "/quality-flags-from-csv", "results": results}


def print_summary(all_results: list[dict[str, Any]]) -> None:
    """Выводит сводку по всем тестам."""
    print("СВОДКА ПО ВСЕМ ТЕСТАМ")

    total_requests = 0
    successful_requests = 0
    total_latency = 0.0
    quality_scores = []

    for endpoint_result in all_results:
        endpoint = endpoint_result["endpoint"]
        results = endpoint_result["results"]

        print(f"\n{endpoint}:")
        for result in results:
            total_requests += 1
            if result.get("status") == 200:
                successful_requests += 1
                if "latency_ms" in result:
                    total_latency += result["latency_ms"]
                if "quality_score" in result:
                    quality_scores.append(result["quality_score"])

    print(f"\nОбщая статистика:")
    print(f"  Всего запросов: {total_requests}")
    print(f"  Успешных: {successful_requests}")
    print(f"  Неудачных: {total_requests - successful_requests}")

    if successful_requests > 0:
        avg_latency = total_latency / successful_requests
        print(f"  Средняя задержка: {avg_latency:.2f} ms")

    if quality_scores:
        avg_score = sum(quality_scores) / len(quality_scores)
        min_score = min(quality_scores)
        max_score = max(quality_scores)
        print(f"  Quality Score:")
        print(f"    Средний: {avg_score:.3f}")
        print(f"    Минимальный: {min_score:.3f}")
        print(f"    Максимальный: {max_score:.3f}")


def main() -> None:
    """Основная функция клиента."""
    import sys

    # Определяем путь к папке data относительно скрипта
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    # CSV-файлы для тестирования
    csv_files = [
        data_dir / "example.csv",
    ]

    # Добавляем дополнительные файлы, если они есть
    if (data_dir / "S02-hw-dataset.csv").exists():
        csv_files.append(data_dir / "S02-hw-dataset.csv")

    print("КЛИЕНТ ДЛЯ ТЕСТИРОВАНИЯ API СЕРВИСА КАЧЕСТВА ДАТАСЕТОВ")
    print(f"Базовый URL: {BASE_URL}")
    print(f"CSV-файлы для тестирования: {len(csv_files)}")

    # Проверяем доступность сервера
    try:
        with httpx.Client(timeout=5.0) as client:
            health_response = client.get(f"{BASE_URL}/health")
            health_response.raise_for_status()
            print(f"\n Сервер доступен: {health_response.json()}")
    except Exception as e:
        print(f"\n Сервер недоступен: {e}")
        print("Убедитесь, что сервер запущен:")
        print("  uv run uvicorn eda_cli.api:app --port 8000")
        sys.exit(1)

    # Выполняем тесты
    all_results = []

    with httpx.Client(timeout=60.0) as client:
        # Тест /quality
        all_results.append(test_quality_endpoint(client))

        # Тест /quality-from-csv
        if csv_files:
            all_results.append(test_quality_from_csv_endpoint(client, csv_files))

        # Тест /quality-flags-from-csv
        if csv_files:
            all_results.append(test_quality_flags_from_csv_endpoint(client, csv_files))

    # Выводим сводку
    print_summary(all_results)

    print("Тестирование завершено!")


if __name__ == "__main__":
    main()

