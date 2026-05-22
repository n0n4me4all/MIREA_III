from __future__ import annotations

"""
Полный пайплайн реальных данных: Citi Bike + погода + parquet.

Запуск из папки project/:
    python -m src.data.download_and_build
    python -m src.data.download_and_build --config configs/config.yaml
"""

import argparse
import subprocess
import sys

from src.utils.config import project_root


def _run(cmd: list[str], cwd) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать данные и собрать hourly_demand.parquet")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-citibike", action="store_true")
    parser.add_argument("--skip-weather", action="store_true")
    args = parser.parse_args()

    root = project_root()
    py = sys.executable
    cfg = args.config

    if not args.skip_citibike:
        _run([py, "-m", "src.data.download_citibike", "--config", cfg], root)
    if not args.skip_weather:
        _run([py, "-m", "src.data.download_weather", "--config", cfg], root)
    _run([py, "-m", "src.data.build_dataset", "--config", cfg], root)
    print("\nГотово. Дальше: python -m src.models.train")


if __name__ == "__main__":
    main()
