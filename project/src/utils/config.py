from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def project_root() -> Path:
    """Корень папки project/ (родитель src/)."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Загружает YAML-конфиг и подмешивает переменные окружения.

    CONFIG_PATH из .env переопределяет путь к файлу конфигурации.
    """
    root = project_root()
    load_dotenv(root / ".env")

    path = config_path or os.getenv("CONFIG_PATH", "configs/config.yaml")
    path = Path(path)
    if not path.is_absolute():
        path = root / path

    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")

    with path.open(encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    cfg["_paths"] = {
        "root": str(root),
        "config": str(path),
    }
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Разрешает относительный путь из секции data/model относительно корня project/."""
    root = Path(cfg["_paths"]["root"])
    for section in ("data", "model", "service"):
        if key in cfg.get(section, {}):
            p = Path(cfg[section][key])
            if p.is_absolute():
                return p
            return root / p
    raise KeyError(f"Путь '{key}' не найден в конфиге")
