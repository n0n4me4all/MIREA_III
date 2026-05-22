from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.models.predict import load_model_bundle
from src.utils.config import load_config, resolve_path


class ModelStore:
    """Хранилище загруженной модели и метаданных."""

    def __init__(self) -> None:
        self._bundle: dict[str, Any] | None = None
        self._metrics: dict[str, Any] | None = None
        self._cfg: dict[str, Any] | None = None

    def load(self, config_path: str | None = None) -> None:
        self._cfg = load_config(config_path)
        artifact_path = resolve_path(self._cfg, "artifact_path")
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {artifact_path}. Запустите: python -m src.models.train"
            )
        self._bundle = load_model_bundle(artifact_path)

        metrics_path = resolve_path(self._cfg, "metrics_path")
        if metrics_path.exists():
            self._metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    @property
    def bundle(self) -> dict[str, Any]:
        if self._bundle is None:
            raise RuntimeError("Модель не загружена")
        return self._bundle

    @property
    def cfg(self) -> dict[str, Any]:
        if self._cfg is None:
            raise RuntimeError("Конфиг не загружен")
        return self._cfg

    @property
    def metrics(self) -> dict[str, Any] | None:
        return self._metrics

    def artifact_path(self) -> Path:
        return resolve_path(self.cfg, "artifact_path")

    def feature_schema_path(self) -> Path:
        return resolve_path(self.cfg, "feature_schema_path")


store = ModelStore()
