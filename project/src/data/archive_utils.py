from __future__ import annotations

import zipfile
from pathlib import Path

from src.utils.logging import setup_logging

logger = setup_logging()


def _is_ignored_path(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name.startswith("._")


def count_csv_files(root: Path) -> int:
    return sum(1 for p in root.rglob("*.csv") if not _is_ignored_path(p))


def has_csv_data(root: Path) -> bool:
    return count_csv_files(root) > 0


def extract_nested_zips(root: Path, *, max_rounds: int = 50) -> int:
    """Распаковывает все .zip под root (годовой архив → помесячные → CSV)."""
    extracted = 0
    for _ in range(max_rounds):
        zips = sorted(p for p in root.rglob("*.zip") if not _is_ignored_path(p))
        if not zips:
            break
        for zip_path in zips:
            dest = zip_path.parent
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(dest)
                zip_path.unlink(missing_ok=True)
                extracted += 1
            except zipfile.BadZipFile as exc:
                logger.warning("Пропуск повреждённого ZIP %s: %s", zip_path, exc)
    return extracted


def ensure_csv_extracted(out_dir: Path) -> int:
    """Распаковывает вложенные архивы и возвращает число CSV."""
    nested = extract_nested_zips(out_dir)
    csv_count = count_csv_files(out_dir)
    if nested:
        logger.info("Вложенных ZIP: %d, CSV: %d в %s", nested, csv_count, out_dir)
    return csv_count
