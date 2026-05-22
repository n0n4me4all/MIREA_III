from __future__ import annotations

import logging
import sys
from typing import Literal


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Настраивает корневой логгер проекта."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger("citibike")
