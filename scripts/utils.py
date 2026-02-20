"""
Shared utilities for AI model downloader scripts.
Импортируется через sys.path из каждого скрипта в scripts/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union, Optional

import yaml


def fmt_size(n: Union[int, float]) -> str:
    """Форматирует размер в байтах в человекочитаемую строку (например, 4.2 GB)."""
    if not n:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_hf_token(creds_path: Path) -> Optional[str]:
    """
    Загружает HuggingFace токен из credentials.yaml или переменной окружения HF_TOKEN.
    credentials.yaml имеет приоритет над окружением.
    """
    token: Optional[str] = None
    if creds_path.is_file():
        with creds_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        token = (raw.get("huggingface", {}) or {}).get("token") or None
    if not token:
        token = os.environ.get("HF_TOKEN") or None
    return token
