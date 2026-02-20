"""
Shared utilities for AI model downloader scripts.
Импортируется через sys.path из каждого скрипта в scripts/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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


@dataclass
class ProxyConfig:
    """
    Конфигурация HTTP-прокси.

    Управляется двумя переменными окружения:
      PROXY_ENABLED=true   — включить proxy (по умолчанию: false)
      PROXY_URL=http://...  — адрес прокси-сервера

    Поддерживаемые схемы PROXY_URL:
      http://host:port                   — HTTP proxy
      https://host:port                  — HTTPS proxy
      http://user:password@host:port     — с аутентификацией
      socks5://host:port                 — SOCKS5, локальный DNS    (требует: pip install requests[socks])
      socks5h://host:port                — SOCKS5, удалённый DNS    (требует: pip install requests[socks])

    Дополнительно поддерживается секция proxy: в credentials.yaml (ниже приоритетом, чем env):
      proxy:
        enabled: true
        url: "http://proxy.example.com:8080"
    """

    enabled: bool = False
    url: str = ""

    @property
    def valid(self) -> bool:
        """True если proxy включён и URL задан."""
        return self.enabled and bool(self.url)

    def apply_to_env(self) -> None:
        """
        Устанавливает HTTP_PROXY / HTTPS_PROXY в os.environ.
        Вызывается один раз после загрузки конфига — до любых HTTP-запросов.
        Библиотеки requests, huggingface_hub, botocore читают эти переменные автоматически.
        """
        if not self.valid:
            return
        os.environ.setdefault("HTTP_PROXY", self.url)
        os.environ.setdefault("HTTPS_PROXY", self.url)
        os.environ.setdefault("http_proxy", self.url)
        os.environ.setdefault("https_proxy", self.url)

    def boto3_proxies(self) -> Optional[dict]:
        """
        Возвращает словарь proxies для явной передачи в botocore.config.Config.
        None если proxy не активен.
        """
        if not self.valid:
            return None
        return {"http": self.url, "https": self.url}


def load_proxy_config(creds_path: Optional[Path] = None) -> ProxyConfig:
    """
    Загружает конфигурацию proxy из переменных окружения и credentials.yaml.

    Приоритет (от высокого к низкому):
      1. PROXY_ENABLED / PROXY_URL из переменных окружения (или .env)
      2. Секция proxy: в credentials.yaml

    Возвращает ProxyConfig. Если proxy не настроен — возвращает ProxyConfig(enabled=False).
    """
    # Читаем из env (загружен через load_dotenv() перед этим вызовом)
    env_enabled = os.environ.get("PROXY_ENABLED", "").strip().lower()
    env_url = os.environ.get("PROXY_URL", "").strip()

    # Загружаем credentials.yaml как fallback
    yaml_enabled: Optional[str] = None
    yaml_url: str = ""

    if creds_path and creds_path.is_file():
        try:
            with creds_path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            proxy_section = raw.get("proxy", {}) or {}
            raw_yaml_enabled = proxy_section.get("enabled")
            if raw_yaml_enabled is not None:
                yaml_enabled = str(raw_yaml_enabled).strip().lower()
            yaml_url = str(proxy_section.get("url", "") or "").strip()
        except Exception:
            pass

    # Применяем приоритет: env > yaml
    enabled_str = env_enabled if env_enabled else yaml_enabled
    url = env_url if env_url else yaml_url

    enabled = enabled_str in ("1", "true", "yes", "on")

    return ProxyConfig(enabled=enabled, url=url)


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
