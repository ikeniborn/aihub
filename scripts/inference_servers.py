"""
inference_servers.py — статус и команды запуска для llama.cpp и vLLM.

Только stdlib, нет внешних зависимостей.
Поддерживает OpenAI-совместимый API (GET /health, GET /v1/models).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import urlparse


def get_server_status(host: str, server_type: str, timeout: int = 3) -> dict:
    """
    Проверяет доступность inference-сервера и возвращает список загруженных моделей.

    Returns:
        {"ok": bool, "models": [str], "error": str|None, "host": str}

    Edge cases:
        - Сервер не запущен        → ok=False, error="Connection refused: ..."
        - llama.cpp грузится       → ok=False, error="Server loading"
        - Сервер без моделей       → ok=True,  models=[]
        - Таймаут 3с               → ok=False, error="Timeout after Ns"
        - /v1/models пустой список → ok=True,  models=[]
    """
    host = host.rstrip("/")
    # Bypass system proxy (HTTP_PROXY / HTTPS_PROXY) для локальных демонов
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # 1. Проверка /health
    try:
        req = urllib.request.Request(
            f"{host}/health",
            headers={"Accept": "application/json"},
        )
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                health = json.loads(body)
            except json.JSONDecodeError:
                health = {}
            status_val = health.get("status", "")
            # llama.cpp возвращает "loading" пока грузит модель
            if isinstance(status_val, str) and status_val.lower() == "loading":
                return {"ok": False, "models": [], "error": "Server loading", "host": host}
    except urllib.error.URLError as e:
        reason = str(e.reason) if hasattr(e, "reason") else str(e)
        return {"ok": False, "models": [], "error": f"Connection refused: {reason}", "host": host}
    except TimeoutError:
        return {"ok": False, "models": [], "error": f"Timeout after {timeout}s", "host": host}
    except Exception as e:
        # socket.timeout и прочие исключения
        msg = str(e)
        if "timed out" in msg.lower():
            return {"ok": False, "models": [], "error": f"Timeout after {timeout}s", "host": host}
        return {"ok": False, "models": [], "error": f"Connection refused: {msg}", "host": host}

    # 2. Получение списка моделей через /v1/models
    try:
        req2 = urllib.request.Request(
            f"{host}/v1/models",
            headers={"Accept": "application/json"},
        )
        with opener.open(req2, timeout=timeout) as resp2:
            body2 = resp2.read().decode("utf-8", errors="replace")
            try:
                mdata = json.loads(body2)
            except json.JSONDecodeError:
                mdata = {}
            model_list = mdata.get("data") or []
            names = [
                m["id"]
                for m in model_list
                if isinstance(m, dict) and m.get("id")
            ]
    except Exception:
        # /v1/models недоступен — сервер жив, но список пуст
        names = []

    return {"ok": True, "models": names, "error": None, "host": host}


def generate_serve_command(
    local_path: str,
    server_type: str,
    host: str = "",
    extra_args: str = "",
) -> str:
    """
    Генерирует команду запуска inference-сервера для указанного файла модели.

    Args:
        local_path:  Полный путь к файлу модели.
        server_type: "llamacpp" или "vllm".
        host:        URL сервера (для извлечения порта).
        extra_args:  Дополнительные аргументы (добавляются в конец).

    Returns:
        Строку с командой запуска.
    """
    # Извлечь порт из host URL
    port = _default_port(server_type)
    if host:
        try:
            parsed = urlparse(host)
            if parsed.port:
                port = parsed.port
        except Exception:
            pass

    if server_type == "llamacpp":
        cmd = f"llama-server -m {local_path} --port {port} -c 4096 --host 0.0.0.0"
    elif server_type == "vllm":
        cmd = f"vllm serve {local_path} --port {port} --host 0.0.0.0"
    else:
        cmd = f"# unknown server_type: {server_type}"

    if extra_args:
        cmd = f"{cmd} {extra_args.strip()}"

    return cmd


def _default_port(server_type: str) -> int:
    """Возвращает порт по умолчанию для указанного типа сервера."""
    return 8000 if server_type == "vllm" else 8080
