#!/usr/bin/env python3
"""
AI Model Browser
================
Ищет модели на HuggingFace Hub по API.
Поддерживает regex-фильтрацию по названию, автору, тегам и именам файлов.
Выводит таблицу результатов или YAML-фрагмент для вставки в models.yaml.

Примеры использования:
    python scripts/browse_models.py --query "qwen"
    python scripts/browse_models.py --query "llama" --regex ".*8[Bb].*"
    python scripts/browse_models.py --author bartowski --file-regex "Q4_K_M\\.gguf$"
    python scripts/browse_models.py --tags gguf text-generation --limit 30
    python scripts/browse_models.py --query "embedding" --show-files --yaml
    python scripts/browse_models.py --author unsloth --file-regex ".*14B.*Q4.*" --yaml
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_hf_token, load_proxy_config  # noqa: E402


# ─── Search ───────────────────────────────────────────────────────────────────


def search_models(
    query: Optional[str],
    author: Optional[str],
    tags: list[str],
    regex: Optional[str],
    limit: int,
    token: Optional[str],
    sort: str = "downloads",
    pipeline_tag: Optional[str] = None,
    library: Optional[str] = None,
    language: Optional[str] = None,
) -> list:
    """
    Ищет модели на HuggingFace Hub.

    Args:
        query:        полнотекстовый поиск (по имени и описанию)
        author:       фильтр по автору/организации (точное совпадение)
        tags:         список тегов для фильтрации
        regex:        regex-паттерн по model_id (применяется после API-запроса)
        limit:        максимальное кол-во результатов в итоге
        token:        HF-токен
        sort:         сортировка (downloads | likes | lastModified)
        pipeline_tag: задача модели (text-generation, text-to-image, ...)
        library:      формат/библиотека (gguf, safetensors, transformers, ...)
        language:     язык модели (ru, en, zh, ...)

    Returns список ModelInfo объектов.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Запрашиваем с запасом если есть regex-фильтр (он отсеет часть результатов)
    fetch_limit = limit * 5 if regex else limit

    kwargs: dict = {
        "limit": fetch_limit,
        "sort": sort,
        "direction": -1,   # по убыванию
    }
    if query:
        kwargs["search"] = query
    if author:
        kwargs["author"] = author
    if tags:
        kwargs["filter"] = tags
    if pipeline_tag:
        kwargs["pipeline_tag"] = pipeline_tag
    if library:
        kwargs["library"] = library
    if language:
        kwargs["language"] = language

    models = list(api.list_models(**kwargs))

    # Применяем regex к model_id
    if regex:
        pattern = re.compile(regex, re.IGNORECASE)
        models = [m for m in models if pattern.search(m.id)]

    return models[:limit]


def list_files_for_repo(
    model_id: str,
    file_regex: Optional[str],
    token: Optional[str],
) -> list[str]:
    """
    Возвращает список файлов в репозитории с опциональной regex-фильтрацией.
    """
    from huggingface_hub import list_repo_files

    try:
        files = list(list_repo_files(model_id, token=token))
    except Exception as exc:
        print(f"  [WARN] Не удалось получить файлы репозитория {model_id!r}: {exc}", file=sys.stderr)
        return []

    if file_regex:
        pattern = re.compile(file_regex, re.IGNORECASE)
        files = [f for f in files if pattern.search(f)]

    return sorted(files)


# ─── Output ───────────────────────────────────────────────────────────────────


def print_models_table(models: list) -> None:
    """Выводит таблицу найденных моделей."""
    if not models:
        print("Ничего не найдено.")
        return

    print(
        f"\n{'#':<4}  {'MODEL ID':<52}  {'TASK':<22}  {'DOWNLOADS':>9}  {'LIKES':>6}  ТЕГИ"
    )
    print("-" * 130)
    for i, m in enumerate(models, 1):
        dl = getattr(m, "downloads", 0) or 0
        likes = getattr(m, "likes", 0) or 0
        task = getattr(m, "pipeline_tag", None) or ""
        tags_raw = getattr(m, "tags", None) or []
        # Показываем только значимые теги (не license:, arxiv:, и т.п.)
        tags = [t for t in tags_raw if ":" not in t][:4]
        tags_str = textwrap.shorten(", ".join(tags), 30)
        print(f"{i:<4}  {m.id:<52}  {task:<22}  {dl:>9,}  {likes:>6}  {tags_str}")
    print(f"\nНайдено: {len(models)} моделей\n")


def print_files_section(model_id: str, files: list[str]) -> None:
    """Выводит список файлов одного репозитория."""
    if not files:
        print("    (файлы не найдены или доступ ограничен)")
        return
    for f in files:
        print(f"    {f}")


def to_yaml_snippet(models: list, file_map: dict[str, list[str]]) -> str:
    """
    Генерирует YAML-фрагмент для вставки в models.yaml.
    Если у модели есть файлы — создаёт отдельную запись на каждый файл.
    """
    entries = []
    for m in models:
        files = file_map.get(m.id, [])
        gated = getattr(m, "gated", False) or False
        base = {
            "dest_dir": "misc",
            "enabled": False,
            "gated": gated,
            "tags": [],
            "vram_gb": 0,
            "description": "",
        }
        if files:
            for fname in files:
                entry = {"repo_id": m.id, "filename": fname, **base}
                entries.append(entry)
        else:
            entry = {"repo_id": m.id, "filename": "FILENAME_HERE", **base}
            entries.append(entry)

    return yaml.dump(entries, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="browse_models.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    search_group = p.add_argument_group("Поиск")
    search_group.add_argument(
        "--query", "-q",
        metavar="TEXT",
        help="Полнотекстовый поиск (по имени модели, описанию)",
    )
    search_group.add_argument(
        "--author", "-a",
        metavar="NAME",
        help="Фильтр по автору или организации (например: bartowski, unsloth, Qwen)",
    )
    search_group.add_argument(
        "--tags", "-t",
        nargs="+",
        metavar="TAG",
        help="Фильтр по тегам (например: gguf text-generation)",
    )
    search_group.add_argument(
        "--regex", "-r",
        metavar="PATTERN",
        help="Regex-фильтр по model_id (регистронезависимый, применяется после API-запроса)",
    )
    search_group.add_argument(
        "--pipeline-tag", "-p",
        metavar="TASK",
        help=(
            "Фильтр по задаче модели (нативный API-параметр HuggingFace).\n"
            "Переменная окружения: BROWSE_PIPELINE_TAG.\n"
            "Примеры: text-generation  text-to-image  automatic-speech-recognition\n"
            "         text-classification  sentence-similarity  image-classification"
        ),
    )
    search_group.add_argument(
        "--library",
        metavar="LIB",
        help=(
            "Фильтр по формату/библиотеке (нативный API-параметр HuggingFace).\n"
            "Переменная окружения: BROWSE_LIBRARY.\n"
            "Примеры: gguf  safetensors  transformers  diffusers  onnx"
        ),
    )
    search_group.add_argument(
        "--language",
        metavar="LANG",
        help=(
            "Фильтр по языку модели (код ISO 639-1).\n"
            "Переменная окружения: BROWSE_LANGUAGE.\n"
            "Примеры: ru  en  zh  de  fr"
        ),
    )
    search_group.add_argument(
        "--sort",
        default="downloads",
        choices=["downloads", "likes", "lastModified"],
        help="Сортировка результатов (по умолчанию: downloads)",
    )
    search_group.add_argument(
        "--limit", "-n",
        type=int,
        default=20,
        metavar="N",
        help="Максимальное количество результатов (по умолчанию: 20)",
    )

    files_group = p.add_argument_group("Файлы в репозитории")
    files_group.add_argument(
        "--show-files",
        action="store_true",
        help="Показать список файлов каждого найденного репозитория",
    )
    files_group.add_argument(
        "--file-regex", "-f",
        metavar="PATTERN",
        help=(
            "Regex-фильтр по именам файлов внутри репозитория (автоматически включает --show-files).\n"
            "Переменная окружения: BROWSE_FILE_REGEX (CLI имеет приоритет).\n"
            "Примеры: 'Q4_K_M\\.gguf$'   '.*14B.*Q4'   '\\.safetensors$'"
        ),
    )

    output_group = p.add_argument_group("Вывод")
    output_group.add_argument(
        "--yaml",
        action="store_true",
        help="Вывести YAML-фрагмент для вставки в models.yaml",
    )

    p.add_argument(
        "--creds",
        default="credentials.yaml",
        metavar="FILE",
        help="Файл с секретами (по умолчанию: credentials.yaml)",
    )

    return p


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    args = build_parser().parse_args()

    if not any([args.query, args.author, args.tags, args.regex,
                args.pipeline_tag, args.library, args.language]):
        print(
            "[ERROR] Укажите хотя бы один параметр поиска:\n"
            "  --query TEXT | --author NAME | --tags TAG ...\n"
            "  --pipeline-tag TASK | --library LIB | --language LANG | --regex PATTERN",
            file=sys.stderr,
        )
        return 1

    load_dotenv(dotenv_path=".env", override=False)

    # Env-дефолты для фильтров (CLI имеет приоритет над .env)
    _env_defaults = [
        ("file_regex",    "BROWSE_FILE_REGEX"),
        ("pipeline_tag",  "BROWSE_PIPELINE_TAG"),
        ("library",       "BROWSE_LIBRARY"),
        ("language",      "BROWSE_LANGUAGE"),
    ]
    for attr, env_key in _env_defaults:
        if getattr(args, attr, None) is None:
            val = os.environ.get(env_key, "").strip() or None
            if val:
                setattr(args, attr, val)
                print(f"[INFO] {env_key} из .env: {val!r}")

    # Configure proxy (reads PROXY_ENABLED + PROXY_URL from env / credentials.yaml)
    proxy_cfg = load_proxy_config(Path(args.creds))
    if proxy_cfg.valid:
        proxy_cfg.apply_to_env()
        print(f"[INFO] Proxy enabled: {proxy_cfg.url}")

    token = load_hf_token(Path(args.creds))

    show_files = args.show_files or bool(args.file_regex)

    # Вывод параметров поиска
    params = []
    if args.query:
        params.append(f"query={args.query!r}")
    if args.author:
        params.append(f"author={args.author!r}")
    if args.tags:
        params.append(f"tags={args.tags}")
    if args.regex:
        params.append(f"regex={args.regex!r}")
    if args.pipeline_tag:
        params.append(f"pipeline_tag={args.pipeline_tag!r}")
    if args.library:
        params.append(f"library={args.library!r}")
    if args.language:
        params.append(f"language={args.language!r}")
    if args.file_regex:
        params.append(f"file-regex={args.file_regex!r}")
    print(f"[INFO] Поиск: {', '.join(params)}  (limit={args.limit}, sort={args.sort})")

    try:
        models = search_models(
            query=args.query,
            author=args.author,
            tags=args.tags or [],
            regex=args.regex,
            limit=args.limit,
            token=token,
            sort=args.sort,
            pipeline_tag=args.pipeline_tag,
            library=args.library,
            language=args.language,
        )
    except Exception as exc:
        print(f"[ERROR] Ошибка HuggingFace API: {exc}", file=sys.stderr)
        return 1

    if not models:
        print("Ничего не найдено. Попробуйте изменить параметры поиска.")
        return 0

    print_models_table(models)

    # Загрузка файлов репозиториев (если нужно)
    file_map: dict[str, list[str]] = {}
    if show_files:
        print(f"Загружаю список файлов ({len(models)} репозиториев) ...\n")
        for m in models:
            files = list_files_for_repo(m.id, args.file_regex, token)
            file_map[m.id] = files
            if files:
                print(f"  [{m.id}]")
                print_files_section(m.id, files)
            else:
                if args.file_regex:
                    pass   # не выводим репо без совпадающих файлов
                else:
                    print(f"  [{m.id}]")
                    print_files_section(m.id, files)
        print()

    # YAML-вывод
    if args.yaml:
        # Если был --file-regex, оставляем только модели с файлами
        if args.file_regex:
            models_with_files = [m for m in models if file_map.get(m.id)]
            if not models_with_files:
                print("[WARN] Ни одна модель не содержит файлов по заданному --file-regex")
                return 0
            snippet_models = models_with_files
        else:
            snippet_models = models

        print("# ── YAML-фрагмент для models.yaml ─────────────────────────────────────────")
        print("# Скопируйте нужные записи в раздел 'models:' файла models.yaml.")
        print("# Заполните: dest_dir, tags, vram_gb, description, enabled: true\n")
        print(to_yaml_snippet(snippet_models, file_map))

    return 0


if __name__ == "__main__":
    sys.exit(main())
