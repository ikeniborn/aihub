#!/usr/bin/env python3
"""
AI Model Downloader
===================
Downloads AI models from HuggingFace Hub according to models.yaml.
Optionally syncs downloaded models to S3-compatible object storage.

Features:
- Idempotent: skips already-downloaded files (ETag-based update detection)
- Atomic downloads: no partial files left on failure
- Gated model support: reads HF_TOKEN from credentials.yaml or .env
- Fast downloads: optional hf_transfer (Rust-based, set HF_HUB_ENABLE_HF_TRANSFER=1)
- S3 sync: upload models to AWS S3, MinIO, Yandex Object Storage, Cloudflare R2
- S3 idempotency: skip upload if object already exists and size matches
- Dry-run mode: validates model list without downloading
- Filter by tag or model name

Usage:
    python scripts/download_models.py                     # Download all enabled models
    python scripts/download_models.py --dry-run           # Validate only, no downloads
    python scripts/download_models.py --force             # Re-download even if file exists
    python scripts/download_models.py --model Phi-4       # Filter by repo_id substring
    python scripts/download_models.py --tag russian       # Filter by tag
    python scripts/download_models.py --list              # List all models and exit
    python scripts/download_models.py --upload-s3         # Also upload to S3 after download
    python scripts/download_models.py --s3-only           # Upload to S3, skip local storage
    python scripts/download_models.py --config custom.yaml
    python scripts/download_models.py --creds my-creds.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Shared utilities (fmt_size, load_hf_token, load_proxy_config)
sys.path.insert(0, str(Path(__file__).parent))
from utils import fmt_size, load_hf_token, load_proxy_config  # noqa: E402


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class ModelEntry:
    """One model entry from models.yaml."""

    repo_id: str
    filename: str
    dest_dir: str = "misc"
    enabled: bool = True
    gated: bool = False
    tags: list[str] = field(default_factory=list)
    vram_gb: float = 0.0
    description: str = ""


@dataclass
class S3Config:
    """S3 / compatible object storage configuration."""

    access_key_id: str = ""
    secret_access_key: str = ""
    region: str = "us-east-1"
    endpoint_url: str = ""   # Empty = standard AWS; set for MinIO, Yandex, etc.
    bucket: str = ""
    prefix: str = "models"

    @property
    def valid(self) -> bool:
        return bool(self.bucket and self.access_key_id and self.secret_access_key)


@dataclass
class DownloadResult:
    """Result of a single model download attempt."""

    model: ModelEntry
    status: str           # SKIP | DOWNLOAD | ERROR | DRYRUN | DISABLED
    local_path: Optional[Path] = None
    error: Optional[str] = None
    size_bytes: int = 0
    s3_status: str = ""   # UPLOADED | SKIP | ERROR | "" (not attempted)
    s3_error: Optional[str] = None


# ─── Credentials Loading ──────────────────────────────────────────────────────


def load_credentials(creds_path: Path) -> tuple[Optional[str], S3Config]:
    """
    Загружает секреты из credentials.yaml, конфиг — из переменных окружения (.env).

    Разделение ответственности:
      credentials.yaml  — только секреты: HF token, S3 access_key_id / secret_access_key
      .env              — только параметры: bucket, region, prefix, endpoint_url

    Структура credentials.yaml:
        huggingface:
          token: hf_...
        s3:
          access_key_id: ...
          secret_access_key: ...

    Returns (hf_token, S3Config).
    """
    # HF token: делегируем в utils.load_hf_token (единое место загрузки)
    hf_token: Optional[str] = load_hf_token(creds_path)

    # S3 secrets из credentials.yaml, fallback на env
    access_key_id = ""
    secret_access_key = ""

    if creds_path.is_file():
        with creds_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        s3_section = raw.get("s3", {}) or {}
        access_key_id = s3_section.get("access_key_id", "")
        secret_access_key = s3_section.get("secret_access_key", "")
        print(f"[INFO] Credentials loaded from: {creds_path}")
    else:
        if creds_path.name != "credentials.yaml":
            print(f"[WARN] Credentials file not found: {creds_path}", file=sys.stderr)

    if not access_key_id:
        access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", "")
    if not secret_access_key:
        secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

    # Конфиг подключения к S3 — только из переменных окружения (.env)
    s3 = S3Config(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        region=os.environ.get("S3_REGION", "us-east-1"),
        endpoint_url=os.environ.get("S3_ENDPOINT_URL", ""),
        bucket=os.environ.get("S3_BUCKET", ""),
        prefix=os.environ.get("S3_PREFIX", "models"),
    )

    return hf_token, s3


# ─── Configuration Loading ────────────────────────────────────────────────────


def load_config(config_path: Path) -> tuple[dict, list[ModelEntry]]:
    """Parse models.yaml; return (settings dict, list of ModelEntry)."""
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    settings: dict = raw.get("settings", {})
    models: list[ModelEntry] = []

    for item in raw.get("models", []):
        try:
            entry = ModelEntry(
                repo_id=item["repo_id"],
                filename=item["filename"],
                dest_dir=item.get("dest_dir", "misc"),
                enabled=item.get("enabled", True),
                gated=item.get("gated", False),
                tags=item.get("tags", []),
                vram_gb=float(item.get("vram_gb", 0)),
                description=item.get("description", ""),
            )
            models.append(entry)
        except KeyError as exc:
            print(f"[WARN] Skipping malformed model entry (missing {exc}): {item}")

    return settings, models


# ─── ETag Helpers ─────────────────────────────────────────────────────────────


def _etag_path(local_file: Path) -> Path:
    """Sidecar file storing the last-known ETag next to the model file."""
    return local_file.with_suffix(local_file.suffix + ".etag")


def _read_local_etag(local_file: Path) -> Optional[str]:
    """Return locally cached ETag string, or None if not available."""
    p = _etag_path(local_file)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip() or None
    return None


def _write_local_etag(local_file: Path, etag: str) -> None:
    """Persist the ETag alongside the model file."""
    _etag_path(local_file).write_text(etag, encoding="utf-8")


def _fetch_remote_etag(repo_id: str, filename: str, token: Optional[str]) -> Optional[str]:
    """
    Fetch remote ETag via a lightweight HTTP HEAD request using
    huggingface_hub.get_hf_file_metadata().  Returns None on any error.
    """
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        url = hf_hub_url(repo_id, filename)
        meta = get_hf_file_metadata(url, token=token)
        return meta.etag
    except Exception:  # noqa: BLE001
        return None


# ─── Idempotency Check ────────────────────────────────────────────────────────


def check_existing(
    local_file: Path,
    model: ModelEntry,
    update_policy: str,
    token: Optional[str],
    force: bool,
) -> str:
    """
    Determine whether download is needed.

    Returns:
        "DOWNLOAD" — file is missing or outdated
        "SKIP"     — file is current, no download needed
    """
    if force:
        return "DOWNLOAD"

    if not local_file.is_file():
        return "DOWNLOAD"

    if update_policy == "skip":
        return "SKIP"

    if update_policy == "etag":
        local_etag = _read_local_etag(local_file)
        if local_etag is None:
            remote_etag = _fetch_remote_etag(model.repo_id, model.filename, token)
            if remote_etag is None:
                tqdm.write(
                    f"  [WARN] Cannot determine ETag for {model.repo_id}/{model.filename};"
                    " defaulting to SKIP (use --force to override)"
                )
                return "SKIP"
            _write_local_etag(local_file, remote_etag)
            return "SKIP"

        remote_etag = _fetch_remote_etag(model.repo_id, model.filename, token)
        if remote_etag is None:
            tqdm.write(
                f"  [WARN] Cannot fetch remote ETag for {model.repo_id}/{model.filename};"
                " keeping existing file"
            )
            return "SKIP"

        if local_etag != remote_etag:
            tqdm.write(f"  [INFO] ETag changed — will re-download {model.filename}")
            return "DOWNLOAD"

        return "SKIP"

    # update_policy == "always"
    return "DOWNLOAD"


# ─── S3 Helpers ───────────────────────────────────────────────────────────────


def _s3_key(model: ModelEntry, prefix: str) -> str:
    """Compute S3 object key for a model: {prefix}/{dest_dir}/{filename}."""
    prefix = prefix.rstrip("/")
    if prefix:
        return f"{prefix}/{model.dest_dir}/{model.filename}"
    return f"{model.dest_dir}/{model.filename}"


def _s3_client(cfg: S3Config, proxies: Optional[dict] = None) -> Any:
    """Create a boto3 S3 client from S3Config. Returns boto3.client (typed as Any).

    Args:
        cfg:     S3 connection configuration.
        proxies: Optional dict {"http": url, "https": url} for proxy routing.
                 Passed explicitly to botocore so it works regardless of env var state.
    """
    try:
        import boto3
        from botocore.config import Config as BotocoreConfig
    except ImportError:
        raise ImportError("boto3 is required for S3 support. Run: pip install boto3")

    kwargs: dict = {
        "aws_access_key_id": cfg.access_key_id,
        "aws_secret_access_key": cfg.secret_access_key,
        "region_name": cfg.region,
    }
    if cfg.endpoint_url:
        kwargs["endpoint_url"] = cfg.endpoint_url
    if proxies:
        kwargs["config"] = BotocoreConfig(proxies=proxies)
    return boto3.client("s3", **kwargs)


def s3_check_exists(
    model: ModelEntry,
    local_file: Optional[Path],
    cfg: S3Config,
    update_policy: str,
    force: bool,
) -> str:
    """
    Check whether the model already exists in S3 with the expected size.

    Returns:
        "SKIP"   — object is current in S3, no upload needed
        "UPLOAD" — object missing or outdated, needs upload
        "ERROR"  — S3 connectivity or permission error
    """
    if force or update_policy == "always":
        return "UPLOAD"

    key = _s3_key(model, cfg.prefix)
    try:
        client = _s3_client(cfg)
        resp = client.head_object(Bucket=cfg.bucket, Key=key)
        remote_size = resp.get("ContentLength", -1)

        if update_policy == "skip":
            return "SKIP"

        # etag policy: compare file sizes (S3 multipart ETags are not MD5)
        if local_file and local_file.is_file():
            local_size = local_file.stat().st_size
            if remote_size == local_size:
                return "SKIP"
            return "UPLOAD"

        return "SKIP"

    except Exception as exc:
        err_str = str(exc)
        # boto3 ClientError для 404: "An error occurred (404) when calling the HeadObject..."
        # или "An error occurred (NoSuchKey) when calling the HeadObject..."
        if any(x in err_str for x in ("404", "NoSuchKey", "Not Found")):
            return "UPLOAD"
        # Auth errors, network failures, etc.
        return "ERROR"


def s3_upload_file(
    local_file: Path,
    model: ModelEntry,
    cfg: S3Config,
) -> tuple[str, Optional[str]]:
    """
    Upload a local file to S3 using multipart for large files.

    Returns:
        ("UPLOADED", None)       on success
        ("ERROR",    error_msg)  on failure
    """
    key = _s3_key(model, cfg.prefix)
    try:
        client = _s3_client(cfg)
        from boto3.s3.transfer import TransferConfig

        file_size = local_file.stat().st_size
        transfer_cfg = TransferConfig(
            multipart_threshold=100 * 1024 * 1024,   # 100 MB
            multipart_chunksize=100 * 1024 * 1024,
            max_concurrency=4,
        )

        endpoint_display = cfg.endpoint_url or "s3.amazonaws.com"
        tqdm.write(
            f"  Uploading to s3://{cfg.bucket}/{key}"
            f"  [{endpoint_display}]  ({fmt_size(file_size)}) ..."
        )
        client.upload_file(
            Filename=str(local_file),
            Bucket=cfg.bucket,
            Key=key,
            Config=transfer_cfg,
        )
        return "UPLOADED", None

    except Exception as exc:
        return "ERROR", str(exc)


# ─── Download ─────────────────────────────────────────────────────────────────


def download_model(
    model: ModelEntry,
    models_root: Path,
    token: Optional[str],
    update_policy: str,
    force: bool,
    dry_run: bool,
    retry_count: int = 3,
    retry_delay: float = 5.0,
) -> DownloadResult:
    """
    Download a single model file.  Idempotent: checks ETag before downloading.
    Uses hf_hub_download() which writes atomically (temp file → rename).
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    local_dir = models_root / model.dest_dir
    local_file = local_dir / model.filename

    # ── Dry-run: validate repo exists ────────────────────────────────────────
    if dry_run:
        try:
            api = HfApi(token=token)
            api.repo_info(repo_id=model.repo_id, repo_type="model")
            return DownloadResult(model=model, status="DRYRUN", local_path=local_file)
        except RepositoryNotFoundError:
            return DownloadResult(
                model=model,
                status="ERROR",
                error=f"Repository not found: {model.repo_id}",
            )
        except Exception as exc:  # noqa: BLE001
            return DownloadResult(
                model=model,
                status="ERROR",
                error=f"Validation failed: {exc}",
            )

    # ── Check if download is needed ───────────────────────────────────────────
    decision = check_existing(local_file, model, update_policy, token, force)
    if decision == "SKIP":
        return DownloadResult(
            model=model,
            status="SKIP",
            local_path=local_file,
            size_bytes=local_file.stat().st_size,
        )

    # ── Perform download with retry ───────────────────────────────────────────
    local_dir.mkdir(parents=True, exist_ok=True)
    last_exc: Optional[Exception] = None

    for attempt in range(retry_count + 1):
        try:
            if attempt > 0:
                tqdm.write(f"  [RETRY {attempt}/{retry_count}] {model.repo_id}/{model.filename}")
            else:
                tqdm.write(f"  Downloading {model.repo_id}/{model.filename} ...")

            downloaded_path = hf_hub_download(
                repo_id=model.repo_id,
                filename=model.filename,
                local_dir=str(local_dir),
                token=token,
            )
            downloaded = Path(downloaded_path)

            if not downloaded.is_file():
                raise FileNotFoundError(f"Expected file not found after download: {downloaded}")

            size = downloaded.stat().st_size
            if size == 0:
                raise ValueError("Downloaded file is empty — possible network error")

            # If HF placed the file in a cache subdir, copy it to flat dest_dir
            if downloaded.parent != local_dir:
                shutil.copy2(str(downloaded), str(local_file))
                downloaded = local_file

            # Store ETag for future idempotency checks
            remote_etag = _fetch_remote_etag(model.repo_id, model.filename, token)
            if remote_etag:
                _write_local_etag(downloaded, remote_etag)

            return DownloadResult(
                model=model,
                status="DOWNLOAD",
                local_path=downloaded,
                size_bytes=downloaded.stat().st_size,
            )

        except EntryNotFoundError:
            # Файл не найден в репо — не имеет смысла повторять
            return DownloadResult(
                model=model,
                status="ERROR",
                error=(
                    f"File '{model.filename}' not found in repo '{model.repo_id}'.\n"
                    f"  Check the filename at: https://huggingface.co/{model.repo_id}"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()

            if attempt >= retry_count:
                break  # все попытки исчерпаны

            # Задержка перед повтором: rate limit — x10 от базовой, остальное — экспоненциально
            if is_rate_limit:
                wait = retry_delay * 10
                tqdm.write(f"  [WARN] Rate limit (429) — ждём {wait:.0f}s перед повтором ...")
            else:
                wait = retry_delay * (2 ** attempt)
                tqdm.write(f"  [WARN] Ошибка (попытка {attempt + 1}/{retry_count + 1}): {exc}")
                tqdm.write(f"         Повтор через {wait:.0f}s ...")
            time.sleep(wait)

    return DownloadResult(model=model, status="ERROR", error=str(last_exc))


# ─── S3 Sync ──────────────────────────────────────────────────────────────────


def sync_model_to_s3(
    result: DownloadResult,
    s3_cfg: S3Config,
    update_policy: str,
    force: bool,
    dry_run: bool,
) -> DownloadResult:
    """
    Optionally upload a successfully downloaded model to S3.
    Mutates result.s3_status and result.s3_error in-place; returns result.
    """
    if result.status == "ERROR" or result.local_path is None:
        result.s3_status = "SKIP"
        return result

    if dry_run:
        result.s3_status = "DRYRUN"
        return result

    if not result.local_path.is_file():
        result.s3_status = "ERROR"
        result.s3_error = f"Local file not found: {result.local_path}"
        return result

    s3_decision = s3_check_exists(
        model=result.model,
        local_file=result.local_path,
        cfg=s3_cfg,
        update_policy=update_policy,
        force=force,
    )

    if s3_decision == "SKIP":
        result.s3_status = "SKIP"
        tqdm.write(f"  [S3-SKIP] Already in s3://{s3_cfg.bucket}/{_s3_key(result.model, s3_cfg.prefix)}")
        return result

    if s3_decision == "ERROR":
        result.s3_status = "ERROR"
        result.s3_error = "Cannot reach S3 (check credentials / endpoint)"
        tqdm.write(f"  [S3-ERR]  {result.s3_error}")
        return result

    status, error = s3_upload_file(result.local_path, result.model, s3_cfg)
    result.s3_status = status
    result.s3_error = error
    if status == "UPLOADED":
        tqdm.write(f"  [S3-OK]   Uploaded to s3://{s3_cfg.bucket}/{_s3_key(result.model, s3_cfg.prefix)}")
    else:
        tqdm.write(f"  [S3-ERR]  Upload failed: {error}")

    return result


# ─── Filtering ────────────────────────────────────────────────────────────────


def apply_filters(
    models: list[ModelEntry],
    model_filter: Optional[str],
    tag_filter: Optional[str],
) -> list[ModelEntry]:
    """Return subset of models matching --model and/or --tag filters."""
    result = models
    if model_filter:
        needle = model_filter.lower()
        result = [m for m in result if needle in m.repo_id.lower() or needle in m.filename.lower()]
    if tag_filter:
        needle = tag_filter.lower()
        result = [m for m in result if any(needle in t.lower() for t in m.tags)]
    return result


# ─── Output Helpers ───────────────────────────────────────────────────────────


def print_model_list(models: list[ModelEntry]) -> None:
    """Print a formatted table of all models for --list."""
    rows = []
    for m in models:
        status = "enabled " if m.enabled else "disabled"
        gated_mark = " [gated]" if m.gated else ""
        repo_line = f"{m.repo_id}{gated_mark}\n  {m.filename}"
        tags = ", ".join(m.tags) if m.tags else "-"
        vram = f"{m.vram_gb:.0f} GB" if m.vram_gb else "-"
        desc = textwrap.shorten(m.description, 60)
        rows.append((status, repo_line, tags, vram, desc))
    print(f"\n{'STATUS':<9}  {'REPO_ID / FILENAME':<50}  {'TAGS':<30}  {'VRAM':<6}  DESCRIPTION")
    print("-" * 130)
    for status, repo, tags, vram, desc in rows:
        first_repo_line, *rest_repo_lines = repo.split("\n")
        print(f"{status:<9}  {first_repo_line:<50}  {tags:<30}  {vram:<6}  {desc}")
        for rl in rest_repo_lines:
            print(f"{'':9}  {rl}")
    print()


def print_results(results: list[DownloadResult], show_s3: bool = False) -> None:
    """Print a summary table after all downloads."""
    counts: dict[str, int] = {"DOWNLOAD": 0, "SKIP": 0, "ERROR": 0, "DRYRUN": 0, "DISABLED": 0}
    s3_counts: dict[str, int] = {"UPLOADED": 0, "SKIP": 0, "ERROR": 0}
    total_bytes = 0

    col_w = 110 + (20 if show_s3 else 0)
    print(f"\n{'STATUS':<10}  {'MODEL':<45}  {'SIZE':<10}  {'S3':<10}  DETAILS" if show_s3
          else f"\n{'STATUS':<10}  {'MODEL':<45}  {'SIZE':<10}  DETAILS")
    print("-" * col_w)

    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        total_bytes += r.size_bytes
        detail = r.error or (str(r.local_path) if r.local_path else "")
        label = r.model.repo_id.split("/")[-1] + "/" + r.model.filename
        label = textwrap.shorten(label, 45)
        s3_col = r.s3_status or "-"
        if show_s3:
            print(f"{r.status:<10}  {label:<45}  {fmt_size(r.size_bytes):<10}  {s3_col:<10}  {detail}")
        else:
            print(f"{r.status:<10}  {label:<45}  {fmt_size(r.size_bytes):<10}  {detail}")
        if r.s3_status:
            s3_counts[r.s3_status] = s3_counts.get(r.s3_status, 0) + 1

    print("-" * col_w)
    print(
        f"Total: {counts['DOWNLOAD']} downloaded, {counts['SKIP']} skipped,"
        f" {counts['ERROR']} errors, {counts['DISABLED']} disabled"
        f"  |  {fmt_size(total_bytes)} on disk"
    )
    if show_s3 and any(s3_counts.values()):
        print(
            f"S3:    {s3_counts.get('UPLOADED', 0)} uploaded,"
            f" {s3_counts.get('SKIP', 0)} skipped,"
            f" {s3_counts.get('ERROR', 0)} errors"
        )
    if counts["ERROR"]:
        print(f"\n[WARN] {counts['ERROR']} model(s) failed to download. See errors above.")
        for r in results:
            if r.status == "ERROR":
                print(f"  - {r.model.repo_id}/{r.model.filename}: {r.error}")
    if show_s3 and s3_counts.get("ERROR", 0):
        print(f"\n[WARN] {s3_counts['ERROR']} S3 upload(s) failed:")
        for r in results:
            if r.s3_status == "ERROR":
                print(f"  - {r.model.repo_id}/{r.model.filename}: {r.s3_error}")


# ─── Argument Parser ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_models.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="models.yaml",
        metavar="FILE",
        help="Path to models YAML config (default: models.yaml)",
    )
    p.add_argument(
        "--creds",
        default="credentials.yaml",
        metavar="FILE",
        help="Path to credentials YAML file (default: credentials.yaml)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate model list and repo availability without downloading",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-upload even if file already exists",
    )
    p.add_argument(
        "--model",
        metavar="SUBSTRING",
        help="Only process models whose repo_id or filename contains SUBSTRING",
    )
    p.add_argument(
        "--tag",
        metavar="TAG",
        help="Only process models that have TAG in their tags list",
    )
    p.add_argument(
        "--list",
        action="store_true",
        dest="list_models",
        help="Print all models in config and exit",
    )
    p.add_argument(
        "--include-disabled",
        action="store_true",
        help="Also process models with enabled: false",
    )
    # Retry / rate-limit options
    retry_group = p.add_argument_group("Retry и задержки")
    retry_group.add_argument(
        "--retries",
        type=int,
        default=None,
        metavar="N",
        help="Количество повторов при ошибке (по умолчанию из models.yaml, иначе 3)",
    )
    retry_group.add_argument(
        "--retry-delay",
        type=float,
        default=None,
        metavar="SECS",
        help="Базовая задержка между повторами в секундах (экспоненциальный backoff, default: 5)",
    )
    retry_group.add_argument(
        "--delay",
        type=float,
        default=None,
        metavar="SECS",
        help="Пауза между скачиваниями разных моделей в секундах (default: 0)",
    )
    # S3 options
    s3_group = p.add_argument_group("S3")
    s3_group.add_argument(
        "--upload-s3",
        action="store_true",
        help="Загрузить скачанные модели в S3 после скачивания",
    )
    s3_group.add_argument(
        "--s3-only",
        action="store_true",
        help="Только S3: модели скачиваются во временную директорию и удаляются после выгрузки",
    )
    return p


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    args = build_parser().parse_args()

    # Load .env first (lowest priority — credentials.yaml overrides these)
    load_dotenv(dotenv_path=".env", override=False)

    # Configure proxy (reads PROXY_ENABLED + PROXY_URL from env / credentials.yaml)
    proxy_cfg = load_proxy_config(Path(args.creds))
    if proxy_cfg.valid:
        proxy_cfg.apply_to_env()
        print(f"[INFO] Proxy enabled: {proxy_cfg.url}")
    else:
        print("[INFO] Proxy: disabled")

    # Load credentials from credentials.yaml (with .env as fallback)
    hf_token, s3_cfg = load_credentials(Path(args.creds))

    if hf_token:
        print(f"[INFO] HF_TOKEN loaded (length={len(hf_token)})")
    else:
        print("[INFO] No HF_TOKEN found — gated models will fail")

    # Enable fast transfers if requested
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        try:
            import hf_transfer  # noqa: F401
            print("[INFO] hf_transfer enabled (fast Rust-based downloads)")
        except ImportError:
            print("[WARN] HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer not installed; ignored")

    # Determine S3 mode
    use_s3 = args.upload_s3 or args.s3_only
    if use_s3 and not s3_cfg.valid:
        print(
            "[ERROR] S3 upload requested but credentials are incomplete.\n"
            "        Set access_key_id, secret_access_key, bucket in credentials.yaml\n"
            "        or AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET env vars.",
            file=sys.stderr,
        )
        return 1

    if use_s3:
        endpoint = s3_cfg.endpoint_url or "s3.amazonaws.com"
        print(f"[INFO] S3 target: s3://{s3_cfg.bucket}/{s3_cfg.prefix}  [{endpoint}]")

    # Load config
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 1

    settings, all_models = load_config(config_path)
    models_root = Path(settings.get("models_dir", "./models"))
    update_policy = settings.get("update_policy", "etag")

    # Retry / delay: CLI переопределяет значения из models.yaml
    retry_count: int = args.retries if args.retries is not None else int(settings.get("retry_count", 3))
    retry_delay: float = args.retry_delay if args.retry_delay is not None else float(settings.get("retry_delay", 5.0))
    inter_delay: float = args.delay if args.delay is not None else float(settings.get("inter_download_delay", 0.0))

    print(f"[INFO] Retry: {retry_count}x, base_delay={retry_delay}s, inter_delay={inter_delay}s")

    # models_dir from settings can be overridden by S3-only mode (use temp dir)
    _s3_tmpdir: Optional[str] = None
    if args.s3_only:
        _s3_tmpdir = tempfile.mkdtemp(prefix="ai_models_")
        models_root = Path(_s3_tmpdir)
        print(f"[INFO] S3-only mode: using temp dir {models_root}")

    print(f"[INFO] Config: {config_path}  |  Models dir: {models_root}  |  Policy: {update_policy}")

    # --list: show all and exit
    if args.list_models:
        print_model_list(all_models)
        return 0

    # Filter disabled (unless --include-disabled)
    candidates = all_models if args.include_disabled else [m for m in all_models if m.enabled]

    # Apply --model / --tag filters
    candidates = apply_filters(candidates, args.model, args.tag)

    if not candidates:
        print("[INFO] No models match the current filters.")
        return 0

    mode_label = "DRY-RUN" if args.dry_run else "DOWNLOAD"
    print(f"\n[{mode_label}] Processing {len(candidates)} model(s):\n")

    results: list[DownloadResult] = []

    try:
        for model in tqdm(candidates, desc="Models", unit="model", leave=True):
            repo_label = f"{model.repo_id}/{model.filename}"
            tqdm.write(f"[{model.repo_id.split('/')[-1]}]")

            if model.gated and not hf_token:
                tqdm.write(
                    f"  [SKIP/GATED] {repo_label}\n"
                    f"  Set HF_TOKEN in credentials.yaml or .env and accept the license at:\n"
                    f"  https://huggingface.co/{model.repo_id}"
                )
                results.append(
                    DownloadResult(
                        model=model,
                        status="ERROR",
                        error="Gated model: HF_TOKEN required",
                    )
                )
                continue

            result = download_model(
                model=model,
                models_root=models_root,
                token=hf_token,
                update_policy=update_policy,
                force=args.force,
                dry_run=args.dry_run,
                retry_count=retry_count,
                retry_delay=retry_delay,
            )

            if result.status == "SKIP":
                tqdm.write(f"  [SKIP] Already current: {result.local_path}")
            elif result.status == "DOWNLOAD":
                tqdm.write(f"  [OK]   Saved to: {result.local_path}  ({fmt_size(result.size_bytes)})")
            elif result.status == "DRYRUN":
                tqdm.write(f"  [OK]   Repository valid: {model.repo_id}")
            elif result.status == "ERROR":
                tqdm.write(f"  [ERR]  {result.error}")

            # S3 upload (if requested and download succeeded)
            if use_s3:
                result = sync_model_to_s3(
                    result=result,
                    s3_cfg=s3_cfg,
                    update_policy=update_policy,
                    force=args.force,
                    dry_run=args.dry_run,
                )

            # S3-only mode: remove local file after successful upload
            if args.s3_only and result.s3_status == "UPLOADED" and result.local_path:
                try:
                    result.local_path.unlink(missing_ok=True)
                except OSError:
                    pass

            results.append(result)

            # Пауза между скачиваниями (для соблюдения rate-limit квот)
            if inter_delay > 0 and model is not candidates[-1]:
                time.sleep(inter_delay)

    finally:
        # Гарантированная очистка temp dir даже при KeyboardInterrupt или исключении
        if _s3_tmpdir is not None:
            shutil.rmtree(_s3_tmpdir, ignore_errors=True)

    print_results(results, show_s3=use_s3)

    errors = sum(1 for r in results if r.status == "ERROR")
    s3_errors = sum(1 for r in results if r.s3_status == "ERROR")
    return 1 if (errors or s3_errors) else 0


if __name__ == "__main__":
    sys.exit(main())
