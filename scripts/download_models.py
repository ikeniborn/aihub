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
import fcntl
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from dotenv import load_dotenv

# Shared utilities (fmt_size, load_hf_token, load_proxy_config)
sys.path.insert(0, str(Path(__file__).parent))
from utils import fmt_size, load_hf_token, load_proxy_config, mask_proxy_url  # noqa: E402


# ─── Process Guard ────────────────────────────────────────────────────────────


_LOCK_FILE = Path(__file__).parent.parent / ".download.lock"


# ─── Hard Bandwidth Cap (tc ingress police) ────────────────────────────────────
#
# hf_xet has no byte-rate-limiting capability — only concurrency control.
# Even with NUM_CONCURRENT_RANGE_GETS=1, a single S3 range-get saturates a
# 100 Mbit link (30-50 MB blocks delivered at full line speed).
#
# IMPORTANT: downloads are *ingress* traffic.  tc tbf on "root" only shapes
# egress (outgoing ACKs) and does NOT limit received bytes.
#
# Correct approach: tc ingress qdisc + police filter.
# All incoming IP packets exceeding the rate are *dropped*; TCP congestion
# control reduces the send window and the source backs off within ~1-2 RTTs.
#
#   tc qdisc  add dev <iface> handle ffff: ingress
#   tc filter add dev <iface> parent ffff: protocol ip u32 match u32 0 0 \
#             police rate <N>mbit burst <B>mb drop flowid :1
#
# Burst = rate_mbps * 0.1 / 8 MB  ≈ 100 ms worth of data (minimum 1 MB).
# Cleanup is registered via atexit so the rule is removed on any exit path.


def _detect_default_iface() -> Optional[str]:
    """Return the network interface used for the default route (e.g. enp1s0)."""
    import subprocess as _sp
    try:
        out = _sp.run(["ip", "route", "show", "default"],
                      capture_output=True, text=True, timeout=5).stdout
        parts = out.split()
        if "dev" in parts:
            idx = parts.index("dev")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    except Exception:
        pass
    return None


def _tc_cmd(args: list, use_sudo: bool) -> bool:
    """Run a tc command, optionally prefixed with sudo. Returns True on success."""
    import shutil as _shutil
    import subprocess as _sp
    tc_bin = _shutil.which("tc") or "/usr/sbin/tc"
    prefix = ["sudo"] if use_sudo else []
    try:
        r = _sp.run(prefix + [tc_bin] + args,
                    capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


_tc_applied_iface: Optional[str] = None   # set when ingress rule is active
_tc_applied_sudo: Optional[bool] = None   # whether sudo was used


def apply_tc_bandwidth_limit(rate_mbps: float) -> bool:
    """
    Apply a hard *download* bandwidth cap via tc ingress policing.

    Uses `tc qdisc ingress` + `police` filter — the only tc mechanism that
    limits received (download) traffic.  Egress TBF is intentionally NOT used
    because it only shapes outgoing packets (ACKs) and has no effect on how
    fast the remote server sends data.

    Tries without sudo first, then with sudo (NOPASSWD supported).
    Registers atexit cleanup to remove the rule on any exit path.
    Returns True if the rule was applied successfully.
    """
    import atexit
    global _tc_applied_iface, _tc_applied_sudo

    iface = _detect_default_iface()
    if not iface:
        return False

    # ~100 ms burst bucket: allows short bursts to smooth TCP behaviour
    burst_mb = max(1, int(rate_mbps * 0.1 / 8))

    for use_sudo in (False, True):
        # Always clean up any pre-existing ingress (and legacy root) qdiscs
        _tc_cmd(["qdisc", "del", "dev", iface, "ingress"], use_sudo=use_sudo)
        _tc_cmd(["qdisc", "del", "dev", iface, "root"],    use_sudo=use_sudo)

        # Step 1: add ingress qdisc
        if not _tc_cmd(
            ["qdisc", "add", "dev", iface, "handle", "ffff:", "ingress"],
            use_sudo=use_sudo,
        ):
            continue

        # Step 2: add police filter — drop packets exceeding the rate
        ok = _tc_cmd([
            "filter", "add", "dev", iface,
            "parent", "ffff:",
            "protocol", "ip",
            "u32", "match", "u32", "0", "0",
            "police", "rate", f"{rate_mbps:.0f}mbit",
            "burst", f"{burst_mb}mb",
            "drop", "flowid", ":1",
        ], use_sudo=use_sudo)

        if ok:
            _tc_applied_iface = iface
            _tc_applied_sudo = use_sudo
            atexit.register(remove_tc_bandwidth_limit)
            return True

        # Filter failed — clean up the ingress qdisc we just added
        _tc_cmd(["qdisc", "del", "dev", iface, "ingress"], use_sudo=use_sudo)

    return False


def remove_tc_bandwidth_limit() -> None:
    """Remove the tc ingress rule applied by apply_tc_bandwidth_limit."""
    global _tc_applied_iface, _tc_applied_sudo
    iface = _tc_applied_iface
    if not iface:
        return
    use_sudo = _tc_applied_sudo or False
    _tc_cmd(["qdisc", "del", "dev", iface, "ingress"], use_sudo=use_sudo)
    _tc_applied_iface = None
    _tc_applied_sudo = None


def _acquire_lock() -> None:
    """
    Acquire an exclusive file lock to prevent multiple simultaneous instances.
    Registers atexit cleanup so the lock is released on any exit path
    (normal, exception, Ctrl+C, sys.exit) — no bandwidth zombies possible.
    """
    import atexit

    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = _LOCK_FILE.open("w")
    try:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        existing_pid = ""
        try:
            existing_pid = _LOCK_FILE.read_text().strip()
        except OSError:
            pass
        pid_info = f"PID {existing_pid}" if existing_pid else "unknown PID"
        print(
            f"[ERROR] Another download_models.py is already running ({pid_info}).\n"
            f"        If no process is running, delete the lock file:\n"
            f"        rm {_LOCK_FILE}",
            file=sys.stderr,
        )
        lock_fd.close()
        sys.exit(2)

    lock_fd.write(str(os.getpid()))
    lock_fd.flush()

    def _release() -> None:
        lock_fd.close()
        try:
            _LOCK_FILE.unlink(missing_ok=True)
        except OSError:
            pass

    atexit.register(_release)


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
    # Ollama support: source determines download backend
    # 'huggingface' (default, backward-compatible) or 'ollama'
    source: str = field(default="huggingface")
    # Ollama model tag (e.g. 'llama3.2:8b-instruct-q4_K_M')
    # Only used when source == 'ollama'
    ollama_model: str = field(default="")


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
                # Ollama fields — backward compatible via .get() defaults
                source=str(item.get("source", "huggingface") or "huggingface"),
                ollama_model=str(item.get("ollama_model", "") or ""),
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

    # ── Ollama: digest-based idempotency (replaces ETag for Ollama sources) ──
    if model.source == "ollama":
        if update_policy == "always":
            return "DOWNLOAD"
        # Read locally stored digest (written as .etag sidecar by ollama_hub)
        local_digest = _read_local_etag(local_file)
        if local_digest is None:
            # No sidecar yet — we cannot tell if file is current, default to SKIP
            print(
                f"  [WARN] No digest sidecar for Ollama model {model.ollama_model};"
                " defaulting to SKIP (use --force to re-download)"
            )
            return "SKIP"
        # Fetch remote digest from OCI manifest
        try:
            import ollama_hub
            info = ollama_hub.get_model_info(model.ollama_model)
            remote_digest = info.get("digest", "")
        except Exception as exc:
            print(
                f"  [WARN] Cannot fetch Ollama manifest for {model.ollama_model}: {exc};"
                " keeping existing file"
            )
            return "SKIP"
        if local_digest != remote_digest:
            print(f"  [INFO] Ollama digest changed — will re-download {model.filename}")
            return "DOWNLOAD"
        return "SKIP"

    if update_policy == "etag":
        local_etag = _read_local_etag(local_file)
        if local_etag is None:
            remote_etag = _fetch_remote_etag(model.repo_id, model.filename, token)
            if remote_etag is None:
                print(
                    f"  [WARN] Cannot determine ETag for {model.repo_id}/{model.filename};"
                    " defaulting to SKIP (use --force to override)"
                )
                return "SKIP"
            _write_local_etag(local_file, remote_etag)
            return "SKIP"

        remote_etag = _fetch_remote_etag(model.repo_id, model.filename, token)
        if remote_etag is None:
            print(
                f"  [WARN] Cannot fetch remote ETag for {model.repo_id}/{model.filename};"
                " keeping existing file"
            )
            return "SKIP"

        if local_etag != remote_etag:
            print(f"  [INFO] ETag changed — will re-download {model.filename}")
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
        print(
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


def _hf_download_with_timeout(timeout_secs: Optional[float], **kwargs) -> str:
    """
    Wrap hf_hub_download() in a daemon thread with a hard timeout.

    If the download stalls longer than timeout_secs, raises TimeoutError.
    The thread is marked daemon=True so it is killed automatically when the
    main process exits — preventing bandwidth zombies.
    """
    if timeout_secs is None or timeout_secs <= 0:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(**kwargs)

    result: list = [None]
    exc: list = [None]

    def _run() -> None:
        try:
            from huggingface_hub import hf_hub_download
            result[0] = hf_hub_download(**kwargs)
        except Exception as e:  # noqa: BLE001
            exc[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_secs)

    if t.is_alive():
        hours = timeout_secs / 3600
        raise TimeoutError(
            f"Download stalled for over {hours:.1f}h with no completion.\n"
            f"  Process will exit to free bandwidth. "
            f"Adjust download_timeout_hours in models.yaml or use --download-timeout."
        )
    if exc[0] is not None:
        raise exc[0]
    return result[0]  # type: ignore[return-value]


def download_model(
    model: ModelEntry,
    models_root: Path,
    token: Optional[str],
    update_policy: str,
    force: bool,
    dry_run: bool,
    retry_count: int = 3,
    retry_delay: float = 5.0,
    download_timeout: Optional[float] = None,
) -> DownloadResult:
    """
    Download a single model file.  Idempotent: checks ETag/digest before downloading.

    For source=='ollama': delegates to ollama_hub.download_model_gguf().
    For source=='huggingface' (default): uses hf_hub_download() (existing path).
    """
    local_dir = models_root / model.dest_dir
    local_file = local_dir / model.filename

    # ── Ollama source: delegate to ollama_hub ─────────────────────────────────
    if model.source == "ollama":
        if not model.ollama_model:
            return DownloadResult(
                model=model,
                status="ERROR",
                error="Ollama model entry is missing 'ollama_model' field",
            )

        if dry_run:
            # Validate: try to fetch manifest (lightweight HEAD equivalent)
            try:
                import ollama_hub
                ollama_hub.get_model_info(model.ollama_model)
                return DownloadResult(model=model, status="DRYRUN", local_path=local_file)
            except Exception as exc:
                return DownloadResult(
                    model=model,
                    status="ERROR",
                    error=f"Ollama manifest validation failed: {exc}",
                )

        decision = check_existing(local_file, model, update_policy, token, force)
        if decision == "SKIP":
            return DownloadResult(
                model=model,
                status="SKIP",
                local_path=local_file,
                size_bytes=local_file.stat().st_size if local_file.is_file() else 0,
            )

        # Delegate download to ollama_hub
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            import ollama_hub

            def _progress_cb(bytes_done: int, total_bytes: int) -> None:
                # Progress already emitted by ollama_hub via [FILEPROGRESS] markers
                pass

            downloaded = ollama_hub.download_model_gguf(
                model.ollama_model,
                dest_path=local_file,
                progress_callback=_progress_cb,
                retry_count=retry_count,
                retry_delay=retry_delay,
            )
            return DownloadResult(
                model=model,
                status="DOWNLOAD",
                local_path=downloaded,
                size_bytes=downloaded.stat().st_size if downloaded.is_file() else 0,
            )
        except Exception as exc:
            return DownloadResult(model=model, status="ERROR", error=str(exc))

    # ── HuggingFace source (default) ──────────────────────────────────────────
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

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

    # ── File-size probe for progress reporting ────────────────────────────────
    # hf_xet downloads entirely in Rust and emits no tqdm to stdout.
    # We get the expected size via HF metadata HEAD request, then poll the
    # .incomplete file in the local cache every 3 s and emit [FILEPROGRESS].
    _expected_bytes: Optional[int] = None
    _monitor_stop = threading.Event()
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url as _hf_hub_url
        _meta = get_hf_file_metadata(_hf_hub_url(model.repo_id, model.filename), token=token)
        _expected_bytes = getattr(_meta, "size", None)
        if _expected_bytes:
            print(f"[FILESIZE] {_expected_bytes}")
            sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass

    def _monitor_progress() -> None:
        _cache_dir = local_dir / ".cache" / "huggingface" / "download"
        while not _monitor_stop.wait(3.0):
            try:
                files = list(_cache_dir.glob("*.incomplete")) if _cache_dir.exists() else []
                if not files:
                    continue
                cur = max(f.stat().st_size for f in files if f.exists())
                tot = _expected_bytes or 0
                print(f"[FILEPROGRESS] {cur}/{tot}")
                sys.stdout.flush()
            except OSError:
                pass

    if _expected_bytes:
        _mt = threading.Thread(target=_monitor_progress, daemon=True)
        _mt.start()
    else:
        _mt = None

    last_exc: Optional[Exception] = None
    try:
        for attempt in range(retry_count + 1):
            try:
                if attempt > 0:
                    print(f"  [RETRY {attempt}/{retry_count}] {model.repo_id}/{model.filename}")
                else:
                    print(f"  Downloading {model.repo_id}/{model.filename} ...")

                downloaded_path = _hf_download_with_timeout(
                    download_timeout,
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

            except TimeoutError:
                raise  # propagate to main() — exit the entire process
            except GatedRepoError:
                # Доступ закрыт — токен есть, но аккаунт не в списке разрешённых.
                # Повторять бессмысленно: нужно запросить доступ вручную.
                return DownloadResult(
                    model=model,
                    status="ERROR",
                    error=(
                        f"Доступ запрещён (gated repo): {model.repo_id}\n"
                        f"  Действие: перейдите на https://huggingface.co/{model.repo_id}\n"
                        f"  и нажмите «Access repository» / «Request access», затем повторите загрузку."
                    ),
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
                is_gated = (
                    "403" in err_str
                    and ("authorized list" in err_str or "gated" in err_str.lower()
                         or "restricted" in err_str.lower())
                )

                # Gated 403 — повторять бессмысленно, нужно действие пользователя
                if is_gated:
                    return DownloadResult(
                        model=model,
                        status="ERROR",
                        error=(
                            f"Доступ запрещён (gated repo): {model.repo_id}\n"
                            f"  Действие: перейдите на https://huggingface.co/{model.repo_id}\n"
                            f"  и нажмите «Access repository» / «Request access», затем повторите загрузку."
                        ),
                    )

                if attempt >= retry_count:
                    break  # все попытки исчерпаны

                # Задержка перед повтором: rate limit — x10 от базовой, остальное — экспоненциально
                if is_rate_limit:
                    wait = retry_delay * 10
                    print(f"  [WARN] Rate limit (429) — ждём {wait:.0f}s перед повтором ...")
                else:
                    wait = retry_delay * (2 ** attempt)
                    print(f"  [WARN] Ошибка (попытка {attempt + 1}/{retry_count + 1}): {exc}")
                    print(f"         Повтор через {wait:.0f}s ...")
                time.sleep(wait)

        return DownloadResult(model=model, status="ERROR", error=str(last_exc))
    finally:
        _monitor_stop.set()


# ─── YAML Write Helper ────────────────────────────────────────────────────────


def _atomic_yaml_write(config_path: Path, data: dict) -> None:
    """Atomic YAML write via tempfile + os.replace (same pattern as model_browser)."""
    fd, tmp = tempfile.mkstemp(dir=str(config_path.parent), suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        os.replace(tmp, str(config_path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def mark_model_s3_synced(config_path: Path, model: "ModelEntry", s3_key: str) -> None:
    """
    Persist s3_synced=true and s3_key into the model's entry in models.yaml.

    s3_key is stored so that if dest_dir changes later, the old S3 path is known
    and the object can be renamed (copy + delete) instead of re-uploaded.
    """
    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for item in raw.get("models", []):
            if item.get("repo_id") == model.repo_id and item.get("filename") == model.filename:
                item["s3_synced"] = True
                item["s3_key"] = s3_key
                break
        _atomic_yaml_write(config_path, raw)
    except Exception as e:
        print(f"  [WARN] Could not update s3_synced in yaml: {e}")


def _s3_multipart_copy(
    client: Any,
    bucket: str,
    src_key: str,
    dst_key: str,
    obj_size: int,
    part_size: int = 512 * 1024 * 1024,  # 512 MiB
) -> None:
    """
    Copy an S3 object larger than 5 GiB using multipart copy.
    Aborts the multipart upload automatically on failure.
    Raises on error.
    """
    mpu = client.create_multipart_upload(Bucket=bucket, Key=dst_key)
    upload_id = mpu["UploadId"]
    parts: list[dict] = []
    try:
        part_num = 1
        offset = 0
        while offset < obj_size:
            end = min(offset + part_size - 1, obj_size - 1)
            resp = client.upload_part_copy(
                Bucket=bucket,
                Key=dst_key,
                UploadId=upload_id,
                PartNumber=part_num,
                CopySource={"Bucket": bucket, "Key": src_key},
                CopySourceRange=f"bytes={offset}-{end}",
            )
            parts.append({"PartNumber": part_num, "ETag": resp["CopyPartResult"]["ETag"]})
            print(f"  [RELOCATE-S3]  Part {part_num}: bytes {offset}–{end}")
            part_num += 1
            offset = end + 1
        client.complete_multipart_upload(
            Bucket=bucket,
            Key=dst_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
    except Exception:
        client.abort_multipart_upload(Bucket=bucket, Key=dst_key, UploadId=upload_id)
        raise


def relocate_model_if_moved(
    config_path: Path,
    model: "ModelEntry",
    models_root: Path,
    s3_cfg: Optional["S3Config"] = None,
) -> bool:
    """
    Detect whether dest_dir changed since the last S3 sync and fix all locations.

    Compares the stored s3_key (written by mark_model_s3_synced) with the key
    derived from the current dest_dir.  If they differ:

      1. Local move  — file is moved from old path to new path (including .etag).
      2. S3 rename   — object is copied to new key then old key is deleted
                       (only when s3_cfg is provided and valid).
      3. yaml update — s3_key is rewritten to the new key so the flag stays
                       consistent for future runs.

    Returns True if any relocation was performed.
    """
    # ── Read stored s3_key ────────────────────────────────────────────────────
    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  [WARN] relocate: cannot read yaml: {e}")
        return False

    stored_s3_key: str = ""
    for item in raw.get("models", []):
        if item.get("repo_id") == model.repo_id and item.get("filename") == model.filename:
            stored_s3_key = str(item.get("s3_key") or "")
            break

    if not stored_s3_key:
        return False  # Never synced to S3 — nothing to relocate

    # ── Compare with expected key from current dest_dir ───────────────────────
    s3_prefix = s3_cfg.prefix if s3_cfg else ""
    expected_key = _s3_key(model, s3_prefix)
    if stored_s3_key == expected_key:
        return False  # dest_dir unchanged — nothing to do

    # ── Derive old dest_dir from stored_s3_key ────────────────────────────────
    prefix_stripped = s3_prefix.rstrip("/")
    rel = stored_s3_key[len(prefix_stripped) + 1:] if prefix_stripped else stored_s3_key
    # rel == "old_dest_dir/filename"
    if not rel.endswith(model.filename):
        print(f"  [WARN] relocate: stored s3_key has unexpected filename: {stored_s3_key!r}")
        return False
    old_dest_dir = rel[: -(len(model.filename) + 1)]  # strip "/filename"

    old_local = models_root / old_dest_dir / model.filename
    new_local = models_root / model.dest_dir / model.filename

    print(f"  [RELOCATE] dest_dir changed: {old_dest_dir!r} → {model.dest_dir!r}")

    # ── 1. Move local file ────────────────────────────────────────────────────
    local_ok = False
    if new_local.is_file():
        print(f"  [RELOCATE-LOCAL] Already at new path: {new_local}")
        # Remove stale duplicate at old path to avoid wasted disk space
        if old_local != new_local and old_local.is_file():
            try:
                old_local.unlink()
                print(f"  [RELOCATE-LOCAL] Removed stale old file: {old_local}")
            except Exception as e:
                print(f"  [WARN] relocate: cannot remove old file: {e}")
            old_etag = _etag_path(old_local)
            if old_etag.is_file():
                try:
                    old_etag.unlink()
                except Exception:
                    pass
        local_ok = True
    elif old_local.is_file():
        try:
            new_local.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_local), str(new_local))
            print(f"  [RELOCATE-LOCAL] Moved:\n"
                  f"    {old_local}\n"
                  f"  → {new_local}")
            # Move .etag sidecar if present
            old_etag = _etag_path(old_local)
            if old_etag.is_file():
                shutil.move(str(old_etag), str(_etag_path(new_local)))
            local_ok = True
        except Exception as e:
            print(f"  [WARN] relocate: local move failed: {e}")
    else:
        print(f"  [WARN] relocate: old local file not found at {old_local}, skipping local move")

    # ── 1b. Move HF cache dir (needed for resumable downloads) ────────────────
    old_cache = models_root / old_dest_dir / ".cache"
    new_cache = new_local.parent / ".cache"
    if old_cache.is_dir() and old_cache.resolve() != new_cache.resolve():
        if new_cache.exists():
            print(f"  [RELOCATE-LOCAL] Cache already exists at new path, skipping cache move")
        else:
            try:
                new_cache.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_cache), str(new_cache))
                print(f"  [RELOCATE-LOCAL] Moved cache dir:\n"
                      f"    {old_cache}\n"
                      f"  → {new_cache}")
            except Exception as e:
                print(f"  [WARN] relocate: cache dir move failed: {e}")

    # ── 1c. Remove old directory if now empty ─────────────────────────────────
    old_dir = old_local.parent
    if old_dir.is_dir() and old_dir.resolve() != new_local.parent.resolve():
        try:
            old_dir.rmdir()  # Succeeds only when empty; OSError otherwise
            print(f"  [RELOCATE-LOCAL] Removed empty old dir: {old_dir}")
        except OSError:
            pass  # Directory not empty — leave it

    # ── 2. Rename S3 object (copy + delete) ──────────────────────────────────
    # CopyObject is limited to 5 GiB; _s3_multipart_copy handles larger objects.
    _S3_COPY_SIMPLE_LIMIT = 5 * 1024 ** 3  # 5 GiB

    s3_ok = False
    if s3_cfg and s3_cfg.valid:
        try:
            client = _s3_client(s3_cfg)
            head = client.head_object(Bucket=s3_cfg.bucket, Key=stored_s3_key)
            obj_size = head["ContentLength"]

            if obj_size <= _S3_COPY_SIMPLE_LIMIT:
                client.copy_object(
                    CopySource={"Bucket": s3_cfg.bucket, "Key": stored_s3_key},
                    Bucket=s3_cfg.bucket,
                    Key=expected_key,
                )
            else:
                print(f"  [RELOCATE-S3]  Large object ({obj_size / 1024**3:.1f} GiB) — using multipart copy")
                _s3_multipart_copy(client, s3_cfg.bucket, stored_s3_key, expected_key, obj_size)

            print(f"  [RELOCATE-S3]  Copied:\n"
                  f"    s3://{s3_cfg.bucket}/{stored_s3_key}\n"
                  f"  → s3://{s3_cfg.bucket}/{expected_key}")
            client.delete_object(Bucket=s3_cfg.bucket, Key=stored_s3_key)
            print(f"  [RELOCATE-S3]  Deleted old key: {stored_s3_key}")
            s3_ok = True
        except Exception as e:
            print(f"  [WARN] relocate: S3 rename failed: {e}\n"
                  f"         Old key preserved: s3://{s3_cfg.bucket}/{stored_s3_key}")

    # ── 3. Update s3_key in yaml ──────────────────────────────────────────────
    if local_ok or s3_ok:
        try:
            with config_path.open("r", encoding="utf-8") as f:
                raw2 = yaml.safe_load(f) or {}
            for item in raw2.get("models", []):
                if item.get("repo_id") == model.repo_id and item.get("filename") == model.filename:
                    item["s3_key"] = expected_key if s3_ok else stored_s3_key
                    break
            _atomic_yaml_write(config_path, raw2)
        except Exception as e:
            print(f"  [WARN] relocate: yaml s3_key update failed: {e}")

    return local_ok or s3_ok


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
        print(f"  [S3-SKIP] Already in s3://{s3_cfg.bucket}/{_s3_key(result.model, s3_cfg.prefix)}")
        return result

    if s3_decision == "ERROR":
        result.s3_status = "ERROR"
        result.s3_error = "Cannot reach S3 (check credentials / endpoint)"
        print(f"  [S3-ERR]  {result.s3_error}")
        return result

    status, error = s3_upload_file(result.local_path, result.model, s3_cfg)
    result.s3_status = status
    result.s3_error = error
    if status == "UPLOADED":
        print(f"  [S3-OK]   Uploaded to s3://{s3_cfg.bucket}/{_s3_key(result.model, s3_cfg.prefix)}")
    else:
        print(f"  [S3-ERR]  Upload failed: {error}")

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
    # Bandwidth / concurrency throttling
    bw_group = p.add_argument_group("Bandwidth и параллельность")
    bw_group.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Максимальное число параллельных соединений при загрузке. "
            "Для hf-xet (default backend, 2025+): устанавливает HF_XET_NUM_CONCURRENT_RANGE_GETS "
            "(default=16 диапазонных запросов на файл). "
            "Для standard HTTP / hf_transfer (legacy): устанавливает HF_HUB_DOWNLOAD_CONCURRENCY. "
            "По умолчанию из models.yaml (hf_download_concurrency), иначе не ограничено. "
            "Рекомендуется 2-4 для снижения нагрузки на канал."
        ),
    )
    bw_group.add_argument(
        "--bandwidth-limit",
        type=float,
        default=None,
        metavar="MBIT",
        help=(
            "Мягкое ограничение скорости загрузки в Mbit/s через авто-расчёт параллельности "
            "(concurrency = bandwidth_mbps / 25, минимум 1). "
            "Не является аппаратным ограничением — для жёсткого лимита используйте "
            "tc/wondershaper/trickle на уровне ОС. "
            "По умолчанию из models.yaml (bandwidth_limit_mbps). "
            "Не учитывается если --max-concurrency задан явно."
        ),
    )
    bw_group.add_argument(
        "--download-timeout",
        type=float,
        default=None,
        metavar="HOURS",
        help=(
            "Максимальное время загрузки одного файла в часах. Если загрузка зависла "
            "дольше этого времени — процесс аварийно завершается (kill zombie). "
            "По умолчанию из models.yaml (download_timeout_hours). 0 = без ограничения."
        ),
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

    # ── Early concurrency + segment-size cap ──────────────────────────────────
    # HF_XET_NUM_CONCURRENT_RANGE_GETS limits the number of simultaneous
    # HTTP range-GET *requests* at the outer level, but each request fetches a
    # *segment* that internally contains HF_XET_NUM_RANGE_IN_SEGMENT_BASE chunks.
    # With default RANGE_IN_SEGMENT_BASE≈16 and RANGE_GETS=2 you still get
    # 2×16=32 simultaneous chunk fetches → network saturation.
    # The only way to actually cap bandwidth via hf-xet env vars is to set BOTH:
    #   HF_XET_NUM_CONCURRENT_RANGE_GETS — outer request concurrency
    #   HF_XET_NUM_RANGE_IN_SEGMENT_BASE/MAX/DELTA — inner segment size
    #
    # Rust lazy statics (once_cell) read env vars on first use of the limiter,
    # not at import time — so setting them before the first download_files() call
    # is sufficient. We set them here early anyway (before import hf_xet in the
    # backend detection block) as an extra safety measure.
    _early_settings: dict = {}
    _early_config = Path(args.config)
    if _early_config.is_file():
        try:
            with open(_early_config) as _f:
                _raw_cfg = yaml.safe_load(_f) or {}
            _early_settings = _raw_cfg.get("settings", {})
        except Exception:
            pass

    _pre_yaml_concurrency = _early_settings.get("hf_download_concurrency")
    _pre_concurrency: Optional[int] = (
        args.max_concurrency if args.max_concurrency is not None
        else (int(_pre_yaml_concurrency) if _pre_yaml_concurrency is not None else None)
    )
    _pre_yaml_bw = _early_settings.get("bandwidth_limit_mbps")
    _pre_bw: Optional[float] = (
        args.bandwidth_limit if args.bandwidth_limit is not None
        else (float(_pre_yaml_bw) if _pre_yaml_bw is not None else None)
    )
    if _pre_bw is not None and _pre_concurrency is None:
        _pre_concurrency = max(1, int(_pre_bw / 25))

    if _pre_concurrency is not None:
        _pre_conc_str = str(_pre_concurrency)
        # HF_XET_NUM_CONCURRENT_RANGE_GETS — only confirmed hf-xet concurrency var
        os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = _pre_conc_str
        # HF_HUB_DOWNLOAD_CONCURRENCY — fallback for standard HTTP backend
        os.environ["HF_HUB_DOWNLOAD_CONCURRENCY"] = _pre_conc_str
    # ─────────────────────────────────────────────────────────────────────────

    # Prevent multiple simultaneous instances — one download process at a time
    _acquire_lock()

    # Configure proxy (reads PROXY_ENABLED + PROXY_URL from env / credentials.yaml)
    proxy_cfg = load_proxy_config(Path(args.creds))
    if proxy_cfg.valid:
        proxy_cfg.apply_to_env()
        print(f"[INFO] Proxy enabled: {mask_proxy_url(proxy_cfg.url)}")
    else:
        print("[INFO] Proxy: disabled")

    # Load credentials from credentials.yaml (with .env as fallback)
    hf_token, s3_cfg = load_credentials(Path(args.creds))

    if hf_token:
        print(f"[INFO] HF_TOKEN loaded (length={len(hf_token)})")
    else:
        print("[INFO] No HF_TOKEN found — gated models will fail")

    # Detect active download backend and log the real protocol in use.
    # hf-xet (Xet storage backend) is now the default for HuggingFace Hub (2025+).
    # hf_transfer (legacy Rust backend) is deprecated and has no effect when hf-xet is installed.
    # HF_HUB_ENABLE_HF_TRANSFER is deprecated; use HF_XET_HIGH_PERFORMANCE for maximum speed.
    _xet_disabled = os.environ.get("HF_HUB_DISABLE_XET", "").strip().lower() in ("1", "true", "yes", "on")
    _hf_transfer_requested = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "") == "1"
    _active_backend = "standard"

    if not _xet_disabled:
        try:
            import hf_xet  # noqa: F401
            _active_backend = "xet"
            print("[INFO] Download backend: hf-xet (Xet storage backend, chunk-based deduplication)")
        except ImportError:
            _xet_disabled = True  # hf-xet not installed, treat as disabled

    if _xet_disabled or _active_backend == "standard":
        if _hf_transfer_requested:
            try:
                import hf_transfer  # noqa: F401
                _active_backend = "hf_transfer"
                print("[INFO] Download backend: hf_transfer (legacy Rust-based, HF_HUB_ENABLE_HF_TRANSFER=1)")
            except ImportError:
                print("[WARN] HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer not installed; using standard HTTP")
        else:
            print("[INFO] Download backend: standard HTTP (huggingface_hub)")

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

    # Per-file download timeout: kills the process if a download stalls (ghost protection).
    _yaml_dt = settings.get("download_timeout_hours")
    _cli_dt = args.download_timeout
    _timeout_hours: Optional[float] = (
        _cli_dt if _cli_dt is not None
        else (float(_yaml_dt) if _yaml_dt is not None else None)
    )
    download_timeout_secs: Optional[float] = (
        _timeout_hours * 3600 if (_timeout_hours is not None and _timeout_hours > 0) else None
    )
    if download_timeout_secs:
        print(f"[INFO] Download timeout: {_timeout_hours:.1f}h per file (ghost process protection ON)")
    else:
        print("[INFO] Download timeout: disabled (set download_timeout_hours in models.yaml)")

    # Bandwidth throttling.
    #
    # hf_xet has NO byte-rate-limiting capability — only concurrency control.
    # Even with NUM_CONCURRENT_RANGE_GETS=1, a single S3 range-get delivers
    # a ~30-50 MB block at full link speed → saturates a 100 Mbit channel.
    #
    # Hard cap strategy: when bandwidth_limit_mbps is set, apply tc tbf (Token
    # Bucket Filter) on the default network interface via apply_tc_bandwidth_limit().
    # tc is removed on exit via atexit (registered inside apply_tc_bandwidth_limit).
    #
    # Concurrency env vars are still set for cosmetic reduction of parallelism:
    #   hf-xet: HF_XET_NUM_CONCURRENT_RANGE_GETS / HF_XET_NUM_RANGE_IN_SEGMENT_*
    #   standard HTTP: HF_HUB_DOWNLOAD_CONCURRENCY
    _yaml_concurrency = settings.get("hf_download_concurrency")
    max_concurrency: Optional[int] = (
        args.max_concurrency if args.max_concurrency is not None
        else (int(_yaml_concurrency) if _yaml_concurrency is not None else None)
    )

    _yaml_bw = settings.get("bandwidth_limit_mbps")
    bandwidth_limit_mbps: Optional[float] = (
        args.bandwidth_limit if args.bandwidth_limit is not None
        else (float(_yaml_bw) if _yaml_bw is not None else None)
    )

    # Hard bandwidth cap via tc ingress policing
    if bandwidth_limit_mbps is not None:
        _tc_ok = apply_tc_bandwidth_limit(bandwidth_limit_mbps)
        if _tc_ok:
            _iface = _tc_applied_iface or "?"
            print(
                f"[INFO] Bandwidth cap: {bandwidth_limit_mbps:.0f} Mbit/s "
                f"(tc ingress police on {_iface} — HARD cap, removed on exit)"
            )
        else:
            print(
                f"[WARN] bandwidth_limit_mbps={bandwidth_limit_mbps:.0f} requested but tc failed "
                f"(need sudo or root). Running unlimited. "
                f"To enable: sudo visudo → add: {os.getenv('USER','user')} ALL=(ALL) NOPASSWD: /usr/sbin/tc"
            )

    # Concurrency env vars (reduce parallelism; not a bandwidth cap for hf-xet)
    _concurrency_auto_derived = False
    if bandwidth_limit_mbps is not None and max_concurrency is None:
        max_concurrency = max(1, int(bandwidth_limit_mbps / 25))
        _concurrency_auto_derived = True

    if max_concurrency is not None:
        _src = "auto-derived from bandwidth_limit_mbps" if _concurrency_auto_derived else "config/CLI"
        _conc_str = str(max_concurrency)
        if _active_backend == "xet":
            # Only HF_XET_NUM_CONCURRENT_RANGE_GETS is a confirmed hf-xet env var
            os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = _conc_str
            os.environ["HF_HUB_DOWNLOAD_CONCURRENCY"] = _conc_str
            print(f"[INFO] Xet concurrency: HF_XET_NUM_CONCURRENT_RANGE_GETS={max_concurrency} [{_src}]")
        else:
            os.environ["HF_HUB_DOWNLOAD_CONCURRENCY"] = _conc_str
            print(f"[INFO] HF_HUB_DOWNLOAD_CONCURRENCY={max_concurrency} [{_src}]")
    else:
        print("[INFO] Concurrency: not limited (using backend defaults)")

    if bandwidth_limit_mbps is None:
        print("[INFO] Bandwidth limit: not set (unlimited)")

    # models_dir from settings can be overridden by S3-only mode (use temp dir)
    _s3_tmpdir: Optional[str] = None
    if args.s3_only:
        _s3_tmpdir = tempfile.mkdtemp(prefix="ai_models_")
        models_root = Path(_s3_tmpdir)
        print(f"[INFO] S3-only mode: using temp dir {models_root}")

    print(f"[INFO] Config: {config_path}  |  Models dir: {models_root}  |  Policy: {update_policy}")

    # Early write-permission check (skip for s3-only: temp dir is always writable)
    if not args.s3_only:
        _check_dir = models_root
        while not _check_dir.exists():
            _check_dir = _check_dir.parent
        if not os.access(_check_dir, os.W_OK):
            _msg = (
                f"[ERROR] No write permission for models_dir: {models_root}\n"
                f"        Checked: {_check_dir}\n"
                f"        Fix: sudo chown -R $(whoami) {_check_dir}"
            )
            if args.dry_run:
                print(_msg.replace("[ERROR]", "[WARN] (dry-run)"))
            else:
                print(_msg, file=sys.stderr)
                return 1

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
    _total = len(candidates)
    print(f"\n[{mode_label}] Processing {_total} model(s):\n")

    # Keep tqdm progress bars enabled — they will be parsed by model_browser.py
    # to extract per-file download percentage from lines like "52%|..." or
    # "[FILE_PROGRESS] 52". Carriage-return refresh lines are stripped in the worker.
    # Do NOT set HF_HUB_DISABLE_PROGRESS_BARS here.

    results: list[DownloadResult] = []

    try:
        for _idx, model in enumerate(candidates, 1):
            repo_label = f"{model.repo_id}/{model.filename}"
            # Structured progress marker parsed by model_browser.py
            print(f"[PROGRESS] {_idx}/{_total}")
            print(f"[{model.repo_id.split('/')[-1]}]")

            if model.gated and not hf_token:
                print(
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

            # Relocate if dest_dir changed since the last S3 sync.
            # Must run before download_model so the local file is found at the
            # correct path and is not re-downloaded unnecessarily.
            if not args.dry_run and not args.s3_only:
                relocate_model_if_moved(
                    config_path=config_path,
                    model=model,
                    models_root=models_root,
                    s3_cfg=s3_cfg,
                )

            try:
                result = download_model(
                    model=model,
                    models_root=models_root,
                    token=hf_token,
                    update_policy=update_policy,
                    force=args.force,
                    dry_run=args.dry_run,
                    retry_count=retry_count,
                    retry_delay=retry_delay,
                    download_timeout=download_timeout_secs,
                )
            except TimeoutError as te:
                print(f"\n[FATAL] {te}", file=sys.stderr)
                print(
                    "[FATAL] Aborting all downloads to prevent bandwidth zombie.\n"
                    "        Fix the network issue, then restart the script.",
                    file=sys.stderr,
                )
                return 3

            if result.status == "SKIP":
                print(f"  [SKIP] Already current: {result.local_path}")
            elif result.status == "DOWNLOAD":
                print(f"  [OK]   Saved to: {result.local_path}  ({fmt_size(result.size_bytes)})")
            elif result.status == "DRYRUN":
                print(f"  [OK]   Repository valid: {model.repo_id}")
            elif result.status == "ERROR":
                print(f"  [ERR]  {result.error}")

            # S3 upload (if requested and download succeeded)
            if use_s3:
                result = sync_model_to_s3(
                    result=result,
                    s3_cfg=s3_cfg,
                    update_policy=update_policy,
                    force=args.force,
                    dry_run=args.dry_run,
                )
                # Persist s3_synced flag + s3_key into models.yaml so the UI can
                # show the S3 badge and future dest_dir moves can rename the object.
                if result.s3_status in ("UPLOADED", "SKIP") and not args.dry_run:
                    mark_model_s3_synced(config_path, model, _s3_key(model, s3_cfg.prefix))

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
