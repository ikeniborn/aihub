#!/usr/bin/env python3
"""
AI Model Downloader
===================
Downloads AI models from HuggingFace Hub according to models.yaml.

Features:
- Idempotent: skips already-downloaded files (ETag-based update detection)
- Atomic downloads: no partial files left on failure
- Gated model support: reads HF_TOKEN from .env
- Fast downloads: optional hf_transfer (Rust-based, set HF_HUB_ENABLE_HF_TRANSFER=1)
- Dry-run mode: validates model list without downloading
- Filter by tag or model name

Usage:
    python scripts/download_models.py                  # Download all enabled models
    python scripts/download_models.py --dry-run        # Validate only, no downloads
    python scripts/download_models.py --force          # Re-download even if file exists
    python scripts/download_models.py --model Phi-4    # Filter by repo_id substring
    python scripts/download_models.py --tag russian    # Filter by tag
    python scripts/download_models.py --list           # List all models and exit
    python scripts/download_models.py --config custom.yaml  # Use alternate config
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


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
class DownloadResult:
    """Result of a single model download attempt."""

    model: ModelEntry
    status: str  # SKIP | DOWNLOAD | ERROR | DRYRUN | DISABLED
    local_path: Optional[Path] = None
    error: Optional[str] = None
    size_bytes: int = 0


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
            # No stored ETag — check file size as fallback
            remote_etag = _fetch_remote_etag(model.repo_id, model.filename, token)
            if remote_etag is None:
                # Cannot determine freshness — be conservative and skip
                print(
                    f"  [WARN] Cannot determine ETag for {model.repo_id}/{model.filename};"
                    " defaulting to SKIP (use --force to override)"
                )
                return "SKIP"
            # Store the ETag now and skip (file was downloaded without ETag tracking)
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


# ─── Download ─────────────────────────────────────────────────────────────────


def download_model(
    model: ModelEntry,
    models_root: Path,
    token: Optional[str],
    update_policy: str,
    force: bool,
    dry_run: bool,
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

    # ── Perform download ──────────────────────────────────────────────────────
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  Downloading {model.repo_id}/{model.filename} ...")
        downloaded_path = hf_hub_download(
            repo_id=model.repo_id,
            filename=model.filename,
            local_dir=str(local_dir),
            token=token,
        )
        downloaded = Path(downloaded_path)

        # Verify the file landed where expected (hf_hub_download may create subdir cache)
        # Resolve to the actual file regardless of HF cache layout
        if not downloaded.is_file():
            raise FileNotFoundError(f"Expected file not found after download: {downloaded}")

        size = downloaded.stat().st_size
        if size == 0:
            raise ValueError("Downloaded file is empty — possible network error")

        # If HF placed the file in a cache subdir, copy it to flat dest_dir
        if downloaded.parent != local_dir:
            import shutil
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
        return DownloadResult(
            model=model,
            status="ERROR",
            error=(
                f"File '{model.filename}' not found in repo '{model.repo_id}'.\n"
                f"  Check the filename at: https://huggingface.co/{model.repo_id}"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return DownloadResult(model=model, status="ERROR", error=str(exc))


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


def _fmt_size(n: int) -> str:
    if n == 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def print_model_list(models: list[ModelEntry]) -> None:
    """Print a formatted table of all models for --list."""
    cols = ("STATUS", "REPO_ID / FILENAME", "TAGS", "VRAM", "DESCRIPTION")
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


def print_results(results: list[DownloadResult]) -> None:
    """Print a summary table after all downloads."""
    counts = {"DOWNLOAD": 0, "SKIP": 0, "ERROR": 0, "DRYRUN": 0, "DISABLED": 0}
    total_bytes = 0
    print(f"\n{'STATUS':<10}  {'MODEL':<45}  {'SIZE':<10}  DETAILS")
    print("-" * 110)
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        total_bytes += r.size_bytes
        detail = r.error or (str(r.local_path) if r.local_path else "")
        label = r.model.repo_id.split("/")[-1] + "/" + r.model.filename
        label = textwrap.shorten(label, 45)
        print(f"{r.status:<10}  {label:<45}  {_fmt_size(r.size_bytes):<10}  {detail}")
    print("-" * 110)
    print(
        f"Total: {counts['DOWNLOAD']} downloaded, {counts['SKIP']} skipped,"
        f" {counts['ERROR']} errors, {counts['DISABLED']} disabled"
        f"  |  {_fmt_size(total_bytes)} on disk"
    )
    if counts["ERROR"]:
        print(f"\n[WARN] {counts['ERROR']} model(s) failed to download. See errors above.")
        for r in results:
            if r.status == "ERROR":
                print(f"  - {r.model.repo_id}/{r.model.filename}: {r.error}")


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
        "--dry-run",
        action="store_true",
        help="Validate model list and repo availability without downloading",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists (ignores ETag / skip policy)",
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
    return p


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    args = build_parser().parse_args()

    # Load credentials from .env (silently OK if file absent)
    load_dotenv(dotenv_path=".env", override=False)

    hf_token: Optional[str] = os.environ.get("HF_TOKEN") or None
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

    # Load config
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 1

    settings, all_models = load_config(config_path)
    models_root = Path(settings.get("models_dir", "./models"))
    update_policy = settings.get("update_policy", "etag")

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

    for model in candidates:
        repo_label = f"{model.repo_id}/{model.filename}"
        print(f"[{model.repo_id.split('/')[-1]}]")

        if model.gated and not hf_token:
            print(
                f"  [SKIP/GATED] {repo_label}\n"
                f"  Set HF_TOKEN in .env and accept the license at:\n"
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
        )
        results.append(result)

        if result.status == "SKIP":
            print(f"  [SKIP] Already current: {result.local_path}")
        elif result.status == "DOWNLOAD":
            print(f"  [OK]   Saved to: {result.local_path}  ({_fmt_size(result.size_bytes)})")
        elif result.status == "DRYRUN":
            print(f"  [OK]   Repository valid: {model.repo_id}")
        elif result.status == "ERROR":
            print(f"  [ERR]  {result.error}")

    print_results(results)

    errors = sum(1 for r in results if r.status == "ERROR")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
