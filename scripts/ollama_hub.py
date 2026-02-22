"""
Ollama Hub — OCI Registry Client + Model Search
================================================
Provides search and download functionality for Ollama models via:
  - OCI registry at registry.ollama.ai (anonymous access for public models)
  - Search via ollama.com/search HTML scraping with /api/tags fallback

The Ollama OCI registry follows the standard OCI Distribution Spec:
  GET registry.ollama.ai/v2/library/{model}/manifests/{tag}
    → OCI manifest JSON with layers
  Find layer with mediaType 'application/vnd.ollama.image.model'
    → blob URL containing the GGUF file

Public functions:
  search_models(query, limit)       → list[dict] with model_tag/description/pulls/params/quantization
  get_model_info(model_tag)         → dict with model_tag/digest/size
  download_model_gguf(model_tag, dest_path, progress_callback)  → Path

File naming:
  model_tag 'llama3.2:8b-instruct-q4_K_M'  → 'llama3.2-8b-instruct-q4_K_M.gguf'
  model_tag 'user/model:tag'               → 'model-tag.gguf'  (last path component)

Idempotency:
  SHA256 digest from OCI manifest stored as .etag sidecar (same pattern as HF ETag).
  Download is skipped if the sidecar digest matches the remote manifest digest.

Progress markers (compatible with model_browser.py worker parser):
  [FILESIZE] {bytes}       — emitted once if Content-Length is available
  [FILEPROGRESS] {cur}/{total}  — emitted during streaming download
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

# Shared utilities (fmt_size, ProxyConfig, load_proxy_config)
sys.path.insert(0, str(Path(__file__).parent))
from utils import fmt_size, ProxyConfig, load_proxy_config  # noqa: E402

__all__ = ["search_models", "get_model_info", "download_model_gguf"]

# ─── Constants ────────────────────────────────────────────────────────────────

_REGISTRY_BASE = "https://registry.ollama.ai/v2"
_SEARCH_URL = "https://ollama.com/search"
_TAGS_URL = "https://ollama.com/api/tags"
_OLLAMA_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"

# Retry settings matching download_models.py pattern
_DEFAULT_RETRY_COUNT = 3
_DEFAULT_RETRY_DELAY = 5.0

# Validation regex for ollama model tags: model[:tag] or user/model[:tag]
_OLLAMA_TAG_RE = re.compile(r"^[\w.\-]+(\/[\w.\-]+)?(:([\w.\-]+))?$")

# Chunk size for streaming download (512 KB)
_CHUNK_SIZE = 512 * 1024


# ─── Proxy-aware requests session ─────────────────────────────────────────────


def _get_session():
    """
    Create a requests.Session with proxy settings applied if configured.
    Proxy is loaded from PROXY_ENABLED/PROXY_URL env vars or credentials.yaml.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required for ollama_hub. Run: pip install requests")

    proxy_cfg: ProxyConfig = load_proxy_config()
    session = requests.Session()
    if proxy_cfg.valid:
        session.proxies = {"http": proxy_cfg.url, "https": proxy_cfg.url}
    # User-Agent matching standard OCI clients
    session.headers.update({"User-Agent": "aihub-ollama-client/1.0"})
    return session


# ─── Model tag parsing ────────────────────────────────────────────────────────


def _parse_model_tag(model_tag: str) -> tuple[str, str, str]:
    """
    Parse a model_tag into (namespace, model, tag).

    Examples:
      'llama3.2'                    → ('library', 'llama3.2', 'latest')
      'llama3.2:8b-instruct-q4_K_M' → ('library', 'llama3.2', '8b-instruct-q4_K_M')
      'user/model:tag'              → ('user', 'model', 'tag')
    """
    if "/" in model_tag:
        namespace, rest = model_tag.split("/", 1)
    else:
        namespace = "library"
        rest = model_tag

    if ":" in rest:
        model, tag = rest.split(":", 1)
    else:
        model, tag = rest, "latest"

    return namespace, model, tag


def _safe_filename(model_tag: str) -> str:
    """
    Derive a safe GGUF filename from model_tag.

    Examples:
      'llama3.2:8b-instruct-q4_K_M' → 'llama3.2-8b-instruct-q4_K_M.gguf'
      'user/mymodel:q4'              → 'mymodel-q4.gguf'

    Uses Path(...).name to strip any directory prefix (R3: path traversal mitigation).
    """
    namespace, model, tag = _parse_model_tag(model_tag)
    # Strip any directory prefix from the model name component
    safe_model = Path(model).name
    if tag and tag != "latest":
        return f"{safe_model}-{tag}.gguf"
    return f"{safe_model}.gguf"


# ─── OCI Manifest ─────────────────────────────────────────────────────────────


def get_manifest(model: str, tag: str, namespace: str = "library") -> dict:
    """
    Fetch OCI manifest from the Ollama registry.

    GET registry.ollama.ai/v2/{namespace}/{model}/manifests/{tag}

    Returns dict with keys:
      digest (str): SHA256 digest of the GGUF blob
      size (int): blob size in bytes
      blob_digest (str): digest reference for the blob URL
      url (str): full blob URL

    Raises:
      ValueError: if manifest structure is unexpected (logs raw response for debugging)
      requests.HTTPError: on non-200 responses
    """
    session = _get_session()
    url = f"{_REGISTRY_BASE}/{namespace}/{model}/manifests/{tag}"
    resp = session.get(url, timeout=30)

    if resp.status_code == 401:
        # Ollama may return 401 with WWW-Authenticate for auth challenge,
        # but public models are accessible — retry without auth is the norm.
        raise ValueError(
            f"Authentication required for {namespace}/{model}:{tag}. "
            "Only public models are supported (anonymous access)."
        )
    resp.raise_for_status()

    try:
        manifest = resp.json()
    except Exception as exc:
        raise ValueError(
            f"Unexpected manifest response for {namespace}/{model}:{tag} — "
            f"not valid JSON. Raw (first 500 chars): {resp.text[:500]!r}"
        ) from exc

    # OCI manifest has 'layers' array; find the GGUF model layer
    layers = manifest.get("layers") or []
    model_layer = None
    for layer in layers:
        if isinstance(layer, dict) and layer.get("mediaType") == _OLLAMA_MODEL_MEDIA_TYPE:
            model_layer = layer
            break

    if model_layer is None:
        # Log raw manifest for debugging (R1 mitigation)
        available_types = [
            layer.get("mediaType", "unknown")
            for layer in layers
            if isinstance(layer, dict)
        ]
        raise ValueError(
            f"No GGUF model layer found in manifest for {namespace}/{model}:{tag}. "
            f"Available mediaTypes: {available_types}. "
            f"Raw manifest (first 1000 chars): {json.dumps(manifest)[:1000]!r}"
        )

    blob_digest = model_layer.get("digest", "")
    blob_size = int(model_layer.get("size", 0))

    if not blob_digest:
        raise ValueError(
            f"Model layer has no digest in manifest for {namespace}/{model}:{tag}. "
            f"Raw layer: {model_layer!r}"
        )

    blob_url = f"{_REGISTRY_BASE}/{namespace}/{model}/blobs/{blob_digest}"

    return {
        "digest": blob_digest,
        "size": blob_size,
        "blob_digest": blob_digest,
        "url": blob_url,
    }


# ─── Model Info ───────────────────────────────────────────────────────────────


def get_model_info(model_tag: str) -> dict:
    """
    Fetch metadata for an Ollama model.

    Returns dict with keys:
      model_tag (str): the input model_tag
      digest (str): SHA256 digest of the GGUF blob (for idempotency)
      size (int): file size in bytes (0 if unknown)
      size_str (str): human-readable size

    Reuses ProxyConfig/load_proxy_config() from utils.py for proxy-aware HTTP.

    Raises:
      ValueError: if model_tag format is invalid
    """
    if not _OLLAMA_TAG_RE.match(model_tag):
        raise ValueError(
            f"Invalid Ollama model tag: {model_tag!r}. "
            "Expected format: model, model:tag, or namespace/model:tag"
        )
    namespace, model, tag = _parse_model_tag(model_tag)
    manifest = get_manifest(model, tag, namespace)
    return {
        "model_tag": model_tag,
        "digest": manifest["digest"],
        "size": manifest["size"],
        "size_str": fmt_size(manifest["size"]) if manifest["size"] else "unknown",
    }


# ─── Search ───────────────────────────────────────────────────────────────────


def _parse_count(text: str) -> int:
    """
    Parse Ollama pull-count strings into integers.

    Examples:
      '13.4K'  → 13_400
      '1.2M'   → 1_200_000
      '234'    → 234
      '1,234'  → 1234
    """
    text = text.strip().replace(",", "")
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    try:
        suffix = text[-1].upper()
        if suffix in multipliers:
            return int(float(text[:-1]) * multipliers[suffix])
        return int(float(text))
    except (ValueError, IndexError):
        return 0


def _parse_ollama_search_html(html: str, limit: int) -> list[dict]:
    """
    Parse model cards from ollama.com/search HTML.

    Ollama's search page (Go/HTMX) renders model cards as:
      <li x-test-model ...>
        <a href="/library/<model>" ...>
          <span x-test-search-response-title>model_name</span>
          <p class="max-w-lg break-words ...">description text</p>
          <span x-test-pull-count>13.4K</span>
          <span x-test-tag-count>12</span>
          <span class="...bg-cyan-50...">tools</span>  <!-- capability badges -->
        </a>
      </li>

    Strategy:
      1. Find each <li x-test-model>...</li> block to isolate model cards.
      2. Within each card, use targeted regex to extract structured data.
      3. Fallback: bare href regex (no metadata) if card isolation fails.

    Returns list[dict] with keys:
      model_tag, description, pulls, params (capability tags), quantization, tag_count
    """
    results: list[dict] = []

    # ── Primary: isolate model cards by <li x-test-model> blocks ─────────────
    # Each card is a <li x-test-model ...>...</li> without nested <li> elements.
    cards = re.findall(r'<li\s+x-test-model[^>]*>(.*?)</li>', html, re.DOTALL)

    if cards:
        for card in cards[:limit]:
            # Model name from href="/library/<model>"
            href_m = re.search(
                r'href=["\'](?:https://ollama\.com)?/library/([a-z][a-z0-9._-]+)["\']',
                card,
            )
            if not href_m:
                continue
            model_name = href_m.group(1)

            # Description from <p class="max-w-lg break-words ...">...</p>
            desc_m = re.search(
                r'<p[^>]*\bmax-w-lg\b[^>]*\bbreak-words\b[^>]*>(.*?)</p>',
                card, re.DOTALL
            )
            description = ""
            if desc_m:
                # Strip any embedded HTML tags, then decode HTML entities
                import html as _html
                raw_desc = re.sub(r'<[^>]+>', '', desc_m.group(1)).strip()
                description = _html.unescape(raw_desc)

            # Pull count from <span x-test-pull-count>13.4K</span>
            pull_m = re.search(
                r'x-test-pull-count[^>]*>\s*([\d.,]+\s*[KMBkmb]?)\s*<',
                card,
            )
            pulls = _parse_count(pull_m.group(1)) if pull_m else 0

            # Tag count from <span x-test-tag-count>12</span>
            tag_count_m = re.search(r'x-test-tag-count[^>]*>\s*(\d+)\s*<', card)
            tag_count = int(tag_count_m.group(1)) if tag_count_m else 0

            # Capability badges from cyan-coloured spans (tools, vision, cloud, etc.)
            capability_tags = re.findall(
                r'<span[^>]+bg-cyan-\d+[^>]*>\s*([^<]{2,30}?)\s*</span>',
                card,
            )
            # Also look for other colored badge spans (some models use teal/green)
            capability_tags += re.findall(
                r'<span[^>]+bg-(?:teal|green|emerald|sky)-\d+[^>]*>\s*([^<]{2,30}?)\s*</span>',
                card,
            )

            results.append({
                "model_tag": model_name,
                "description": description,
                "pulls": pulls,
                "params": ", ".join(capability_tags) if capability_tags else "",
                "quantization": "",
                "tag_count": tag_count,
                "size": 0,  # filled in by _fetch_sizes_parallel
            })

        if results:
            return results

    # ── Fallback: href regex (names only, no metadata) ────────────────────────
    # Used when card isolation fails (e.g. site restructure).
    seen: set[str] = set()
    for m in re.finditer(
        r'href=["\'](?:https://ollama\.com)?/library/([a-z][a-z0-9._-]{1,})(?=[/"\'#?])',
        html,
    ):
        name = m.group(1)
        if re.search(r'\.[a-z]{2,4}$', name) or name in seen:
            continue
        seen.add(name)
        results.append({
            "model_tag": name,
            "description": "",
            "pulls": 0,
            "params": "",
            "quantization": "",
            "tag_count": 0,
            "size": 0,
        })
        if len(results) >= limit:
            break

    return results


def _fetch_top_models_fallback(limit: int, session) -> list[dict]:
    """
    Fallback: fetch top models from ollama.com/api/tags.
    Called when HTML scraping returns 0 results (R4 mitigation).
    Returns list[dict] in the same format as search_models.
    """
    try:
        resp = session.get(_TAGS_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        models = data if isinstance(data, list) else data.get("models", [])
        results = []
        for m in models[:limit]:
            if isinstance(m, dict):
                name = m.get("name") or m.get("model") or ""
                results.append({
                    "model_tag": name,
                    "description": str(m.get("description") or ""),
                    "pulls": 0,
                    "params": "",
                    "quantization": "",
                    "tag_count": 0,
                    "size": 0,
                })
            elif isinstance(m, str):
                results.append({
                    "model_tag": m,
                    "description": "",
                    "pulls": 0,
                    "params": "",
                    "quantization": "",
                    "tag_count": 0,
                    "size": 0,
                })
        return results
    except Exception as exc:
        print(
            f"[WARN] ollama_hub: /api/tags fallback failed: {exc}",
            file=sys.stderr,
        )
        return []


def _fetch_sizes_parallel(model_names: list[str], timeout: int = 20) -> dict[str, int]:
    """
    Fetch GGUF file sizes for models by fetching OCI manifests in parallel.

    For each model, fetches registry.ollama.ai/v2/library/<model>/manifests/latest
    and reads the 'size' field of the GGUF layer blob. This is a lightweight
    manifest-only request (JSON, ~1KB), NOT a download of the actual GGUF file.

    Returns dict mapping model_name → size_bytes (0 if unavailable).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not model_names:
        return {}

    def _fetch_one(name: str) -> tuple[str, int]:
        try:
            manifest = get_manifest(name, "latest", "library")
            return name, int(manifest.get("size", 0) or 0)
        except Exception:
            return name, 0

    results: dict[str, int] = {}
    max_workers = min(len(model_names), 8)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, name): name for name in model_names}
            for future in as_completed(futures, timeout=timeout):
                try:
                    name, size = future.result(timeout=5)
                    if size:
                        results[name] = size
                except Exception:
                    pass
    except Exception:
        pass  # TimeoutError from as_completed — return partial results

    return results


def search_models(query: Optional[str] = None, limit: int = 20) -> list[dict]:
    """
    Search Ollama models.

    Primary: GET https://ollama.com/search?q={query} — HTML card parsing.
    Fallback: /api/tags for top models when HTML yields no results.

    After getting model names, fetches OCI manifests in parallel to get
    the GGUF file size (latest variant) for each model.

    Returns list[dict] with keys:
      model_tag (str):    e.g. 'llama3.2'
      description (str):  model description
      pulls (int):        download count
      params (str):       capability tags (tools, vision, cloud, …)
      quantization (str): quantization level (empty — resolved per-tag, not globally)
      tag_count (int):    number of available tags/variants
      size (int):         GGUF file size in bytes for the :latest variant (0 if unknown)
    """
    session = _get_session()
    # Use a browser-like UA to avoid bot-detection on the search page
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    })
    results: list[dict] = []

    try:
        params: dict = {}
        if query:
            params["q"] = query
        resp = session.get(_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        results = _parse_ollama_search_html(resp.text, limit)
    except Exception as exc:
        print(
            f"[WARN] ollama_hub: HTML search failed ({exc}), trying /api/tags fallback",
            file=sys.stderr,
        )

    # R4 fallback: if scrape returned nothing, try /api/tags
    if not results:
        results = _fetch_top_models_fallback(limit, session)

    results = results[:limit]

    # Fetch file sizes from OCI registry in parallel (best-effort)
    if results:
        model_names = [r["model_tag"] for r in results if r.get("model_tag")]
        sizes = _fetch_sizes_parallel(model_names)
        for r in results:
            r["size"] = sizes.get(r["model_tag"], 0)

    return results


# ─── ETag / Digest Sidecar ────────────────────────────────────────────────────
# Same .etag sidecar pattern as download_models.py (_write_local_etag / _read_local_etag)


def _etag_path(local_file: Path) -> Path:
    """Sidecar file storing the last-known digest next to the model file."""
    return local_file.with_suffix(local_file.suffix + ".etag")


def _read_local_digest(local_file: Path) -> Optional[str]:
    """Return locally cached digest string, or None if not available."""
    p = _etag_path(local_file)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip() or None
    return None


def _write_local_digest(local_file: Path, digest: str) -> None:
    """Persist the SHA256 digest alongside the model file."""
    _etag_path(local_file).write_text(digest, encoding="utf-8")


# ─── Download ─────────────────────────────────────────────────────────────────


def download_model_gguf(
    model_tag: str,
    dest_path: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    retry_count: int = _DEFAULT_RETRY_COUNT,
    retry_delay: float = _DEFAULT_RETRY_DELAY,
) -> Path:
    """
    Download a GGUF model file from the Ollama OCI registry.

    Idempotency: compares SHA256 digest sidecar with remote manifest digest.
    If they match, skips the download and returns dest_path.

    Emits progress markers compatible with model_browser.py worker:
      [FILESIZE] {bytes}        — once, if Content-Length header is present (R6)
      [FILEPROGRESS] {cur}/{total}  — during streaming

    Args:
      model_tag:  Ollama model tag (e.g. 'llama3.2:8b-instruct-q4_K_M')
      dest_path:  Full destination path for the GGUF file.
                  If it's a directory, filename is derived from model_tag.
      progress_callback: Optional callback(bytes_done, total_bytes).
      retry_count: Number of retries on failure.
      retry_delay: Base delay between retries (exponential backoff).

    Returns:
      Path to the downloaded GGUF file.

    Raises:
      ValueError: if model_tag format is invalid, or on manifest parse errors (R1 mitigation)
      requests.HTTPError: on non-200 HTTP responses
    """
    if not _OLLAMA_TAG_RE.match(model_tag):
        raise ValueError(
            f"Invalid Ollama model tag: {model_tag!r}. "
            "Expected format: model, model:tag, or namespace/model:tag"
        )

    # Resolve destination path
    if dest_path.is_dir() or not dest_path.suffix:
        dest_path = dest_path / _safe_filename(model_tag)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch manifest to get digest and blob URL
    namespace, model, tag = _parse_model_tag(model_tag)
    manifest = get_manifest(model, tag, namespace)
    remote_digest = manifest["digest"]
    expected_size = manifest["size"]

    # Idempotency check: compare stored digest with remote
    if dest_path.is_file():
        local_digest = _read_local_digest(dest_path)
        if local_digest and local_digest == remote_digest:
            print(f"  [SKIP] Digest match — already current: {dest_path}")
            return dest_path

    # Announce expected file size (R6: only if available from manifest)
    if expected_size:
        print(f"[FILESIZE] {expected_size}")
        sys.stdout.flush()

    session = _get_session()
    last_exc: Optional[Exception] = None

    for attempt in range(retry_count + 1):
        try:
            if attempt > 0:
                print(f"  [RETRY {attempt}/{retry_count}] {model_tag}")
            else:
                print(f"  Downloading {model_tag} from Ollama registry ...")

            resp = session.get(manifest["url"], stream=True, timeout=(15, 300))
            resp.raise_for_status()

            # Get actual size from Content-Length header (may differ from manifest if compressed)
            content_length_str = resp.headers.get("Content-Length")
            total_bytes: int = 0
            if content_length_str:
                try:
                    total_bytes = int(content_length_str)
                    # If manifest didn't have size, emit [FILESIZE] from header
                    if not expected_size:
                        print(f"[FILESIZE] {total_bytes}")
                        sys.stdout.flush()
                except ValueError:
                    pass
            elif expected_size:
                total_bytes = expected_size

            # Stream download with progress reporting
            bytes_done = 0
            sha256 = hashlib.sha256()

            # Write to a temp file first (atomic rename on success)
            import tempfile as _tempfile
            import os as _os
            tmp_fd, tmp_path = _tempfile.mkstemp(dir=str(dest_path.parent), suffix=".gguf.tmp")
            try:
                with _os.fdopen(tmp_fd, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            sha256.update(chunk)
                            bytes_done += len(chunk)
                            # Emit progress marker
                            print(f"[FILEPROGRESS] {bytes_done}/{total_bytes}")
                            sys.stdout.flush()
                            if progress_callback is not None:
                                progress_callback(bytes_done, total_bytes)

                # Atomic rename
                _os.replace(tmp_path, str(dest_path))
            except Exception:
                # Clean up temp file on failure
                try:
                    _os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            # Verify download size
            if bytes_done == 0:
                raise ValueError("Downloaded file is empty — possible network error")

            # Store digest sidecar for future idempotency checks
            # Note: remote_digest is the OCI blob digest (sha256:...) from the manifest
            _write_local_digest(dest_path, remote_digest)

            actual_size_str = fmt_size(bytes_done)
            print(f"  [OK]   Saved to: {dest_path}  ({actual_size_str})")
            return dest_path

        except Exception as exc:
            last_exc = exc
            if attempt >= retry_count:
                break
            wait = retry_delay * (2 ** attempt)
            print(
                f"  [WARN] Download error (attempt {attempt + 1}/{retry_count + 1}): {exc}\n"
                f"         Retrying in {wait:.0f}s ..."
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Failed to download {model_tag} after {retry_count + 1} attempts. "
        f"Last error: {last_exc}"
    )
