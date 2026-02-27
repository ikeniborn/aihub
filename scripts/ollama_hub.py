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
  register_with_ollama(local_file, ollama_name, ollama_models_dir)  → None

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

import html as _html
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

# Shared utilities (fmt_size, ProxyConfig, load_proxy_config)
sys.path.insert(0, str(Path(__file__).parent))
from utils import fmt_size, ProxyConfig, load_proxy_config  # noqa: E402

__all__ = ["search_models", "get_model_info", "get_model_tags", "download_model_gguf",
           "register_with_ollama"]

# ─── Constants ────────────────────────────────────────────────────────────────

_REGISTRY_BASE = "https://registry.ollama.ai/v2"
_SEARCH_URL = "https://ollama.com/search"
_TAGS_URL = "https://ollama.com/api/tags"
_LIBRARY_BASE = "https://ollama.com/library"

# Tag page regex: matches <span class="group-hover:underline">model:tag</span>
_TAG_PAGE_RE = re.compile(
    r'<span\s+class="group-hover:underline">[^:<]+:([^<]{1,80})</span>'
)
# Size regex for HTML: matches "4.1 GB", "400 MB", "1.2 KB" etc.
_HTML_SIZE_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(GB|MB|KB)\b', re.IGNORECASE)
_OLLAMA_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"

# Retry settings matching download_models.py pattern
_DEFAULT_RETRY_COUNT = 3
_DEFAULT_RETRY_DELAY = 5.0

# Validation regex for ollama model tags: model[:tag] or user/model[:tag]
_OLLAMA_TAG_RE = re.compile(r"^[\w.\-]+(\/[\w.\-]+)?(:([\w.\-]+))?$")

# Chunk size for streaming download (512 KB)
_CHUNK_SIZE = 512 * 1024

# Capability badge regex — matches any Tailwind bg-{color}-{shade} span
# Covers all common badge colours used by ollama.com (tools, vision, cloud, embed, thinking, …)
_CAP_BADGE_RE = re.compile(
    r'<span[^>]+\bbg-(?:'
    r'cyan|teal|green|emerald|sky|blue|indigo|violet|purple|fuchsia|'
    r'pink|rose|orange|amber|yellow|lime|red'
    r')-\d+[^>]*>(.*?)</span>',
    re.DOTALL,
)


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


def get_model_tags(model_name: str) -> list[dict]:
    """
    Fetch all available tags (size/quant variants) for an Ollama model.

    Scrapes https://ollama.com/library/{model}/tags — the same public page
    the browser shows. The OCI registry /v2/.../tags/list endpoint is not
    supported by registry.ollama.ai (returns 404).

    Returns list[dict] with keys:
      name (str):       tag string, e.g. '7b-instruct-q4_K_M'
      size_bytes (int): file size in bytes scraped from the page (0 if unknown)

    Order preserved as on the page ('latest' first, then by size/quant).
    """
    # Strip any existing tag / namespace to get the bare model name
    base = model_name.split(":")[0].split("/")[-1]
    url = f"{_LIBRARY_BASE}/{base}/tags"
    session = _get_session()
    # Use a browser-like User-Agent to avoid bot detection
    session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120"})
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    # Split page by tag anchor spans; each block starts right after the span
    parts = re.split(r'<span\s+class="group-hover:underline">', resp.text)
    seen: set = set()
    result = []
    _multipliers = {"KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}
    for part in parts[1:]:  # skip content before the first tag span
        # Extract tag name: first match is "model:tag</span>"
        m_tag = re.match(r'[^:<]*:?([^<]{1,80})</span>', part)
        if not m_tag:
            continue
        tag = m_tag.group(1).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        # Try to extract file size from the next ~600 chars of this block
        size_bytes = 0
        m_size = _HTML_SIZE_RE.search(part[:600])
        if m_size:
            num = float(m_size.group(1))
            unit = m_size.group(2).upper()
            size_bytes = int(num * _multipliers.get(unit, 0))
        result.append({"name": tag, "size_bytes": size_bytes})
    return result


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

            # Capability badges: any Tailwind bg-{color}-{shade} span
            # (tools, vision, cloud, embed, thinking, reasoning, code, etc.)
            seen_caps: set = set()
            capability_tags = []
            for raw in _CAP_BADGE_RE.findall(card):
                # Strip inner HTML (SVG icons etc.), decode entities, normalise spaces
                text = _html.unescape(re.sub(r'<[^>]+>', '', raw)).strip()
                text = ' '.join(text.split())
                if 2 <= len(text) <= 30 and text not in seen_caps:
                    seen_caps.add(text)
                    capability_tags.append(text)

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


def search_models(
    query: Optional[str] = None,
    limit: int = 20,
    capabilities: Optional[list] = None,
) -> list[dict]:
    """
    Search Ollama models.

    Primary: GET https://ollama.com/search?q={query}&c={cap} — HTML card parsing.
    Fallback: /api/tags for top models when HTML yields no results.

    Args:
      query:        Search query string.
      limit:        Maximum number of results to return.
      capabilities: Optional list of capability filter values to pass as ?c= to
                    ollama.com (e.g. ['tools', 'thinking']). Multiple values produce
                    separate requests whose results are merged and de-duplicated.

    Returns list[dict] with keys:
      model_tag (str):    e.g. 'llama3.2'
      description (str):  model description
      pulls (int):        download count
      params (str):       capability tags (tools, vision, cloud, thinking, embedding, …)
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

    # Build list of (q, c) request pairs: one per capability filter, or one without
    requests_args: list[tuple] = []
    if capabilities:
        for cap in capabilities:
            cap = str(cap).strip().lower()
            if cap:
                requests_args.append((query, cap))
    if not requests_args:
        requests_args = [(query, None)]

    seen_tags: set = set()
    for req_q, req_cap in requests_args:
        try:
            req_params: dict = {}
            if req_q:
                req_params["q"] = req_q
            if req_cap:
                req_params["c"] = req_cap
            resp = session.get(_SEARCH_URL, params=req_params, timeout=15)
            resp.raise_for_status()
            page_results = _parse_ollama_search_html(resp.text, limit)
            for r in page_results:
                tag = r.get("model_tag", "")
                if tag and tag not in seen_tags:
                    seen_tags.add(tag)
                    results.append(r)
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

    import os as _os

    # Deterministic part file — survives crashes and enables resume
    part_path = Path(str(dest_path) + ".part")
    session = _get_session()
    last_exc: Optional[Exception] = None

    for attempt in range(retry_count + 1):
        try:
            if attempt > 0:
                print(f"  [RETRY {attempt}/{retry_count}] {model_tag}")
            else:
                print(f"  Downloading {model_tag} from Ollama registry ...")

            # Determine resume offset from existing part file
            resume_from = 0
            if part_path.is_file():
                part_size = part_path.stat().st_size
                if 0 < part_size < (expected_size or part_size + 1):
                    resume_from = part_size
                    pct = (resume_from * 100 // expected_size) if expected_size else 0
                    print(
                        f"  [RESUME] Found partial: {fmt_size(resume_from)} / "
                        f"{fmt_size(expected_size or 0)} ({pct}%) — resuming"
                    )
                else:
                    # Empty or oversized part file — start fresh
                    part_path.unlink(missing_ok=True)

            # Build request headers
            req_headers: dict = {}
            if resume_from > 0:
                req_headers["Range"] = f"bytes={resume_from}-"

            resp = session.get(
                manifest["url"], stream=True, timeout=(15, 300), headers=req_headers
            )

            # Handle 416 Range Not Satisfiable — server says file is complete
            if resp.status_code == 416:
                if part_path.is_file():
                    _os.replace(str(part_path), str(dest_path))
                    _write_local_digest(dest_path, remote_digest)
                    print(f"  [RESUME] File already complete (416) — renamed part file")
                    return dest_path
                raise RuntimeError("Server returned 416 but no part file present")

            resp.raise_for_status()

            # Determine write mode and starting offset from response status
            if resp.status_code == 206:
                file_mode = "ab"
                bytes_done = resume_from
            else:
                # 200 — server doesn't support ranges; start fresh
                file_mode = "wb"
                bytes_done = 0
                resume_from = 0

            # Resolve total bytes for progress reporting
            content_length_str = resp.headers.get("Content-Length")
            total_bytes: int = expected_size or 0
            if content_length_str:
                try:
                    cl = int(content_length_str)
                    # For 206, Content-Length is remaining bytes; full = offset + remaining
                    total_bytes = resume_from + cl
                    if not expected_size:
                        print(f"[FILESIZE] {total_bytes}")
                        sys.stdout.flush()
                except ValueError:
                    pass

            # Stream to part file (append on resume, overwrite on fresh start)
            with open(part_path, file_mode) as f:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bytes_done += len(chunk)
                        print(f"[FILEPROGRESS] {bytes_done}/{total_bytes}")
                        sys.stdout.flush()
                        if progress_callback is not None:
                            progress_callback(bytes_done, total_bytes)

            if bytes_done == 0 and resume_from == 0:
                raise ValueError("Downloaded file is empty — possible network error")

            # Atomic rename and persist digest sidecar
            _os.replace(str(part_path), str(dest_path))
            _write_local_digest(dest_path, remote_digest)

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
            # Part file is kept — next attempt will resume from where we left off
            time.sleep(wait)

    raise RuntimeError(
        f"Failed to download {model_tag} after {retry_count + 1} attempts. "
        f"Last error: {last_exc}"
    )


# ─── Ollama registration (zero-copy) ──────────────────────────────────────────


def register_with_ollama(
    local_file: Path,
    ollama_name: str,
    ollama_models_dir: Optional[Path] = None,
) -> None:
    """Register a locally downloaded GGUF with Ollama without data duplication.

    Algorithm:
      1. Read SHA256 digest from the .etag sidecar (format: ``sha256:hexhash``).
      2. Pre-create a hardlink at ``$OLLAMA_MODELS/blobs/sha256-{hash}`` pointing
         to *local_file*.  On cross-device filesystems falls back to a symlink.
         If neither succeeds, a warning is printed and step 3 continues — Ollama
         will copy the file in that case (standard behavior, no data loss).
      3. Run ``ollama create {ollama_name} -f <Modelfile>`` where the Modelfile
         contains a single ``FROM {local_file}`` directive.
         Because the blob already exists at the expected path, Ollama only writes
         the manifest and skips copying — achieving zero storage duplication.

    Requirements:
      - ``ollama`` must be installed and available on PATH.
      - Hardlink only avoids duplication when *local_file* and ``$OLLAMA_MODELS``
        reside on the same filesystem/device.

    Args:
      local_file:        Path to the downloaded GGUF file.
      ollama_name:       Ollama model name (e.g. ``'qwen3:7b'``, ``'llama3.2'``).
      ollama_models_dir: Override for ``$OLLAMA_MODELS`` directory.
                         Defaults to the env var, then ``~/.ollama``.

    Raises:
      FileNotFoundError:          if *local_file* does not exist.
      subprocess.CalledProcessError: if ``ollama create`` exits non-zero.
    """
    import os as _os
    import subprocess
    import tempfile

    if not local_file.is_file():
        raise FileNotFoundError(f"GGUF file not found: {local_file}")

    # Resolve $OLLAMA_MODELS directory
    if ollama_models_dir is None:
        env_val = _os.environ.get("OLLAMA_MODELS")
        ollama_models_dir = Path(env_val) if env_val else Path.home() / ".ollama"

    # Step 1: read SHA256 from .etag sidecar written during download
    digest = _read_local_digest(local_file)  # "sha256:hexhash" or None

    # Step 2: pre-create hardlink (or symlink) in blobs/ so Ollama skips the copy
    if digest and digest.startswith("sha256:"):
        hex_hash = digest[len("sha256:"):]
        blobs_dir = ollama_models_dir / "blobs"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        blob_path = blobs_dir / f"sha256-{hex_hash}"

        if blob_path.exists():
            print(f"  [OLLAMA] Blob already present: {blob_path.name}")
        else:
            try:
                _os.link(local_file, blob_path)
                print(f"  [OLLAMA] Hardlink created: {blob_path.name}")
            except OSError:
                # Cross-device — fall back to absolute symlink
                try:
                    blob_path.symlink_to(local_file.resolve())
                    print(
                        f"  [OLLAMA] Symlink created: {blob_path.name} → {local_file}"
                    )
                except OSError as sym_err:
                    print(
                        f"  [OLLAMA] Warning: could not pre-create blob ({sym_err}); "
                        "ollama create will copy the file"
                    )
    else:
        print(
            "  [OLLAMA] Warning: no .etag digest found — "
            "skipping blob pre-creation; ollama create will copy the file"
        )

    # Step 3: create Modelfile and run `ollama create`
    modelfile_content = f"FROM {local_file.resolve()}\n"
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".Modelfile", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(modelfile_content)
        tmp.close()
        print(f"  [OLLAMA] Running: ollama create {ollama_name}")
        subprocess.run(
            ["ollama", "create", ollama_name, "-f", tmp.name],
            check=True,
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    print(f"  [OLLAMA] Registered '{ollama_name}' successfully")
