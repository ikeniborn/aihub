#!/usr/bin/env python3
"""
AI Model Browser — Web UI
=========================
HTTP-сервер для просмотра и управления AI-моделями из models.yaml.

Использует только Python stdlib (http.server, json, pathlib, tempfile, os).
Без Flask/FastAPI, без внешних зависимостей.

Запуск:
    python scripts/model_browser.py
    python scripts/model_browser.py --port 9000
    python scripts/model_browser.py --config models.yaml --port 9000

Открыть в браузере: http://localhost:9000

API:
    GET  /api/models  — список моделей с размерами файлов на диске
    POST /api/save    — обновить enabled в models.yaml атомарно
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import yaml


# ─── HTML ─────────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Model Browser</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  header { background: #1a1d27; border-bottom: 1px solid #2d3748; padding: 16px 24px;
           display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  header h1 { font-size: 1.25rem; font-weight: 700; color: #a78bfa; white-space: nowrap; }
  .controls { display: flex; gap: 10px; flex-wrap: wrap; flex: 1; align-items: center; }
  .controls input[type=text] {
    background: #2d3748; border: 1px solid #4a5568; border-radius: 6px;
    color: #e2e8f0; padding: 6px 12px; font-size: 0.875rem; width: 220px;
    outline: none; transition: border-color 0.15s;
  }
  .controls input[type=text]:focus { border-color: #a78bfa; }
  .tag-filter { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .tag-btn {
    background: #2d3748; border: 1px solid #4a5568; border-radius: 4px;
    color: #94a3b8; padding: 4px 10px; font-size: 0.75rem; cursor: pointer;
    transition: all 0.15s; white-space: nowrap;
  }
  .tag-btn:hover { border-color: #a78bfa; color: #c4b5fd; }
  .tag-btn.active { background: #4c1d95; border-color: #a78bfa; color: #ede9fe; }
  .toggle-label {
    display: flex; align-items: center; gap: 6px; cursor: pointer;
    font-size: 0.8rem; color: #94a3b8; white-space: nowrap; user-select: none;
  }
  .toggle-label input { accent-color: #a78bfa; width: 15px; height: 15px; }
  .save-btn {
    background: #6d28d9; border: none; border-radius: 6px; color: #fff;
    padding: 7px 20px; font-size: 0.875rem; font-weight: 600; cursor: pointer;
    transition: background 0.15s; white-space: nowrap; margin-left: auto;
  }
  .save-btn:hover { background: #7c3aed; }
  .save-btn:disabled { background: #4a5568; cursor: not-allowed; opacity: 0.6; }
  .save-msg { font-size: 0.8rem; color: #68d391; display: none; }
  main { padding: 20px 24px; }
  .stats { font-size: 0.8rem; color: #718096; margin-bottom: 14px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th {
    background: #1a1d27; color: #94a3b8; font-weight: 600; text-align: left;
    padding: 10px 12px; border-bottom: 2px solid #2d3748; position: sticky;
    top: 0; z-index: 10; white-space: nowrap;
  }
  td { padding: 9px 12px; border-bottom: 1px solid #1e2533; vertical-align: top; }
  tr:hover td { background: #1a1f2e; }
  tr.row-disabled td { opacity: 0.45; }
  tr.row-disabled:hover td { background: #161b28; }
  .status-dot {
    display: inline-block; width: 9px; height: 9px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle; flex-shrink: 0;
  }
  .dot-downloaded { background: #48bb78; box-shadow: 0 0 5px #48bb7866; }
  .dot-notfound   { background: #4a5568; }
  .cell-filename { font-family: 'SF Mono', Consolas, monospace; font-size: 0.78rem;
                   color: #a0aec0; max-width: 220px; word-break: break-all; }
  .cell-repo { color: #a78bfa; font-size: 0.8rem; }
  .cell-size { font-family: monospace; font-size: 0.82rem; white-space: nowrap; }
  .cell-size.downloaded { color: #68d391; }
  .cell-size.notfound   { color: #4a5568; }
  .tags-cell { display: flex; flex-wrap: wrap; gap: 4px; }
  .tag {
    background: #2d3748; border-radius: 3px; padding: 2px 7px;
    font-size: 0.7rem; color: #94a3b8; white-space: nowrap;
  }
  .vram { font-size: 0.78rem; color: #718096; white-space: nowrap; }
  .desc { color: #718096; font-size: 0.78rem; max-width: 260px; }
  .gated-badge {
    display: inline-block; background: #744210; color: #fcd34d;
    border-radius: 3px; padding: 1px 6px; font-size: 0.65rem; font-weight: 700;
    margin-left: 4px; vertical-align: middle;
  }
  .enabled-cb { accent-color: #a78bfa; width: 16px; height: 16px; cursor: pointer; }
  #empty-msg { text-align: center; color: #4a5568; padding: 40px; font-size: 0.9rem; }
</style>
</head>
<body>
<header>
  <h1>AI Model Browser</h1>
  <div class="controls">
    <input type="text" id="search" placeholder="Поиск по модели, описанию..." oninput="applyFilters()">
    <div class="tag-filter" id="tag-filter"></div>
    <label class="toggle-label">
      <input type="checkbox" id="only-downloaded" onchange="applyFilters()">
      только скачанные
    </label>
  </div>
  <button class="save-btn" id="save-btn" onclick="saveChanges()" disabled>Сохранить</button>
  <span class="save-msg" id="save-msg">Сохранено!</span>
</header>
<main>
  <div class="stats" id="stats">Загрузка...</div>
  <table id="models-table">
    <thead>
      <tr>
        <th>Вкл</th>
        <th>Статус</th>
        <th>Модель / Файл</th>
        <th>Теги</th>
        <th>VRAM</th>
        <th>Размер</th>
        <th>Описание</th>
      </tr>
    </thead>
    <tbody id="models-body"></tbody>
  </table>
  <div id="empty-msg" style="display:none">Ничего не найдено</div>
</main>

<script>
let allModels = [];
let changes = {};    // key: repo_id+"||"+filename → new enabled value
let activeTags = new Set();

function fmtSize(bytes) {
  if (!bytes || bytes === 0) return '—';
  const units = ['B','KB','MB','GB','TB'];
  let v = bytes, u = 0;
  while (v >= 1024 && u < units.length - 1) { v /= 1024; u++; }
  return v.toFixed(1) + ' ' + units[u];
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

function modelKey(m) { return m.repo_id + '||' + m.filename; }

function collectTags(models) {
  const tags = new Set();
  models.forEach(m => (m.tags || []).forEach(t => tags.add(t)));
  return [...tags].sort();
}

function renderTagFilter(tags) {
  const el = document.getElementById('tag-filter');
  el.innerHTML = '';
  tags.forEach(t => {
    const btn = document.createElement('button');
    btn.className = 'tag-btn' + (activeTags.has(t) ? ' active' : '');
    btn.textContent = t;
    btn.onclick = () => toggleTag(t, btn);
    el.appendChild(btn);
  });
}

function toggleTag(tag, btn) {
  if (activeTags.has(tag)) {
    activeTags.delete(tag);
    btn.classList.remove('active');
  } else {
    activeTags.add(tag);
    btn.classList.add('active');
  }
  applyFilters();
}

function applyFilters() {
  const q = document.getElementById('search').value.toLowerCase().trim();
  const onlyDl = document.getElementById('only-downloaded').checked;
  const tbody = document.getElementById('models-body');
  const rows = tbody.querySelectorAll('tr');
  let visible = 0;
  rows.forEach(row => {
    const idx = parseInt(row.dataset.idx);
    const m = allModels[idx];
    if (!m) return;
    let show = true;
    if (q) {
      const hay = (m.repo_id + ' ' + m.filename + ' ' + (m.description || '')).toLowerCase();
      if (!hay.includes(q)) show = false;
    }
    if (show && onlyDl && !m.downloaded) show = false;
    if (show && activeTags.size > 0) {
      const mt = new Set(m.tags || []);
      for (const t of activeTags) { if (!mt.has(t)) { show = false; break; } }
    }
    row.style.display = show ? '' : 'none';
    if (show) visible++;
  });
  const em = document.getElementById('empty-msg');
  em.style.display = visible === 0 ? '' : 'none';
  document.getElementById('stats').textContent =
    `Показано: ${visible} из ${allModels.length} моделей` +
    (Object.keys(changes).length > 0 ? ` · Изменений: ${Object.keys(changes).length}` : '');
}

function onEnabledChange(cb, m) {
  const key = modelKey(m);
  const origEnabled = m.enabled;
  const newEnabled = cb.checked;
  if (newEnabled === origEnabled) {
    delete changes[key];
  } else {
    changes[key] = newEnabled;
  }
  const row = cb.closest('tr');
  if (newEnabled) {
    row.classList.remove('row-disabled');
  } else {
    row.classList.add('row-disabled');
  }
  const saveBtn = document.getElementById('save-btn');
  saveBtn.disabled = Object.keys(changes).length === 0;
  applyFilters();
}

function renderModels(models) {
  const tbody = document.getElementById('models-body');
  tbody.innerHTML = '';
  models.forEach((m, i) => {
    const key = modelKey(m);
    const enabled = key in changes ? changes[key] : m.enabled;
    const downloaded = m.downloaded;
    const tr = document.createElement('tr');
    tr.dataset.idx = i;
    if (!enabled) tr.classList.add('row-disabled');

    // enabled checkbox
    const tdCb = document.createElement('td');
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.className = 'enabled-cb'; cb.checked = enabled;
    cb.onchange = () => onEnabledChange(cb, m);
    tdCb.appendChild(cb);

    // status dot
    const tdStatus = document.createElement('td');
    const dot = document.createElement('span');
    dot.className = 'status-dot ' + (downloaded ? 'dot-downloaded' : 'dot-notfound');
    dot.title = downloaded ? 'Скачана' : 'Не скачана';
    tdStatus.appendChild(dot);

    // repo + filename
    const tdName = document.createElement('td');
    tdName.innerHTML =
      `<div class="cell-repo">${escHtml(m.repo_id)}${m.gated ? '<span class="gated-badge">GATED</span>' : ''}</div>` +
      `<div class="cell-filename">${escHtml(m.filename)}</div>`;

    // tags
    const tdTags = document.createElement('td');
    tdTags.innerHTML = '<div class="tags-cell">' +
      (m.tags || []).map(t => `<span class="tag">${escHtml(t)}</span>`).join('') +
      '</div>';

    // vram
    const tdVram = document.createElement('td');
    tdVram.className = 'vram';
    tdVram.textContent = m.vram_gb ? m.vram_gb + ' GB' : '—';

    // size
    const tdSize = document.createElement('td');
    tdSize.className = 'cell-size ' + (downloaded ? 'downloaded' : 'notfound');
    tdSize.textContent = downloaded ? fmtSize(m.disk_size_bytes) : '—';

    // description
    const tdDesc = document.createElement('td');
    tdDesc.className = 'desc';
    tdDesc.textContent = m.description || '';

    tr.append(tdCb, tdStatus, tdName, tdTags, tdVram, tdSize, tdDesc);
    tbody.appendChild(tr);
  });
}

async function loadModels() {
  try {
    const resp = await fetch('/api/models');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    allModels = await resp.json();
    const tags = collectTags(allModels);
    renderTagFilter(tags);
    renderModels(allModels);
    applyFilters();
  } catch (e) {
    document.getElementById('stats').textContent = 'Ошибка загрузки: ' + e.message;
  }
}

async function saveChanges() {
  const btn = document.getElementById('save-btn');
  const msg = document.getElementById('save-msg');
  btn.disabled = true;
  msg.style.display = 'none';
  const updates = Object.entries(changes).map(([key, enabled]) => {
    const [repo_id, filename] = key.split('||');
    return { repo_id, filename, enabled };
  });
  try {
    const resp = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: 'HTTP ' + resp.status }));
      throw new Error(err.error || 'HTTP ' + resp.status);
    }
    // Apply changes to allModels in-place
    for (const [key, enabled] of Object.entries(changes)) {
      const [repo_id, filename] = key.split('||');
      const m = allModels.find(x => x.repo_id === repo_id && x.filename === filename);
      if (m) m.enabled = enabled;
    }
    changes = {};
    msg.textContent = 'Сохранено!';
    msg.style.color = '#68d391';
    msg.style.display = '';
    setTimeout(() => { msg.style.display = 'none'; }, 3000);
    applyFilters();
  } catch (e) {
    msg.textContent = 'Ошибка: ' + e.message;
    msg.style.color = '#fc8181';
    msg.style.display = '';
    btn.disabled = Object.keys(changes).length === 0;
  }
}

loadModels();
</script>
</body>
</html>
"""


# ─── Config Loading ────────────────────────────────────────────────────────────


def _fmt_size(n: int) -> str:
    """Format byte count to human-readable string. Reused from download_models.py pattern."""
    if n == 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_yaml(config_path: Path) -> dict:
    """Load and parse YAML file, return raw dict."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_models_json(config_path: Path) -> list[dict]:
    """
    Read models.yaml and return list of model dicts enriched with disk info.

    For each model, checks if the file exists at:
        {settings.models_dir}/{dest_dir}/{filename}
    and adds disk_size_bytes (int or 0) and downloaded (bool).
    """
    raw = load_yaml(config_path)
    settings: dict = raw.get("settings", {})
    models_dir = Path(settings.get("models_dir", "./models"))

    # Resolve models_dir relative to config_path location
    if not models_dir.is_absolute():
        models_dir = config_path.parent / models_dir

    result = []
    for item in raw.get("models", []):
        if not isinstance(item, dict):
            continue
        repo_id = item.get("repo_id", "")
        filename = item.get("filename", "")
        dest_dir = item.get("dest_dir", "misc")
        if not repo_id or not filename:
            continue

        local_file = models_dir / dest_dir / filename
        downloaded = local_file.is_file()
        disk_size_bytes = int(local_file.stat().st_size) if downloaded else 0

        result.append({
            "repo_id": repo_id,
            "filename": filename,
            "dest_dir": dest_dir,
            "enabled": bool(item.get("enabled", True)),
            "gated": bool(item.get("gated", False)),
            "tags": list(item.get("tags", []) or []),
            "vram_gb": float(item.get("vram_gb", 0) or 0),
            "description": str(item.get("description", "") or ""),
            "downloaded": downloaded,
            "disk_size_bytes": disk_size_bytes,
            "disk_size_str": _fmt_size(disk_size_bytes) if downloaded else "-",
        })

    return result


def save_models(config_path: Path, updates: list[dict]) -> None:
    """
    Atomically update 'enabled' field in models.yaml for given models.

    Updates is a list of {repo_id, filename, enabled} dicts.
    Uses tempfile + os.replace() for POSIX atomic write (R1 mitigation).
    Comments in models.yaml will be lost (acceptable per task requirements).
    """
    raw = load_yaml(config_path)
    update_map = {
        (u["repo_id"], u["filename"]): bool(u["enabled"])
        for u in updates
    }

    models_list = raw.get("models", []) or []
    for item in models_list:
        if not isinstance(item, dict):
            continue
        key = (item.get("repo_id", ""), item.get("filename", ""))
        if key in update_map:
            item["enabled"] = update_map[key]

    # Atomic write: write to temp in same dir, then rename (R1 mitigation)
    config_dir = config_path.parent
    fd, tmp_path = tempfile.mkstemp(dir=str(config_dir), suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(raw, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        os.replace(tmp_path, str(config_path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── HTTP Handler ──────────────────────────────────────────────────────────────


class ModelBrowserHandler(BaseHTTPRequestHandler):
    """HTTP request handler for AI Model Browser."""

    config_path: Path  # set by server factory

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """Suppress default access log; print minimal info."""
        print(f"  {self.address_string()} {format % args}")

    def _send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path == "/index.html":
            self._send_html(HTML_PAGE)
        elif self.path == "/api/models":
            try:
                models = get_models_json(self.config_path)
                self._send_json(models)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
                updates = payload.get("updates", [])
                if not isinstance(updates, list):
                    raise ValueError("'updates' must be a list")
                save_models(self.config_path, updates)
                self._send_json({"status": "ok", "updated": len(updates)})
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
        else:
            self._send_json({"error": "Not found"}, status=404)


# ─── Server Factory ────────────────────────────────────────────────────────────


def make_handler(config_path: Path):
    """Create a handler class with config_path bound."""

    class Handler(ModelBrowserHandler):
        pass

    Handler.config_path = config_path
    return Handler


# ─── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="model_browser.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--port", "-p",
        type=int,
        default=9000,
        metavar="PORT",
        help="Порт HTTP-сервера (по умолчанию: 9000)",
    )
    p.add_argument(
        "--host",
        default="0.0.0.0",
        metavar="HOST",
        help="Адрес для привязки (по умолчанию: 0.0.0.0)",
    )
    p.add_argument(
        "--config",
        default="models.yaml",
        metavar="FILE",
        help="Путь к models.yaml (по умолчанию: models.yaml)",
    )
    return p


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)

    if not config_path.is_file():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        print(
            "  Run from the project root directory or use --config to specify path.",
            file=sys.stderr,
        )
        return 1

    # Validate YAML is parseable before starting server
    try:
        load_yaml(config_path)
    except Exception as e:
        print(f"[ERROR] Cannot parse {config_path}: {e}", file=sys.stderr)
        return 1

    handler_class = make_handler(config_path)
    server = HTTPServer((args.host, args.port), handler_class)

    url = f"http://localhost:{args.port}"
    print(f"[INFO] AI Model Browser")
    print(f"[INFO] Config: {config_path.resolve()}")
    print(f"[INFO] Listening on {args.host}:{args.port}")
    print(f"[INFO] Open in browser: {url}")
    print("[INFO] Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
