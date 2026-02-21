#!/usr/bin/env python3
"""
AI Model Browser — Web UI
=========================
HTTP-сервер для просмотра и управления AI-моделями из models.yaml.

Использует только Python stdlib (http.server, json, pathlib, tempfile, os).
Без Flask/FastAPI, без внешних зависимостей (кроме huggingface_hub для поиска).

Запуск:
    python scripts/model_browser.py
    python scripts/model_browser.py --port 9000
    python scripts/model_browser.py --config models.yaml --port 9000 --open

Открыть в браузере: http://localhost:9000

API:
    GET  /api/settings            — текущие settings из models.yaml (concurrency, bandwidth, timeout)
    GET  /api/models              — список моделей с размерами файлов на диске
    POST /api/save                — обновить enabled в models.yaml атомарно
    GET  /api/search?q=&author=&file_regex=&limit=  — поиск на HuggingFace Hub
    POST /api/add                 — добавить модели из результатов поиска в models.yaml
    POST /api/download/start      — запустить загрузку enabled моделей в фоновом потоке
    GET  /api/download/stream     — SSE поток прогресса загрузки (running, current, log, status_map)
    POST /api/download/cancel     — отменить текущую загрузку
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any, Optional, TypedDict
from urllib.parse import urlparse, parse_qs

import yaml

# Shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils import fmt_size, load_hf_token  # noqa: E402


# ─── Threaded HTTP server ──────────────────────────────────────────────────────


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP-сервер с поддержкой параллельных запросов (для медленного HF API)."""

    daemon_threads = True


# ─── Download State ────────────────────────────────────────────────────────────

class DownloadState(TypedDict):
    running: bool
    cancelled: bool
    process: Optional[subprocess.Popen[str]]  # active download process or None
    log: list[str]
    current: str                    # model name currently being downloaded
    progress: int                   # 0–100
    model_count: int
    done_count: int
    status_map: dict[str, str]      # model name → "DOWNLOAD"|"SKIP"|"ERROR"


_dl_lock: threading.Lock = threading.Lock()
download_state: DownloadState = {
    "running": False,
    "cancelled": False,
    "process": None,
    "log": [],
    "current": "",
    "progress": 0,
    "model_count": 0,
    "done_count": 0,
    "status_map": {},
}


# ─── HTML ─────────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Model Browser</title>
<style>
  :root {
    --bg-base:     #13161f;
    --bg-surface:  #1c2030;
    --bg-elevated: #1e2840;
    --border:      #2e3a52;
    --text-primary:   #e2e8f0;
    --text-secondary: #a8b8d0;
    --text-muted:     #8898aa;
    --accent:      #a78bfa;
    --accent-dark: #6d28d9;
    --accent-hover:#7c3aed;
    --green:  #68d391;
    --red:    #fc8181;
    --yellow: #fcd34d;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg-base); color: var(--text-primary); min-height: 100vh; }

  /* ── Header ── */
  header { background: var(--bg-surface); border-bottom: 1px solid var(--border);
           padding: 12px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.2rem; font-weight: 700; color: var(--accent); }

  /* ── Tabs ── */
  .tabs { display: flex; background: var(--bg-surface);
          border-bottom: 1px solid var(--border); padding: 0 24px; }
  .tab-btn { background: none; border: none; border-bottom: 2px solid transparent;
             color: var(--text-muted); padding: 10px 20px; font-size: 0.875rem;
             font-weight: 500; cursor: pointer; transition: all 0.15s; margin-bottom: -1px; }
  .tab-btn:hover { color: var(--text-secondary); }
  .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* ── Controls bar (tab: Мои модели) ── */
  .controls-bar { background: var(--bg-surface); border-bottom: 1px solid var(--border);
                  padding: 8px 24px; display: flex; align-items: center;
                  gap: 10px; flex-wrap: wrap; }
  .dl-bar { background: var(--bg-elevated); border-bottom: 1px solid var(--border);
            padding: 7px 24px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .controls-bar input[type=text] {
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 6px;
    color: var(--text-primary); padding: 6px 12px; font-size: 0.875rem; width: 220px;
    outline: none; transition: border-color 0.15s;
  }
  .controls-bar input[type=text]:focus { border-color: var(--accent); }

  /* ── Tag filter ── */
  .tag-filter { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .tag-btn {
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 4px;
    color: var(--text-muted); padding: 4px 10px; font-size: 0.75rem; cursor: pointer;
    transition: all 0.15s; white-space: nowrap;
  }
  .tag-btn:hover { border-color: var(--accent); color: #c4b5fd; }
  .tag-btn.active { background: #3b1f8c; border-color: var(--accent); color: #ede9fe; }

  /* ── Toggle label ── */
  .toggle-label {
    display: flex; align-items: center; gap: 6px; cursor: pointer;
    font-size: 0.8rem; color: var(--text-muted); white-space: nowrap; user-select: none;
  }
  .toggle-label input { accent-color: var(--accent); width: 15px; height: 15px; }

  /* ── Buttons ── */
  .btn-primary {
    background: var(--accent-dark); border: none; border-radius: 6px; color: #fff;
    padding: 7px 18px; font-size: 0.875rem; font-weight: 600; cursor: pointer;
    transition: background 0.15s; white-space: nowrap;
  }
  .btn-primary:hover:not(:disabled) { background: var(--accent-hover); }
  .btn-primary:disabled { background: var(--border); cursor: not-allowed; opacity: 0.6; }
  .btn-secondary {
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 6px;
    color: var(--text-secondary); padding: 7px 18px; font-size: 0.875rem;
    cursor: pointer; transition: all 0.15s; white-space: nowrap;
  }
  .btn-secondary:hover { border-color: var(--accent); color: var(--accent); }

  /* ── Status messages ── */
  .status-msg { font-size: 0.8rem; display: none; }
  .status-msg.ok  { color: var(--green);  display: inline; }
  .status-msg.err { color: var(--red);    display: inline; }

  /* ── Main content ── */
  main { padding: 16px 24px; }
  .stats { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 12px; }

  /* ── Table ── */
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th {
    background: var(--bg-surface); color: var(--text-secondary); font-weight: 600;
    text-align: left; padding: 10px 12px; border-bottom: 2px solid var(--border);
    position: sticky; top: 0; z-index: 10; white-space: nowrap;
  }
  td { padding: 9px 12px; border-bottom: 1px solid #1d2438; vertical-align: top; }
  tr:hover td { background: var(--bg-elevated); }
  tr.row-disabled td { opacity: 0.40; }
  tr.row-disabled:hover td { background: #181e2e; }

  /* ── Status dot ── */
  .status-dot {
    display: inline-block; width: 9px; height: 9px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle; flex-shrink: 0;
  }
  .dot-downloaded { background: #48bb78; box-shadow: 0 0 5px #48bb7866; }
  .dot-notfound   { background: #5a6a82; }

  /* ── My-models cell styles ── */
  .cell-filename { font-family: 'SF Mono', Consolas, monospace; font-size: 0.78rem;
                   color: var(--text-secondary); max-width: 220px; word-break: break-all; }
  .cell-repo { color: var(--accent); font-size: 0.82rem; font-weight: 500; }
  .cell-repo a { color: inherit; text-decoration: none; }
  .cell-repo a:hover { text-decoration: underline; }
  .cell-size { font-family: monospace; font-size: 0.82rem; white-space: nowrap; }
  .cell-size.downloaded { color: var(--green); }
  .cell-size.notfound   { color: #5a6a82; }
  .tags-cell { display: flex; flex-wrap: wrap; gap: 4px; }
  .tag {
    background: #243050; border-radius: 3px; padding: 2px 7px;
    font-size: 0.7rem; color: #a8c0e0; white-space: nowrap;
  }
  .vram { font-size: 0.78rem; color: var(--text-muted); white-space: nowrap; }
  .desc { color: var(--text-muted); font-size: 0.78rem; max-width: 260px; }
  .gated-badge {
    display: inline-block; background: #744210; color: var(--yellow);
    border-radius: 3px; padding: 1px 6px; font-size: 0.65rem; font-weight: 700;
    margin-left: 4px; vertical-align: middle;
  }
  .enabled-cb { accent-color: var(--accent); width: 16px; height: 16px; cursor: pointer; }
  #empty-msg { text-align: center; color: #5a6a82; padding: 40px; font-size: 0.9rem; }
  .btn-delete {
    background: none; border: 1px solid #7f1d1d; border-radius: 4px;
    color: #fca5a5; padding: 2px 8px; font-size: 0.75rem; cursor: pointer;
    white-space: nowrap; opacity: 0.6; transition: opacity 0.15s, background 0.15s;
  }
  .btn-delete:hover { opacity: 1; background: #450a0a; }
  .btn-edit {
    background: none; border: 1px solid #1e3a5f; border-radius: 4px;
    color: #7dd3fc; padding: 2px 7px; font-size: 0.75rem; cursor: pointer;
    opacity: 0.55; transition: opacity 0.15s, background 0.15s;
  }
  .btn-edit:hover { opacity: 1; background: #0c1e33; }
  .btn-edit.active { opacity: 1; background: #0f2d47; border-color: #3b82f6; }
  .btn-sm { padding: 4px 12px; font-size: 0.8rem; }
  td.td-actions { width: 72px; text-align: center; white-space: nowrap; }
  /* ── Inline model edit row ── */
  .edit-row > td {
    padding: 0; background: #0a1120;
    border-left: 3px solid var(--accent); border-bottom: 2px solid var(--accent);
  }
  .model-edit-form {
    display: flex; flex-wrap: wrap; gap: 10px 14px;
    padding: 14px 18px; align-items: flex-end;
  }
  .model-edit-field { display: flex; flex-direction: column; gap: 3px; }
  .model-edit-field label { font-size: 0.72rem; color: var(--text-muted); white-space: nowrap; }
  .model-edit-input {
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 4px;
    color: var(--text-primary); padding: 4px 8px; font-size: 0.8rem;
    outline: none; transition: border-color 0.15s;
  }
  .model-edit-input:focus { border-color: var(--accent); }
  .model-edit-input.w-tags  { width: 200px; }
  .model-edit-input.w-vram  { width: 70px; }
  .model-edit-input.w-desc  { width: 260px; }
  .model-edit-input.w-dir   { width: 150px; }
  .dir-move-note { font-size: 0.68rem; color: var(--yellow); margin-top: 2px; }
  .dir-error-hint { font-size: 0.68rem; color: #f87171; margin-top: 2px; display: none; }
  .input-dir-invalid { border-color: #dc2626 !important; background: #1c0a0a !important; }
  .model-edit-actions { display: flex; gap: 8px; align-items: center; padding-bottom: 1px; }
  /* ── Group rows (dest_dir tree) ── */
  .group-row-l1 { background: #111827; cursor: pointer; user-select: none; }
  .group-row-l1:hover { background: #1a2535; }
  .group-row-l1 > td { padding: 7px 12px; font-size: 0.82rem; font-weight: 600; color: #a78bfa; border-bottom: 1px solid #1e2a40; }
  .group-row-l2 { background: #0d1525; cursor: pointer; user-select: none; }
  .group-row-l2:hover { background: #131e30; }
  .group-row-l2 > td { padding: 5px 12px 5px 28px; font-size: 0.78rem; color: #7dd3fc; border-bottom: 1px solid #1a2030; }
  .group-toggle { margin-right: 6px; font-size: 0.75rem; display: inline-block; width: 10px; }
  .group-count { font-size: 0.7rem; color: var(--text-muted); font-weight: normal; margin-left: 6px; }

  /* ── HF Search panel ── */
  .hf-panel { padding: 20px 24px; }
  .hf-search-form {
    display: flex; flex-direction: column;
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 8px; margin-bottom: 14px; overflow: hidden;
  }
  /* Row 1: primary search fields */
  .hf-search-row {
    display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end;
    padding: 14px 16px;
  }
  /* Row 2: file & content filters */
  .hf-filters-row {
    display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start;
    padding: 10px 16px 14px; border-top: 1px solid var(--border);
    background: var(--bg-base);
  }
  .hf-filter-group { display: flex; flex-direction: column; gap: 5px; }
  .hf-filter-group label { font-size: 0.72rem; color: var(--text-muted); }

  .hf-field { display: flex; flex-direction: column; gap: 4px; }
  .hf-field label { font-size: 0.75rem; color: var(--text-muted); }
  .hf-field input, .hf-field select {
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 6px;
    color: var(--text-primary); padding: 7px 12px; font-size: 0.875rem;
    outline: none; transition: border-color 0.15s;
  }
  .hf-field input:focus, .hf-field select:focus { border-color: var(--accent); }
  .hf-field input[type=text]   { width: 180px; }
  .hf-field input[type=number] { width: 70px; }

  /* ── Regex field with validator ── */
  .hf-field-regex { flex: 1; min-width: 260px; }
  .hf-field-regex input { width: 100%; font-family: 'SF Mono', Consolas, monospace; font-size: 0.82rem; }
  .regex-valid   { border-color: var(--green) !important; }
  .regex-invalid { border-color: var(--red)   !important; }
  .regex-error {
    font-size: 0.7rem; color: var(--red); margin-top: 3px; line-height: 1.3;
  }
  .regex-presets {
    display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px;
  }
  .regex-chip {
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 4px;
    padding: 2px 7px; font-size: 0.68rem; color: var(--text-secondary);
    cursor: pointer; transition: border-color 0.13s, color 0.13s; white-space: nowrap;
    font-family: 'SF Mono', Consolas, monospace; user-select: none;
  }
  .regex-chip:hover { border-color: var(--accent); color: var(--accent); }
  .regex-chip.chip-clear {
    font-family: inherit; color: var(--text-muted); border-style: dashed;
  }
  .regex-chip.chip-clear:hover { border-color: var(--red); color: var(--red); }

  /* ── Search filter chips (language / pipeline_tag) ── */
  .filter-chips {
    display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px;
  }
  .filter-chip {
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 4px;
    padding: 2px 8px; font-size: 0.7rem; color: var(--text-secondary);
    cursor: pointer; transition: all 0.13s; user-select: none;
  }
  .filter-chip:hover  { border-color: var(--accent); color: var(--accent); }
  .filter-chip.active { border-color: var(--accent); color: var(--accent);
                        background: #2d1f6e; }
  .hf-field select {
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 6px;
    color: var(--text-primary); padding: 7px 10px; font-size: 0.82rem;
    outline: none; cursor: pointer; transition: border-color 0.15s;
    width: 200px; appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%238898aa' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 10px center; padding-right: 28px;
  }
  /* Size range inputs */
  .size-range { display: flex; align-items: center; gap: 6px; }
  .size-range input {
    width: 75px; background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: 5px; color: var(--text-primary); padding: 5px 8px;
    font-size: 0.82rem; outline: none; transition: border-color 0.15s;
  }
  .size-range input:focus { border-color: var(--accent); }
  .size-range span { color: var(--text-muted); font-size: 0.8rem; }

  #hf-search-status {
    min-height: 22px; font-size: 0.85rem; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px; color: var(--text-muted);
  }
  #hf-search-status.err { color: var(--red); }

  /* Spinner */
  .spinner {
    display: inline-block; width: 15px; height: 15px;
    border: 2px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin 0.7s linear infinite; flex-shrink: 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── HF Results ── */
  #hf-results-wrapper { display: none; }
  .hf-results-controls {
    display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;
  }
  #hf-results-count { font-size: 0.8rem; color: var(--text-muted); }

  .cell-hf-repo { color: var(--accent); font-size: 0.82rem; font-weight: 500; }
  .cell-hf-repo a { color: inherit; text-decoration: none; }
  .cell-hf-repo a:hover { text-decoration: underline; }
  .cell-hf-files { font-family: 'SF Mono', Consolas, monospace; font-size: 0.72rem;
                   color: var(--text-secondary); }
  .file-item { display: block; padding: 1px 0; }
  .file-size { color: var(--text-muted); font-size: 0.68rem; margin-left: 6px; font-family: monospace; }
  .downloads { font-size: 0.8rem; color: var(--text-muted); white-space: nowrap; }

  /* ── HF result meta (pipeline / likes / description) ── */
  .cell-hf-meta { font-size: 0.71rem; color: var(--text-muted); margin-top: 5px; line-height: 1.6; }
  .hf-pipeline-badge {
    display: inline-block; border-radius: 3px; padding: 1px 6px;
    font-size: 0.63rem; font-weight: 600; margin-right: 5px; vertical-align: middle;
    background: #1a2d4a; color: #7dd3fc; border: 1px solid #1e3a5f;
  }
  .hf-likes { color: #f9a8d4; margin-right: 6px; }
  .hf-desc  { color: #8898aa; font-style: italic; }
  .select-cb { accent-color: var(--accent); width: 15px; height: 15px; cursor: pointer; }
  #hf-empty-msg { text-align: center; color: #5a6a82; padding: 40px; font-size: 0.9rem; display: none; }

  /* ── HF Add form ── */
  .hf-add-form {
    display: none; background: var(--bg-surface);
    border: 1px solid var(--accent); border-radius: 8px;
    padding: 18px; margin-top: 16px;
  }
  .hf-add-form h3 { font-size: 0.95rem; color: var(--text-secondary); margin-bottom: 12px; }
  .hf-add-scroll { overflow-x: auto; margin-bottom: 14px; }
  .hf-add-table { width: 100%; border-collapse: collapse; }
  .hf-add-table th {
    font-size: 0.72rem; color: var(--text-muted); text-align: left;
    padding: 4px 8px; border-bottom: 1px solid var(--border); white-space: nowrap;
  }
  .hf-add-table td { padding: 5px 6px; border-bottom: 1px solid #1a2030; vertical-align: middle; }
  .hf-add-table .cell-add-model { min-width: 160px; }
  .hf-add-table .cell-add-model .add-repo { font-size: 0.78rem; color: #a78bfa; }
  .hf-add-table .cell-add-model .add-fname { font-size: 0.7rem; color: var(--text-muted); }
  .col-add-dir  { width: 130px; }
  .col-add-tags { width: 170px; }
  .col-add-vram { width: 70px; }
  .col-add-desc { min-width: 200px; }
  .hf-add-input {
    width: 100%; box-sizing: border-box;
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 4px;
    color: var(--text-primary); padding: 4px 8px; font-size: 0.78rem;
    outline: none; transition: border-color 0.15s;
  }
  .hf-add-input:focus { border-color: var(--accent); }
  .hf-add-actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }

  /* ── Download Controls (before start) ── */
  .dl-controls {
    display: flex; align-items: center; gap: 12px; margin-left: auto; flex-wrap: wrap;
  }
  .dl-ctrl-group {
    display: flex; align-items: center; gap: 5px;
  }
  .dl-ctrl-label {
    font-size: 0.78rem; color: var(--text-secondary); white-space: nowrap; cursor: help;
  }
  .dl-ctrl-input {
    width: 54px; padding: 4px 6px; border-radius: 5px;
    border: 1px solid var(--border); background: var(--bg-elevated);
    color: var(--text-primary); font-size: 0.82rem; text-align: center;
  }
  .dl-ctrl-input:focus { outline: none; border-color: var(--accent); }
  .dl-save-btn {
    padding: 4px 10px; font-size: 0.78rem; border-radius: 5px; cursor: pointer;
    border: 1px solid var(--border); background: var(--bg-elevated);
    color: var(--text-secondary); transition: color 0.15s, border-color 0.15s, background 0.15s;
    white-space: nowrap;
  }
  .dl-save-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--bg-surface); }
  .dl-save-btn.saved  { border-color: var(--green); color: var(--green); }
  .dl-save-btn.error  { border-color: var(--red);   color: var(--red); }
  .dl-ctrl-checkbox { width: 15px; height: 15px; accent-color: var(--accent); cursor: pointer; }

  /* ── Download Panel ── */
  .download-panel {
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 8px; margin: 10px 24px 0; padding: 14px 18px;
  }
  .download-panel-header {
    display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap;
  }
  .download-panel-header > span:first-child { font-size: 0.82rem; color: var(--text-secondary); flex-shrink: 0; }
  .dl-current { font-size: 0.82rem; color: var(--accent); font-weight: 500; flex: 1;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
  .dl-progress-track {
    background: var(--bg-elevated); border-radius: 4px; height: 6px;
    margin-bottom: 8px; overflow: hidden;
  }
  .dl-progress-bar {
    background: var(--accent); height: 100%; border-radius: 4px;
    transition: width 0.4s ease; width: 0%;
  }
  @keyframes dl-indeterminate {
    0%   { transform: translateX(-150%); }
    100% { transform: translateX(550%); }
  }
  .dl-progress-bar.indeterminate {
    width: 20% !important; transition: none;
    animation: dl-indeterminate 1.4s ease-in-out infinite;
  }
  .dl-log {
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 5px;
    color: var(--text-muted); font-family: 'SF Mono', Consolas, monospace; font-size: 0.72rem;
    height: 120px; overflow-y: auto; padding: 8px 10px; white-space: pre-wrap;
    word-break: break-all; resize: vertical; transition: height 0.2s ease;
  }
  .dl-log--expanded { height: 55vh; }
  .dl-icon-btn {
    background: none; border: 1px solid var(--border); border-radius: 4px;
    color: var(--text-muted); font-size: 0.72rem; cursor: pointer; padding: 2px 7px;
    transition: color 0.13s, border-color 0.13s; white-space: nowrap; flex-shrink: 0;
  }
  .dl-icon-btn:hover { color: var(--accent); border-color: var(--accent); }
  .dl-icon-btn.cancel { border-color: #553030; color: var(--red); }
  .dl-icon-btn.cancel:hover { border-color: var(--red); background: #3a1a1a; }
  .dl-badge {
    display: inline-block; border-radius: 3px; padding: 1px 6px;
    font-size: 0.65rem; font-weight: 700; white-space: nowrap;
  }
  .dl-badge-dl   { background: #1a3a2a; color: var(--green); }
  .dl-badge-skip { background: #252a38; color: var(--text-muted); }
  .dl-badge-err  { background: #3a1a1a; color: var(--red); }

  /* ── Download log line colors ── */
  .dl-log .log-skip { color: #718096; font-style: italic; }
  .dl-log .log-ok   { color: var(--green); }
  .dl-log .log-err  { color: var(--red); }
  .dl-log .log-warn { color: var(--yellow); }
</style>
</head>
<body>

<header>
  <h1>AI Model Browser</h1>
</header>

<nav class="tabs">
  <button class="tab-btn active" data-tab="my-models" onclick="switchTab(this)">Мои модели</button>
  <button class="tab-btn" data-tab="hf-search" onclick="switchTab(this)">Поиск HuggingFace</button>
</nav>

<!-- ═══════════════════════ TAB: Мои модели ═══════════════════════ -->
<div id="tab-my-models" class="tab-panel active">
  <!-- Блок 1: Фильтры и управление списком -->
  <div class="controls-bar">
    <input type="text" id="search" placeholder="Поиск по модели, описанию..." oninput="applyFilters()">
    <div class="tag-filter" id="tag-filter"></div>
    <label class="toggle-label">
      <input type="checkbox" id="only-enabled" onchange="applyFilters()">
      только enabled
    </label>
    <label class="toggle-label">
      <input type="checkbox" id="only-downloaded" onchange="applyFilters()">
      только скачанные
    </label>
    <button class="btn-primary" id="save-btn" onclick="saveChanges()" disabled style="margin-left:auto">Сохранить изменения</button>
    <span class="status-msg" id="save-msg"></span>
  </div>
  <!-- Блок 2: Параметры и запуск загрузки -->
  <div class="dl-bar">
    <div class="dl-controls" id="dl-controls">
      <div class="dl-ctrl-group">
        <span class="dl-ctrl-label" title="Параллельных range-get запросов (HF_XET_NUM_CONCURRENT_RANGE_GETS).&#10;Уменьшает число TCP-соединений, но не ограничивает скорость канала.&#10;Для жёсткого лимита используйте поле «Лимит Mbit/s».">Параллельность:</span>
        <input type="number" id="dl-concurrency" class="dl-ctrl-input" value="4" min="1" max="64" step="1">
      </div>
      <div class="dl-ctrl-group">
        <span class="dl-ctrl-label" title="Таймаут на загрузку одного файла (часы).&#10;При зависании процесс завершится. 0 = без ограничений.">Таймаут ч:</span>
        <input type="number" id="dl-timeout" class="dl-ctrl-input" value="2" min="0" max="48" step="0.5">
      </div>
      <div class="dl-ctrl-group">
        <span class="dl-ctrl-label" title="Жёсткий лимит скорости загрузки через tc ingress policing.&#10;Применяется на сетевом интерфейсе — ограничивает входящий трафик.&#10;Снимается автоматически при завершении. 0 или пусто = без лимита.">Лимит Mbit/s:</span>
        <input type="number" id="dl-bandwidth" class="dl-ctrl-input" value="" min="0" max="10000" step="5" placeholder="∞">
      </div>
      <div class="dl-ctrl-group">
        <span class="dl-ctrl-label" title="Автоматически загружать модели в S3 после скачивания.&#10;Требует настройки credentials.yaml (access_key_id, secret_access_key)&#10;и переменных окружения: S3_BUCKET, S3_ENDPOINT_URL, S3_REGION, S3_PREFIX.">Sync → S3:</span>
        <input type="checkbox" id="dl-sync-s3" class="dl-ctrl-checkbox">
      </div>
      <button class="dl-save-btn" id="dl-save-btn" onclick="saveSettings()" title="Сохранить параметры в models.yaml">Сохранить настройки</button>
      <button class="btn-secondary" id="dl-start-btn" onclick="dlStart()" title="Скачать все enabled модели">▶ Скачать enabled</button>
    </div>
    <button class="btn-secondary" id="dl-cancel-btn" onclick="dlCancel()" style="display:none;margin-left:auto">Отменить</button>
    <span class="status-msg" id="dl-status-msg"></span>
  </div>
  <div class="download-panel" id="download-panel" style="display:none">
    <div class="download-panel-header">
      <span>Загрузка:</span>
      <span class="dl-current" id="dl-current">—</span>
      <span id="dl-counter" style="font-size:0.78rem;color:var(--text-muted)"></span>
      <button class="dl-icon-btn" id="dl-expand-btn" onclick="dlToggleLog()" title="Развернуть / свернуть лог">⤢ Развернуть</button>
      <button class="dl-icon-btn cancel" id="dl-cancel-btn-panel" onclick="dlCancel()" style="display:none">✕ Отменить</button>
    </div>
    <div class="dl-progress-track">
      <div class="dl-progress-bar" id="dl-progress-bar"></div>
    </div>
    <div class="dl-log" id="dl-log"></div>
  </div>
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
          <th></th>
        </tr>
      </thead>
      <tbody id="models-body"></tbody>
    </table>
    <div id="empty-msg" style="display:none">Ничего не найдено</div>
  </main>
</div>

<!-- ═══════════════════════ TAB: HuggingFace поиск ═══════════════════════ -->
<div id="tab-hf-search" class="tab-panel">
  <div class="hf-panel">

    <div class="hf-search-form">
      <!-- Строка 1: основные поля поиска -->
      <div class="hf-search-row">
        <div class="hf-field">
          <label>Запрос</label>
          <input type="text" id="hf-query" placeholder="llama, qwen, deepseek..." onkeydown="if(event.key==='Enter')hfSearch()">
        </div>
        <div class="hf-field">
          <label>Автор</label>
          <input type="text" id="hf-author" placeholder="bartowski, unsloth..." onkeydown="if(event.key==='Enter')hfSearch()">
        </div>
        <div class="hf-field">
          <label>Тип задачи</label>
          <select id="hf-pipeline-tag">
            <option value="">— любой —</option>
            <option value="text-generation">Текстовая генерация (LLM)</option>
            <option value="text-to-image">Генерация изображений</option>
            <option value="image-to-text">Описание изображений</option>
            <option value="automatic-speech-recognition">Распознавание речи (ASR)</option>
            <option value="text-to-speech">Синтез речи (TTS)</option>
            <option value="feature-extraction">Эмбеддинги</option>
            <option value="sentence-similarity">Семантическое сходство</option>
            <option value="translation">Перевод</option>
            <option value="text-classification">Классификация текста</option>
            <option value="zero-shot-classification">Zero-shot классификация</option>
          </select>
        </div>
        <div class="hf-field">
          <label>Лимит</label>
          <input type="number" id="hf-limit" value="20" min="1" max="50">
        </div>
        <button class="btn-primary" id="hf-search-btn" onclick="hfSearch()" style="align-self:flex-end">Искать</button>
      </div>
      <!-- Строка 2: фильтры по файлам -->
      <div class="hf-filters-row">
        <div class="hf-filter-group hf-field-regex">
          <label>Regex файлов <span id="hf-regex-hint" style="font-size:0.65rem;color:var(--text-muted);margin-left:4px"></span></label>
          <input type="text" id="hf-file-regex" placeholder="Q4_K_M\.gguf$"
                 oninput="onRegexInput()" onkeydown="if(event.key==='Enter')hfSearch()">
          <div id="hf-regex-error" class="regex-error" style="display:none"></div>
          <div class="regex-presets">
            <span class="regex-chip" onclick="setRegexPreset('Q4_K_M\\.gguf$')"        title="Квантизация Q4_K_M">Q4_K_M</span>
            <span class="regex-chip" onclick="setRegexPreset('Q8_0\\.gguf$')"          title="Квантизация Q8_0">Q8_0</span>
            <span class="regex-chip" onclick="setRegexPreset('Q5_K_M\\.gguf$')"        title="Квантизация Q5_K_M">Q5_K_M</span>
            <span class="regex-chip" onclick="setRegexPreset('Q6_K\\.gguf$')"          title="Квантизация Q6_K">Q6_K</span>
            <span class="regex-chip" onclick="setRegexPreset('IQ4_XS\\.gguf$')"        title="iMatrix Q4 XS">IQ4_XS</span>
            <span class="regex-chip" onclick="setRegexPreset('IQ4_NL\\.gguf$')"        title="iMatrix Q4 NL">IQ4_NL</span>
            <span class="regex-chip" onclick="setRegexPreset('Q4_K_M\\.gguf$|Q8_0\\.gguf$')"  title="Q4_K_M или Q8_0">Q4|Q8</span>
            <span class="regex-chip" onclick="setRegexPreset('\\.gguf$')"              title="Все GGUF файлы">.gguf</span>
            <span class="regex-chip" onclick="setRegexPreset('\\.safetensors$')"       title="Safetensors файлы">.safetensors</span>
            <span class="regex-chip chip-clear" onclick="setRegexPreset('')"           title="Очистить поле">✕ сбросить</span>
          </div>
        </div>
        <div class="hf-filter-group">
          <label>Язык</label>
          <div class="filter-chips" id="lang-chips">
            <span class="filter-chip active" data-lang="" onclick="toggleLangChip(this)">Любой</span>
            <span class="filter-chip" data-lang="ru" onclick="toggleLangChip(this)">🇷🇺 ru</span>
            <span class="filter-chip" data-lang="en" onclick="toggleLangChip(this)">🇬🇧 en</span>
            <span class="filter-chip" data-lang="zh" onclick="toggleLangChip(this)">🇨🇳 zh</span>
            <span class="filter-chip" data-lang="ja" onclick="toggleLangChip(this)">🇯🇵 ja</span>
            <span class="filter-chip" data-lang="de" onclick="toggleLangChip(this)">🇩🇪 de</span>
            <span class="filter-chip" data-lang="fr" onclick="toggleLangChip(this)">🇫🇷 fr</span>
          </div>
        </div>
        <div class="hf-filter-group">
          <label>Размер файла, МБ</label>
          <div class="size-range">
            <input type="number" id="hf-min-size" placeholder="от" min="0" step="100" oninput="hfApplySizeFilter()">
            <span>–</span>
            <input type="number" id="hf-max-size" placeholder="до" min="0" step="100" oninput="hfApplySizeFilter()">
            <span style="color:var(--text-muted);font-size:0.72rem">МБ</span>
          </div>
        </div>
      </div>
    </div>

    <div id="hf-search-status"></div>

    <div id="hf-results-wrapper">
      <div class="hf-results-controls">
        <span id="hf-results-count"></span>
        <button class="btn-primary" id="hf-add-btn" onclick="hfShowAddForm()" disabled>
          Добавить выбранные
        </button>
      </div>

      <table>
        <thead>
          <tr>
            <th><input type="checkbox" class="select-cb" id="hf-select-all" onchange="hfToggleAll(this)"></th>
            <th>Репозиторий</th>
            <th>Файлы</th>
            <th>Загрузки</th>
            <th>Теги</th>
          </tr>
        </thead>
        <tbody id="hf-results-body"></tbody>
      </table>
      <div id="hf-empty-msg">Ничего не найдено — попробуйте другие параметры поиска</div>

      <div class="hf-add-form" id="hf-add-form">
        <h3 id="hf-add-title">Добавить выбранные модели</h3>
        <div class="hf-add-scroll">
          <table class="hf-add-table">
            <thead>
              <tr>
                <th>Модель / Файл</th>
                <th class="col-add-dir">Папка (dest_dir)</th>
                <th class="col-add-tags">Теги</th>
                <th class="col-add-vram">VRAM GB</th>
                <th class="col-add-desc">Описание</th>
              </tr>
            </thead>
            <tbody id="hf-add-rows"></tbody>
          </table>
        </div>
        <div class="hf-add-actions">
          <label class="toggle-label">
            <input type="checkbox" id="add-enabled"> Включить сразу (enabled: true)
          </label>
          <button class="btn-primary" id="add-confirm-btn" onclick="hfConfirmAdd()">Добавить в models.yaml</button>
          <button class="btn-secondary" onclick="hfCancelAdd()">Отмена</button>
          <span class="status-msg" id="add-status"></span>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════════════════
//  Shared helpers
// ═══════════════════════════════════════════════════════════════════

function fmtSize(bytes) {
  if (!bytes) return '—';
  const u = ['B','KB','MB','GB','TB'];
  let v = bytes, i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return v.toFixed(1) + ' ' + u[i];
}

function fmtNum(n) {
  if (!n) return '0';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function switchTab(btn) {
  const name = btn.dataset.tab;
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}

// ═══════════════════════════════════════════════════════════════════
//  Tab: Мои модели
// ═══════════════════════════════════════════════════════════════════

let allModels = [];
let changes = {};
let activeTags = new Set();
let collapseState = {}; // group key ('l1:llm', 'l2:llm/qwen') → collapsed bool

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
  if (activeTags.has(tag)) { activeTags.delete(tag); btn.classList.remove('active'); }
  else                      { activeTags.add(tag);    btn.classList.add('active'); }
  applyFilters();
}

function applyFilters() {
  const q = document.getElementById('search').value.toLowerCase().trim();
  const onlyEnabled = document.getElementById('only-enabled').checked;
  const onlyDl = document.getElementById('only-downloaded').checked;
  const tbody = document.getElementById('models-body');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  let visible = 0;

  // Step 1: mark each model row (has data-idx) with data-fpass
  rows.forEach(row => {
    if (row.dataset.idx === undefined || row.classList.contains('edit-row')) return;
    const m = allModels[parseInt(row.dataset.idx)];
    if (!m) return;
    let show = true;
    if (q && !(m.repo_id + ' ' + m.filename + ' ' + (m.description || '')).toLowerCase().includes(q)) show = false;
    // Check enabled state — prefer unsaved change if present
    if (show && onlyEnabled) {
      const key = m.repo_id + '::' + m.filename;
      const effectiveEnabled = key in changes ? changes[key] : m.enabled;
      if (!effectiveEnabled) show = false;
    }
    if (show && onlyDl && !m.downloaded) show = false;
    if (show && activeTags.size > 0) {
      const mt = new Set(m.tags || []);
      for (const t of activeTags) { if (!mt.has(t)) { show = false; break; } }
    }
    row.dataset.fpass = show ? '1' : '0';
    if (show) visible++;
  });

  // Step 2: apply visibility for model rows and group headers
  rows.forEach(row => {
    if (row.classList.contains('edit-row')) return;
    const g = row.dataset.group;
    if (g) {
      // Group header row
      if (g.startsWith('l1:')) {
        const l1 = g.slice(3);
        const hasVisible = rows.some(r => r.dataset.l1 === l1 && !r.dataset.group && r.dataset.fpass === '1');
        row.style.display = hasVisible ? '' : 'none';
      } else if (g.startsWith('l2:')) {
        const l2 = g.slice(3);
        const l1 = row.dataset.l1;
        if (collapseState['l1:' + l1]) { row.style.display = 'none'; return; }
        const hasVisible = rows.some(r => r.dataset.l2 === l2 && r.dataset.fpass === '1');
        row.style.display = hasVisible ? '' : 'none';
      }
    } else if (row.dataset.idx !== undefined) {
      // Model row
      if (row.dataset.fpass === '0') { row.style.display = 'none'; return; }
      const l1 = row.dataset.l1;
      const l2 = row.dataset.l2;
      const l1c = !!collapseState['l1:' + l1];
      const l2c = l2 ? !!collapseState['l2:' + l2] : false;
      row.style.display = (l1c || l2c) ? 'none' : '';
    }
  });

  document.getElementById('empty-msg').style.display = visible === 0 ? '' : 'none';
  document.getElementById('stats').textContent =
    `Показано: ${visible} из ${allModels.length} моделей` +
    (Object.keys(changes).length > 0 ? ` · Изменений: ${Object.keys(changes).length}` : '');
}

function onEnabledChange(cb, m) {
  const key = modelKey(m);
  const newEnabled = cb.checked;
  if (newEnabled === m.enabled) delete changes[key];
  else changes[key] = newEnabled;
  cb.closest('tr').classList.toggle('row-disabled', !newEnabled);
  document.getElementById('save-btn').disabled = Object.keys(changes).length === 0;
  applyFilters();
}

function renderModelRow(m, i) {
  const key = modelKey(m);
  const enabled = key in changes ? changes[key] : m.enabled;
  const tr = document.createElement('tr');
  tr.dataset.idx = i;
  tr.dataset.fpass = '1';
  const parts = (m.dest_dir || 'misc').split('/');
  tr.dataset.l1 = parts[0];
  if (parts.length > 1) tr.dataset.l2 = m.dest_dir;
  if (!enabled) tr.classList.add('row-disabled');

  const tdCb = document.createElement('td');
  const cb = document.createElement('input');
  cb.type = 'checkbox'; cb.className = 'enabled-cb'; cb.checked = enabled;
  cb.onchange = () => onEnabledChange(cb, m);
  tdCb.appendChild(cb);

  const tdStatus = document.createElement('td');
  const dot = document.createElement('span');
  dot.className = 'status-dot ' + (m.downloaded ? 'dot-downloaded' : 'dot-notfound');
  dot.title = m.downloaded ? 'Скачана' : 'Не скачана';
  tdStatus.appendChild(dot);

  const tdName = document.createElement('td');
  tdName.innerHTML =
    `<div class="cell-repo"><a href="https://huggingface.co/${escHtml(m.repo_id)}" target="_blank">${escHtml(m.repo_id)}</a>${m.gated ? '<span class="gated-badge">GATED</span>' : ''}</div>` +
    `<div class="cell-filename">${escHtml(m.filename)}</div>`;

  const tdTags = document.createElement('td');
  tdTags.innerHTML = '<div class="tags-cell">' +
    (m.tags || []).map(t => `<span class="tag">${escHtml(t)}</span>`).join('') + '</div>';

  const tdVram = document.createElement('td');
  tdVram.className = 'vram';
  tdVram.textContent = m.vram_gb ? m.vram_gb + ' GB' : '—';

  const tdSize = document.createElement('td');
  tdSize.className = 'cell-size ' + (m.downloaded ? 'downloaded' : 'notfound');
  tdSize.textContent = m.downloaded ? fmtSize(m.disk_size_bytes) : '—';

  const tdDesc = document.createElement('td');
  tdDesc.className = 'desc';
  tdDesc.textContent = m.description || '';

  const tdActions = document.createElement('td');
  tdActions.className = 'td-actions';
  const btnEdit = document.createElement('button');
  btnEdit.className = 'btn-edit';
  btnEdit.textContent = '✎';
  btnEdit.title = 'Редактировать атрибуты';
  btnEdit.onclick = () => modelEditToggle(m, tr, btnEdit);
  const btnDel = document.createElement('button');
  btnDel.className = 'btn-delete';
  btnDel.textContent = '✕';
  btnDel.title = m.downloaded
    ? 'Удалить файл и убрать из models.yaml'
    : 'Убрать из models.yaml';
  btnDel.onclick = () => deleteModel(m);
  tdActions.append(btnEdit, ' ', btnDel);

  tr.append(tdCb, tdStatus, tdName, tdTags, tdVram, tdSize, tdDesc, tdActions);
  return tr;
}

function toggleGroup(key) {
  collapseState[key] = !collapseState[key];
  const collapsed = collapseState[key];
  const tbody = document.getElementById('models-body');
  const rows = Array.from(tbody.querySelectorAll('tr'));

  // Update toggle icon on the clicked header
  const headerRow = tbody.querySelector(`tr[data-group="${key}"]`);
  if (headerRow) headerRow.querySelector('.group-toggle').textContent = collapsed ? '▶' : '▼';

  if (key.startsWith('l1:')) {
    const l1 = key.slice(3);
    rows.forEach(row => {
      if (row.dataset.group === key || row.classList.contains('edit-row')) return;
      if (row.dataset.l1 !== l1) return;
      if (collapsed) {
        row.style.display = 'none';
      } else {
        if (row.dataset.group) {
          // L2 header: show (L1 is open)
          row.style.display = '';
        } else {
          // Model row: respect L2 collapse and filter
          const l2 = row.dataset.l2;
          const l2c = l2 ? !!collapseState['l2:' + l2] : false;
          row.style.display = (l2c || row.dataset.fpass === '0') ? 'none' : '';
        }
      }
    });
  } else if (key.startsWith('l2:')) {
    const l2 = key.slice(3);
    rows.forEach(row => {
      if (row.classList.contains('edit-row')) return;
      if (row.dataset.l2 !== l2) return;
      row.style.display = (collapsed || row.dataset.fpass === '0') ? 'none' : '';
    });
  }
}

function renderModels(models) {
  const tbody = document.getElementById('models-body');
  tbody.innerHTML = '';

  // Build tree: l1 → { direct: [idx,...], sub: { l2: [idx,...] }, subOrder: [] }
  const tree = {};
  const treeOrder = [];
  models.forEach((m, i) => {
    const parts = (m.dest_dir || 'misc').split('/');
    const l1 = parts[0];
    if (!tree[l1]) { tree[l1] = { direct: [], sub: {}, subOrder: [] }; treeOrder.push(l1); }
    if (parts.length > 1) {
      const l2 = m.dest_dir;
      if (!tree[l1].sub[l2]) { tree[l1].sub[l2] = []; tree[l1].subOrder.push(l2); }
      tree[l1].sub[l2].push(i);
    } else {
      tree[l1].direct.push(i);
    }
  });

  treeOrder.forEach(l1 => {
    const node = tree[l1];
    const l1Collapsed = !!collapseState['l1:' + l1];
    const totalCount = node.direct.length + Object.values(node.sub).reduce((a, v) => a + v.length, 0);

    // L1 group header
    const l1Tr = document.createElement('tr');
    l1Tr.className = 'group-row-l1';
    l1Tr.dataset.group = 'l1:' + l1;
    const l1Td = document.createElement('td');
    l1Td.colSpan = 8;
    l1Td.innerHTML =
      `<span class="group-toggle">${l1Collapsed ? '▶' : '▼'}</span>` +
      `${escHtml(l1)}<span class="group-count">${totalCount}</span>`;
    l1Tr.appendChild(l1Td);
    l1Tr.onclick = () => toggleGroup('l1:' + l1);
    tbody.appendChild(l1Tr);

    // L2 sub-groups
    node.subOrder.forEach(l2 => {
      const l2Collapsed = !!collapseState['l2:' + l2];
      const l2Tr = document.createElement('tr');
      l2Tr.className = 'group-row-l2';
      l2Tr.dataset.group = 'l2:' + l2;
      l2Tr.dataset.l1 = l1;
      l2Tr.style.display = l1Collapsed ? 'none' : '';
      const l2Td = document.createElement('td');
      l2Td.colSpan = 8;
      const l2Label = l2.split('/').slice(1).join('/');
      l2Td.innerHTML =
        `<span class="group-toggle">${l2Collapsed ? '▶' : '▼'}</span>` +
        `${escHtml(l2Label)}<span class="group-count">${node.sub[l2].length}</span>`;
      l2Tr.appendChild(l2Td);
      l2Tr.onclick = () => toggleGroup('l2:' + l2);
      tbody.appendChild(l2Tr);

      node.sub[l2].forEach(idx => {
        const mTr = renderModelRow(models[idx], idx);
        mTr.style.display = (l1Collapsed || l2Collapsed) ? 'none' : '';
        tbody.appendChild(mTr);
      });
    });

    // Direct (no L2) model rows
    node.direct.forEach(idx => {
      const mTr = renderModelRow(models[idx], idx);
      mTr.style.display = l1Collapsed ? 'none' : '';
      tbody.appendChild(mTr);
    });
  });
}

// ── Inline model editing ───────────────────────────────────────────────────────

function modelEditToggle(m, mainTr, btn) {
  const existingEdit = mainTr.nextElementSibling;
  if (existingEdit && existingEdit.classList.contains('edit-row')) {
    existingEdit.remove();
    btn.classList.remove('active');
    return;
  }
  // Close any other open edit rows
  document.querySelectorAll('.edit-row').forEach(r => r.remove());
  document.querySelectorAll('.btn-edit.active').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  const editTr = document.createElement('tr');
  editTr.className = 'edit-row';

  const td = document.createElement('td');
  td.colSpan = 8;

  const form = document.createElement('div');
  form.className = 'model-edit-form';

  // Tags
  form.appendChild(_editField('Теги (через пробел)', 'tags', 'text', (m.tags || []).join(' '), 'w-tags'));
  // VRAM
  form.appendChild(_editField('VRAM (GB)', 'vram', 'number', m.vram_gb || 0, 'w-vram'));
  // Description
  form.appendChild(_editField('Описание', 'desc', 'text', m.description || '', 'w-desc'));
  // dest_dir
  const dirWrap = _editField('Папка (dest_dir)', 'dir', 'text', m.dest_dir || 'misc', 'w-dir');
  const dirInp = dirWrap.querySelector('input');
  const dirErr = document.createElement('div');
  dirErr.className = 'dir-error-hint';
  dirInp.insertAdjacentElement('afterend', dirErr);
  dirInp.addEventListener('input', () => applyDirValidation(dirInp, dirErr));
  if (m.downloaded) {
    const note = document.createElement('div');
    note.className = 'dir-move-note';
    note.textContent = '⚠ файл будет перемещён при изменении';
    dirWrap.appendChild(note);
  }
  form.appendChild(dirWrap);
  // Gated
  const gatedWrap = document.createElement('div');
  gatedWrap.className = 'model-edit-field';
  const gatedLbl = document.createElement('label');
  gatedLbl.textContent = 'Gated (требует токен)';
  const gatedCb = document.createElement('input');
  gatedCb.type = 'checkbox'; gatedCb.className = 'enabled-cb'; gatedCb.dataset.field = 'gated';
  gatedCb.checked = !!m.gated;
  gatedWrap.append(gatedLbl, gatedCb);
  form.appendChild(gatedWrap);

  // Actions
  const actWrap = document.createElement('div');
  actWrap.className = 'model-edit-actions';
  const btnSave = document.createElement('button');
  btnSave.className = 'btn-primary btn-sm'; btnSave.textContent = 'Сохранить';
  btnSave.onclick = () => modelEditSave(m, editTr, mainTr, btn);
  const btnCancel = document.createElement('button');
  btnCancel.className = 'btn-secondary btn-sm'; btnCancel.textContent = 'Отмена';
  btnCancel.onclick = () => { editTr.remove(); btn.classList.remove('active'); };
  const statusEl = document.createElement('span');
  statusEl.className = 'status-msg'; statusEl.dataset.role = 'edit-status';
  actWrap.append(btnSave, btnCancel, statusEl);
  form.appendChild(actWrap);

  td.appendChild(form);
  editTr.appendChild(td);
  mainTr.insertAdjacentElement('afterend', editTr);
  form.querySelector('input').focus();
}

function _editField(label, fieldName, type, value, widthCls) {
  const wrap = document.createElement('div');
  wrap.className = 'model-edit-field';
  const lbl = document.createElement('label'); lbl.textContent = label;
  const inp = document.createElement('input');
  inp.type = type; inp.value = value;
  inp.className = 'model-edit-input ' + widthCls;
  inp.dataset.field = fieldName;
  wrap.append(lbl, inp);
  return wrap;
}

async function modelEditSave(m, editTr, mainTr, btn) {
  const update = { repo_id: m.repo_id, filename: m.filename };
  editTr.querySelectorAll('input[data-field]').forEach(inp => {
    const f = inp.dataset.field;
    if (f === 'tags')  update.tags = inp.value.trim() ? inp.value.trim().split(/\s+/) : [];
    if (f === 'vram')  update.vram_gb = parseFloat(inp.value) || 0;
    if (f === 'desc')  update.description = inp.value.trim();
    if (f === 'dir')   update.dest_dir = inp.value.trim() || 'misc';
    if (f === 'gated') update.gated = inp.checked;
  });

  // Validate dest_dir before saving
  const dirInp = editTr.querySelector('input[data-field="dir"]');
  const dirErr = dirInp ? dirInp.nextElementSibling : null;
  if (dirInp && !applyDirValidation(dirInp, dirErr)) {
    const st = editTr.querySelector('[data-role=edit-status]');
    if (st) { st.className = 'status-msg error'; st.textContent = 'Исправьте ошибки перед сохранением'; }
    return;
  }

  const statusEl = editTr.querySelector('[data-role=edit-status]');
  const btnSave = editTr.querySelector('.btn-primary');
  btnSave.disabled = true;
  statusEl.className = 'status-msg'; statusEl.textContent = 'Сохранение…';

  try {
    const resp = await fetch('/api/save', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates: [update] }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || 'HTTP ' + resp.status);
    editTr.remove();
    btn.classList.remove('active');
    loadModels();
  } catch (e) {
    statusEl.className = 'status-msg err';
    statusEl.textContent = 'Ошибка: ' + e.message;
    btnSave.disabled = false;
  }
}

async function deleteModel(m) {
  const hasFile = m.downloaded;
  const msg = hasFile
    ? `Удалить файл «${m.filename}» с диска и убрать запись из models.yaml?`
    : `Убрать «${m.filename}» из models.yaml?\n(Файл не найден на диске)`;
  if (!confirm(msg)) return;
  try {
    const resp = await fetch('/api/delete', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id: m.repo_id, filename: m.filename }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'HTTP ' + resp.status);
    loadModels();
  } catch (e) {
    alert('Ошибка удаления: ' + e.message);
  }
}

async function loadModels() {
  try {
    const resp = await fetch('/api/models');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    allModels = await resp.json();
    renderTagFilter(collectTags(allModels));
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
  msg.className = 'status-msg'; msg.style.display = 'none';
  const updates = Object.entries(changes).map(([key, enabled]) => {
    const [repo_id, filename] = key.split('||');
    return { repo_id, filename, enabled };
  });
  try {
    const resp = await fetch('/api/save', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: 'HTTP ' + resp.status }));
      throw new Error(err.error || 'HTTP ' + resp.status);
    }
    for (const [key, enabled] of Object.entries(changes)) {
      const [repo_id, filename] = key.split('||');
      const m = allModels.find(x => x.repo_id === repo_id && x.filename === filename);
      if (m) m.enabled = enabled;
    }
    changes = {};
    msg.className = 'status-msg ok'; msg.textContent = 'Сохранено!'; msg.style.display = '';
    setTimeout(() => { msg.style.display = 'none'; }, 3000);
    applyFilters();
  } catch (e) {
    msg.className = 'status-msg err'; msg.textContent = 'Ошибка: ' + e.message; msg.style.display = '';
    btn.disabled = Object.keys(changes).length === 0;
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Regex validator helpers
// ═══════════════════════════════════════════════════════════════════

function onRegexInput() {
  const input  = document.getElementById('hf-file-regex');
  const errEl  = document.getElementById('hf-regex-error');
  const hint   = document.getElementById('hf-regex-hint');
  const val    = input.value.trim();
  if (!val) {
    input.classList.remove('regex-valid', 'regex-invalid');
    errEl.style.display = 'none';
    hint.textContent = '';
    return true;
  }
  try {
    const re = new RegExp(val, 'i');
    input.classList.add('regex-valid');
    input.classList.remove('regex-invalid');
    errEl.style.display = 'none';
    // Count how many sample names match the regex for quick feedback
    const samples = [
      'model-Q4_K_M.gguf','model-Q8_0.gguf','model-Q5_K_M.gguf','model-Q6_K.gguf',
      'model-IQ4_XS.gguf','model-IQ4_NL.gguf','model-F16.gguf',
      'model.safetensors','model.bin','model-Q4_0.gguf',
    ];
    const matches = samples.filter(s => re.test(s));
    hint.textContent = matches.length ? `✓ совпадает: ${matches.join(', ')}` : '✓ нет совпадений с примерами';
    hint.style.color = matches.length ? 'var(--green)' : 'var(--text-muted)';
    return true;
  } catch(e) {
    input.classList.add('regex-invalid');
    input.classList.remove('regex-valid');
    errEl.textContent = '✗ ' + e.message;
    errEl.style.display = '';
    hint.textContent = '';
    return false;
  }
}

function setRegexPreset(pattern) {
  const input = document.getElementById('hf-file-regex');
  input.value = pattern;
  onRegexInput();
  input.focus();
}

function toggleLangChip(chip) {
  document.querySelectorAll('#lang-chips .filter-chip').forEach(c => c.classList.remove('active'));
  chip.classList.add('active');
}

// ═══════════════════════════════════════════════════════════════════
//  Tab: Поиск HuggingFace
// ═══════════════════════════════════════════════════════════════════

let hfResults = [];
let hfSelected = new Map(); // "repo_id||filename" → {repo_id, filename, gated, tags, description, pipeline_tag}

async function hfSearch() {
  const q           = document.getElementById('hf-query').value.trim();
  const author      = document.getElementById('hf-author').value.trim();
  const fileRegex   = document.getElementById('hf-file-regex').value.trim();
  const pipelineTag = document.getElementById('hf-pipeline-tag').value;
  const activeLangChip = document.querySelector('#lang-chips .filter-chip.active');
  const language    = activeLangChip ? activeLangChip.dataset.lang : '';
  const limit       = Math.min(50, Math.max(1, parseInt(document.getElementById('hf-limit').value) || 20));

  if (!q && !author && !fileRegex && !pipelineTag && !language) {
    setHfStatus('Укажите хотя бы один параметр поиска', true);
    return;
  }

  // Validate regex before sending to server
  if (fileRegex && !onRegexInput()) {
    setHfStatus('Исправьте regex перед поиском', true);
    return;
  }

  const btn = document.getElementById('hf-search-btn');
  btn.disabled = true;
  hfSelected.clear();
  document.getElementById('hf-add-btn').disabled = true;
  document.getElementById('hf-add-form').style.display = 'none';
  document.getElementById('hf-results-wrapper').style.display = 'none';

  const params = new URLSearchParams();
  if (q) params.set('q', q);
  if (author) params.set('author', author);
  if (fileRegex) params.set('file_regex', fileRegex);
  if (pipelineTag) params.set('pipeline_tag', pipelineTag);
  if (language) params.set('language', language);
  params.set('limit', limit);

  setHfStatus('<span class="spinner"></span>&nbsp;Поиск на HuggingFace...', false);

  try {
    const resp = await fetch('/api/search?' + params.toString());
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'HTTP ' + resp.status);
    hfResults = data.results || [];
    renderHfResults(hfResults);
    const note = hfResults.length !== data.count
      ? ` (показано ${hfResults.length} из ${data.count})`
      : ` (${data.count})`;
    setHfStatus('Найдено репозиториев: ' + data.count + (data.count !== hfResults.length ? `, строк: ${hfResults.reduce((s,r)=>s+(r.files.length||1),0)}` : ''), false);
  } catch (e) {
    setHfStatus('Ошибка: ' + escHtml(e.message), true);
  } finally {
    btn.disabled = false;
  }
}

function setHfStatus(html, isErr) {
  const el = document.getElementById('hf-search-status');
  el.innerHTML = html;
  el.className = isErr ? 'err' : '';
}

function renderHfResults(results) {
  const tbody = document.getElementById('hf-results-body');
  tbody.innerHTML = '';
  document.getElementById('hf-select-all').checked = false;
  document.getElementById('hf-select-all').indeterminate = false;
  document.getElementById('hf-empty-msg').style.display = results.length === 0 ? '' : 'none';
  document.getElementById('hf-results-wrapper').style.display = 'block';
  document.getElementById('hf-results-count').textContent = '';

  results.forEach(r => {
    // files is now [{name, size_bytes}, ...] or [] → normalise to [{name, size_bytes}] or [null]
    const rawFiles = r.files && r.files.length > 0 ? r.files : [null];
    rawFiles.forEach((fileObj, fi) => {
      const fname = fileObj ? (typeof fileObj === 'object' ? fileObj.name : fileObj) : null;
      const fsize = fileObj && typeof fileObj === 'object' ? fileObj.size_bytes : null;
      const key = r.repo_id + '||' + (fname || '');
      const tr = document.createElement('tr');
      tr.dataset.key = key;
      // Store size_bytes on the row so the filter can read it
      if (fsize != null) tr.dataset.sizeBytes = fsize;

      // checkbox — не рендерим для строк без файла (нет смысла выбирать)
      const tdCb = document.createElement('td');
      if (fname) {
        const cb = document.createElement('input');
        cb.type = 'checkbox'; cb.className = 'select-cb'; cb.dataset.key = key;
        cb.onchange = () => hfSelectionChange(cb, r, fname);
        tdCb.appendChild(cb);
      }
      tr.appendChild(tdCb);

      // repo — только на первой строке файлов репозитория
      if (fi === 0) {
        const tdRepo = document.createElement('td');
        tdRepo.className = 'cell-hf-repo';
        if (rawFiles.length > 1) tdRepo.rowSpan = rawFiles.length;
        const pipelineBadge = r.pipeline_tag
          ? `<span class="hf-pipeline-badge">${escHtml(r.pipeline_tag)}</span>` : '';
        const likesHtml = r.likes > 0
          ? `<span class="hf-likes">♥ ${fmtNum(r.likes)}</span>` : '';
        const descHtml = r.description
          ? `<span class="hf-desc">${escHtml(r.description)}</span>` : '';
        tdRepo.innerHTML =
          `<a href="https://huggingface.co/${escHtml(r.repo_id)}" target="_blank">${escHtml(r.repo_id)}</a>` +
          (r.gated ? '<span class="gated-badge">GATED</span>' : '') +
          (pipelineBadge || likesHtml || descHtml
            ? `<div class="cell-hf-meta">${pipelineBadge}${likesHtml}${descHtml}</div>` : '');
        tr.appendChild(tdRepo);
      }

      // filename + size (size on new line for readability)
      const tdFile = document.createElement('td');
      tdFile.className = 'cell-hf-files';
      if (fname) {
        const sizeHtml = fsize != null
          ? `<span class="file-size" style="display:block;margin-top:2px">${fmtSize(fsize)}</span>`
          : '';
        tdFile.innerHTML = `<span class="file-item">${escHtml(fname)}${sizeHtml}</span>`;
      } else {
        tdFile.innerHTML = `<span style="color:var(--text-muted);font-style:italic">нет файлов по фильтру</span>`;
      }
      tr.appendChild(tdFile);

      // downloads + likes — только на первой строке
      if (fi === 0) {
        const tdDl = document.createElement('td');
        tdDl.className = 'downloads';
        if (rawFiles.length > 1) tdDl.rowSpan = rawFiles.length;
        tdDl.textContent = fmtNum(r.downloads);
        tr.appendChild(tdDl);

        const tdTags = document.createElement('td');
        if (rawFiles.length > 1) tdTags.rowSpan = rawFiles.length;
        tdTags.innerHTML = '<div class="tags-cell">' +
          (r.tags || []).slice(0, 6).map(t => `<span class="tag">${escHtml(t)}</span>`).join('') +
          '</div>';
        tr.appendChild(tdTags);
      }

      tbody.appendChild(tr);
    });
  });

  // Apply size filter after rendering
  hfApplySizeFilter();
}

function hfSelectionChange(cb, r, fname) {
  const key = r.repo_id + '||' + (fname || '');
  if (cb.checked) {
    hfSelected.set(key, { repo_id: r.repo_id, filename: fname || '', gated: r.gated, tags: r.tags || [], description: r.description || '', pipeline_tag: r.pipeline_tag || '' });
  } else {
    hfSelected.delete(key);
  }
  document.getElementById('hf-add-btn').disabled = hfSelected.size === 0;
  const allCbs = document.querySelectorAll('#hf-results-body .select-cb');
  const checked = document.querySelectorAll('#hf-results-body .select-cb:checked').length;
  document.getElementById('hf-select-all').checked = allCbs.length > 0 && checked === allCbs.length;
  document.getElementById('hf-select-all').indeterminate = checked > 0 && checked < allCbs.length;
}

function hfToggleAll(masterCb) {
  document.querySelectorAll('#hf-results-body .select-cb').forEach(cb => {
    if (cb.closest('tr').style.display === 'none') return; // skip filtered-out rows
    cb.checked = masterCb.checked;
    const key = cb.dataset.key;
    const [repo_id, fname] = key.split('||');
    const r = hfResults.find(x => x.repo_id === repo_id);
    if (!r) return;
    if (masterCb.checked) {
      hfSelected.set(key, { repo_id, filename: fname, gated: r.gated, tags: r.tags || [], description: r.description || '', pipeline_tag: r.pipeline_tag || '' });
    } else {
      hfSelected.delete(key);
    }
  });
  document.getElementById('hf-add-btn').disabled = hfSelected.size === 0;
}

function hfApplySizeFilter() {
  const minMbVal = document.getElementById('hf-min-size').value.trim();
  const maxMbVal = document.getElementById('hf-max-size').value.trim();
  const MB = 1024 * 1024;
  const minBytes = minMbVal !== '' ? parseFloat(minMbVal) * MB : null;
  const maxBytes = maxMbVal !== '' ? parseFloat(maxMbVal) * MB : null;
  const hasFilter = (minBytes !== null && !isNaN(minBytes)) || (maxBytes !== null && !isNaN(maxBytes));
  document.querySelectorAll('#hf-results-body tr').forEach(tr => {
    const sizeBytes = tr.dataset.sizeBytes != null ? parseFloat(tr.dataset.sizeBytes) : null;
    let hide = false;
    // Hide only rows where size is known AND violates the filter
    if (hasFilter && sizeBytes != null) {
      if (minBytes !== null && !isNaN(minBytes) && sizeBytes < minBytes) hide = true;
      if (maxBytes !== null && !isNaN(maxBytes) && sizeBytes > maxBytes) hide = true;
    }
    tr.style.display = hide ? 'none' : '';
    if (hide) {
      const cb = tr.querySelector('.select-cb');
      if (cb && cb.checked) {
        cb.checked = false;
        hfSelected.delete(cb.dataset.key);
      }
    }
  });
  document.getElementById('hf-add-btn').disabled = hfSelected.size === 0;
}

// ── helpers for add form pre-fill ─────────────────────────────────────────────

// ── dest_dir validation ────────────────────────────────────────────────────────
// Allowed: 1–3 segments, each [a-z0-9][a-z0-9_-]*, separated by /
// Examples: misc  llm  llm/qwen  llm/qwen/chat
const _DIR_RE = /^[a-z0-9][a-z0-9_-]*(?:\/[a-z0-9][a-z0-9_-]*){0,2}$/;

function validateDestDir(val) {
  if (!val || !val.trim()) return 'Обязательное поле (мин. 1 уровень, напр. misc)';
  const v = val.trim();
  if (!_DIR_RE.test(v))
    return 'Формат: a-z/0-9/_/- · от 1 до 3 уровней через /  (напр. llm  llm/qwen  llm/qwen/chat)';
  return null;
}

// Validates input in-place: auto-lowercases, replaces \\ → /, shows/hides hint.
// Returns true if valid (or empty).
function applyDirValidation(input, hintEl) {
  input.value = input.value.replace(/\\/g, '/').replace(/[A-Z]/g, c => c.toLowerCase());
  const err = validateDestDir(input.value);
  if (err) {
    input.classList.add('input-dir-invalid');
    if (hintEl) { hintEl.textContent = err; hintEl.style.display = ''; }
  } else {
    input.classList.remove('input-dir-invalid');
    if (hintEl) { hintEl.textContent = ''; hintEl.style.display = 'none'; }
  }
  return !err;
}

function estimateVram(fname, repoId) {
  // Estimate VRAM (GB) from filename + model size string
  const hay = ((fname || '') + '-' + (repoId || '')).toUpperCase();
  const sizeM = hay.match(/[-_](\d+(?:\.\d+)?)[B](?:[-_A-Z]|$)/);
  if (!sizeM) return 0;
  const pb = parseFloat(sizeM[1]);
  const quantMap = [
    ['Q2_K', 3.0], ['Q3_K_S', 3.5], ['Q3_K_M', 3.5], ['Q3_K_L', 4.0],
    ['Q4_0', 4.0], ['IQ4_XS', 4.0], ['Q4_K_S', 4.0], ['Q4_K_M', 4.5], ['Q4_K', 4.5],
    ['Q5_0', 5.0], ['Q5_K_S', 5.0], ['Q5_K_M', 5.5], ['Q5_K', 5.5],
    ['Q6_K', 6.5], ['Q8_0', 8.0], ['F16', 16.0], ['BF16', 16.0],
  ];
  let bpw = 4.5;
  for (const [q, b] of quantMap) { if (hay.includes(q)) { bpw = b; break; } }
  return Math.max(1, Math.round(pb * bpw / 8 + 0.5));
}

function suggestDestDir(repoId, pipelineTag, tags) {
  // Auto-suggest dest_dir from model type, name, language
  const lower = (repoId || '').toLowerCase();
  const t = (tags || []).map(s => s.toLowerCase());
  const pipe = (pipelineTag || '').toLowerCase();

  // Embeddings
  if (pipe === 'sentence-similarity' || pipe === 'feature-extraction' ||
      t.includes('embeddings') || lower.includes('embed'))
    return 'embeddings';

  // Image generation
  if (pipe === 'text-to-image' || pipe === 'image-to-image' || t.includes('image_gen')) {
    if (lower.includes('flux'))   return 'image_gen/flux';
    if (lower.includes('sdxl') || lower.includes('stable-diffusion')) return 'image_gen/sdxl';
    return 'image_gen';
  }

  // LLM variants
  if (pipe === 'text-generation' || pipe === 'text2text-generation' || t.includes('llm')) {
    if (t.includes('russian') || lower.includes('saiga') || lower.includes('t-lite') ||
        lower.includes('gigachat') || lower.includes('ru-'))
      return 'llm/russian';
    if (t.includes('code') || lower.includes('coder') || lower.includes('code'))
      return 'llm/code';
    // Model family
    if (lower.includes('llama'))      return 'llm/llama';
    if (lower.includes('qwen'))       return 'llm/qwen';
    if (lower.includes('deepseek'))   return 'llm/deepseek';
    if (lower.includes('phi'))        return 'llm/phi';
    if (lower.includes('mistral') || lower.includes('mixtral')) return 'llm/mistral';
    if (lower.includes('gemma'))      return 'llm/gemma';
    if (lower.includes('falcon'))     return 'llm/falcon';
    return 'llm';
  }
  return 'misc';
}

function hfShowAddForm() {
  const n = hfSelected.size;
  const word = n === 1 ? 'модель' : n < 5 ? 'модели' : 'моделей';
  document.getElementById('hf-add-title').textContent = `Добавить ${n} ${word} в models.yaml`;
  document.getElementById('add-status').className = 'status-msg';
  document.getElementById('add-status').style.display = 'none';

  // Build per-model editable rows
  const tbody = document.getElementById('hf-add-rows');
  tbody.innerHTML = '';
  for (const [key, item] of hfSelected.entries()) {
    const autoDir  = suggestDestDir(item.repo_id, item.pipeline_tag, item.tags || []);
    const autoTags = (item.tags || []).join(' ');
    const autoVram = estimateVram(item.filename, item.repo_id);
    const autoDesc = item.description || '';

    const tr = document.createElement('tr');
    tr.dataset.key = key;

    // Модель / файл (read-only display)
    const tdM = document.createElement('td');
    tdM.className = 'cell-add-model';
    const short = item.repo_id.split('/').pop();
    tdM.innerHTML =
      `<div class="add-repo" title="${escHtml(item.repo_id)}">${escHtml(short)}</div>` +
      `<div class="add-fname">${escHtml(item.filename)}</div>`;

    // dest_dir
    const tdDir = document.createElement('td');
    const inDir = document.createElement('input');
    inDir.className = 'hf-add-input add-dest-dir'; inDir.value = autoDir; inDir.placeholder = 'misc';
    const inDirErr = document.createElement('div');
    inDirErr.className = 'dir-error-hint';
    inDir.addEventListener('input', () => applyDirValidation(inDir, inDirErr));
    tdDir.append(inDir, inDirErr);

    // tags
    const tdTags = document.createElement('td');
    const inTags = document.createElement('input');
    inTags.className = 'hf-add-input add-tags'; inTags.value = autoTags; inTags.placeholder = 'llm chat 8b';
    tdTags.appendChild(inTags);

    // vram
    const tdVram = document.createElement('td');
    const inVram = document.createElement('input');
    inVram.type = 'number'; inVram.className = 'hf-add-input add-vram';
    inVram.value = autoVram; inVram.min = '0'; inVram.step = '1';
    tdVram.appendChild(inVram);

    // description
    const tdDesc = document.createElement('td');
    const inDesc = document.createElement('input');
    inDesc.className = 'hf-add-input add-description'; inDesc.value = autoDesc; inDesc.placeholder = 'описание';
    tdDesc.appendChild(inDesc);

    tr.append(tdM, tdDir, tdTags, tdVram, tdDesc);
    tbody.appendChild(tr);
  }

  document.getElementById('hf-add-form').style.display = 'block';
  document.getElementById('hf-add-form').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hfCancelAdd() {
  document.getElementById('hf-add-form').style.display = 'none';
}

async function hfConfirmAdd() {
  const enabled = document.getElementById('add-enabled').checked;
  const models = [];
  for (const tr of document.querySelectorAll('#hf-add-rows tr')) {
    const key = tr.dataset.key;
    const item = hfSelected.get(key);
    if (!item) continue;
    if (!item.filename) {
      setAddStatus('Ошибка: у некоторых записей нет имени файла — используйте «Regex файлов» при поиске', true);
      return;
    }
    const tagsRaw = tr.querySelector('.add-tags').value.trim();
    models.push({
      repo_id: item.repo_id, filename: item.filename,
      enabled, gated: item.gated,
      dest_dir:    tr.querySelector('.add-dest-dir').value.trim() || 'misc',
      tags:        tagsRaw ? tagsRaw.split(/\s+/).filter(Boolean) : [],
      vram_gb:     parseFloat(tr.querySelector('.add-vram').value) || 0,
      description: tr.querySelector('.add-description').value.trim(),
    });
  }
  if (models.length === 0) {
    setAddStatus('Нет моделей для добавления', true);
    return;
  }

  // Validate all dest_dir fields before sending
  let dirHasErrors = false;
  for (const tr of document.querySelectorAll('#hf-add-rows tr')) {
    const inDir = tr.querySelector('.add-dest-dir');
    if (!inDir) continue;
    const hintEl = inDir.nextElementSibling;
    if (!applyDirValidation(inDir, hintEl)) dirHasErrors = true;
  }
  if (dirHasErrors) {
    setAddStatus('Исправьте ошибки в поле «Папка» перед добавлением', true);
    return;
  }

  const btn = document.getElementById('add-confirm-btn');
  btn.disabled = true;
  setAddStatus('<span class="spinner"></span>&nbsp;Сохранение...', false);

  try {
    const resp = await fetch('/api/add', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ models }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'HTTP ' + resp.status);
    const skippedNote = data.skipped && data.skipped.length > 0
      ? ` · Пропущено дубликатов: ${data.skipped.length}` : '';
    setAddStatus(`Добавлено: ${data.added}${skippedNote}`, false);
    hfSelected.clear();
    document.getElementById('hf-add-btn').disabled = true;
    document.querySelectorAll('#hf-results-body .select-cb').forEach(cb => cb.checked = false);
    document.getElementById('hf-select-all').checked = false;
    document.getElementById('hf-select-all').indeterminate = false;
    loadModels(); // обновляем список моделей в фоне
  } catch (e) {
    setAddStatus('Ошибка: ' + escHtml(e.message), true);
  } finally {
    btn.disabled = false;
  }
}

function setAddStatus(html, isErr) {
  const el = document.getElementById('add-status');
  el.innerHTML = html;
  el.className = 'status-msg' + (isErr ? ' err' : ' ok');
  el.style.display = html ? '' : 'none';
}

function dlToggleLog() {
  const logEl = document.getElementById('dl-log');
  const btn = document.getElementById('dl-expand-btn');
  const expanded = logEl.classList.toggle('dl-log--expanded');
  btn.textContent = expanded ? '⤡ Свернуть' : '⤢ Развернуть';
}

// ═══════════════════════════════════════════════════════════════════
//  Download — SSE client
// ═══════════════════════════════════════════════════════════════════

let _dlEventSource = null;
let _dlLastData = null;

// ─── Shared SSE message handler ───────────────────────────────────────────────
function _dlOnMessage(d) {
  if (d.current) {
    document.getElementById('dl-current').textContent = d.current;
  }

  // Progress bar — indeterminate animation while file is downloading (progress=0)
  const barEl = document.getElementById('dl-progress-bar');
  if (d.running && d.progress === 0) {
    barEl.classList.add('indeterminate');
  } else {
    barEl.classList.remove('indeterminate');
    barEl.style.width = (d.progress || 0) + '%';
  }

  if (d.model_count > 0) {
    document.getElementById('dl-counter').textContent =
      d.done_count + ' / ' + d.model_count;
  }

  if (d.log && d.log.length > 0) {
    const logEl = document.getElementById('dl-log');
    const atBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 20;
    logEl.innerHTML = d.log.map(line => {
      let cls = '';
      if (/\[SKIP\]/.test(line))      cls = 'log-skip';
      else if (/\[OK\]/.test(line))   cls = 'log-ok';
      else if (/\[ERR/.test(line))    cls = 'log-err';
      else if (/\[WORKER/.test(line)) cls = 'log-warn';
      return cls ? `<span class="${cls}">${escHtml(line)}</span>` : escHtml(line);
    }).join('\n');
    if (atBottom) logEl.scrollTop = logEl.scrollHeight;
  }

  _dlLastData = d;
  if (!d.running) { dlDone(); }
}

// ─── Attach panel + SSE (called after start or on page-load resume) ──────────
function dlAttach() {
  const controls = document.getElementById('dl-controls');
  const cancelBtn = document.getElementById('dl-cancel-btn');
  const panel = document.getElementById('download-panel');
  controls.style.display = 'none';
  cancelBtn.style.display = '';
  cancelBtn.disabled = false;
  panel.style.display = '';
  document.getElementById('dl-cancel-btn-panel').style.display = '';
  document.getElementById('dl-cancel-btn-panel').disabled = false;
  if (_dlEventSource) { _dlEventSource.close(); }
  _dlEventSource = new EventSource('/api/download/stream');
  _dlEventSource.onmessage = function(e) {
    let d; try { d = JSON.parse(e.data); } catch(_) { return; }
    _dlOnMessage(d);
  };
  _dlEventSource.onerror = function() { dlDone(); };
}

async function dlStart() {
  const startBtn  = document.getElementById('dl-start-btn');
  const statusMsg = document.getElementById('dl-status-msg');

  const _concRaw = parseInt(document.getElementById('dl-concurrency').value, 10);
  const concurrency = (_concRaw > 0) ? _concRaw : 4;
  const _toRaw  = parseFloat(document.getElementById('dl-timeout').value);
  const timeout = (!isNaN(_toRaw) && _toRaw >= 0) ? _toRaw : 2;
  const _bwRaw  = parseFloat(document.getElementById('dl-bandwidth').value);
  const bandwidth = (!isNaN(_bwRaw) && _bwRaw > 0) ? _bwRaw : null;
  const syncS3 = document.getElementById('dl-sync-s3').checked;

  startBtn.disabled = true;
  statusMsg.style.display = 'none';
  try {
    const resp = await fetch('/api/download/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_concurrency: concurrency, download_timeout: timeout, bandwidth_limit: bandwidth, sync_s3: syncS3 }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      statusMsg.className = 'status-msg err';
      statusMsg.textContent = 'Ошибка запуска: ' + (data.error || 'HTTP ' + resp.status);
      statusMsg.style.display = '';
      startBtn.disabled = false;
      return;
    }
  } catch (e) {
    statusMsg.className = 'status-msg err';
    statusMsg.textContent = 'Ошибка: ' + e.message;
    statusMsg.style.display = '';
    startBtn.disabled = false;
    return;
  }

  _dlLastData = null;
  document.getElementById('dl-log').textContent = '';
  document.getElementById('dl-current').textContent = '—';
  document.getElementById('dl-progress-bar').classList.remove('indeterminate');
  document.getElementById('dl-progress-bar').style.width = '0%';
  document.getElementById('dl-counter').textContent = '';
  dlAttach();
}

function dlDone() {
  if (_dlEventSource) { _dlEventSource.close(); _dlEventSource = null; }
  const startBtn  = document.getElementById('dl-start-btn');
  const cancelBtn = document.getElementById('dl-cancel-btn');
  const controls  = document.getElementById('dl-controls');
  const panel     = document.getElementById('download-panel');

  document.getElementById('dl-progress-bar').style.width = '100%';
  cancelBtn.style.display = 'none';
  document.getElementById('dl-cancel-btn-panel').style.display = 'none';
  controls.style.display = '';
  startBtn.disabled = false;

  // Build completion summary from last SSE snapshot
  const currentEl = document.getElementById('dl-current');
  if (_dlLastData && _dlLastData.status_map && Object.keys(_dlLastData.status_map).length > 0) {
    const counts = { DOWNLOAD: 0, SKIP: 0, ERROR: 0 };
    Object.values(_dlLastData.status_map).forEach(v => { if (v in counts) counts[v]++; });
    const parts = [];
    if (counts.DOWNLOAD > 0) parts.push(`<span class="dl-badge dl-badge-dl">↓ ${counts.DOWNLOAD} скачано</span>`);
    if (counts.SKIP > 0)     parts.push(`<span class="dl-badge dl-badge-skip">— ${counts.SKIP} пропущено</span>`);
    if (counts.ERROR > 0)    parts.push(`<span class="dl-badge dl-badge-err">✗ ${counts.ERROR} ошибок</span>`);
    currentEl.innerHTML = parts.length > 0 ? parts.join(' ') : 'Завершено';
  } else {
    currentEl.textContent = 'Завершено';
  }

  // Refresh model status dots after download
  loadModels();

  // Auto-hide panel after 8 seconds
  setTimeout(() => { panel.style.display = 'none'; }, 8000);
}

async function dlCancel() {
  document.getElementById('dl-cancel-btn').disabled = true;
  document.getElementById('dl-cancel-btn-panel').disabled = true;
  try {
    await fetch('/api/download/cancel', { method: 'POST' });
  } catch (_) {}
}

// Init
loadModels();
loadSettings();
checkRunningDownload();

// Resume download panel if a download is already running (e.g. after page refresh)
function checkRunningDownload() {
  const probe = new EventSource('/api/download/stream');
  probe.onmessage = function(e) {
    probe.close();
    let d; try { d = JSON.parse(e.data); } catch(_) { return; }
    if (!d.running) return;
    // Download is in progress — show panel and attach SSE without POSTing /start
    document.getElementById('dl-start-btn').disabled = true;
    dlAttach();
  };
  probe.onerror = function() { probe.close(); };
}

async function loadSettings() {
  try {
    const resp = await fetch('/api/settings');
    if (!resp.ok) return;
    const s = await resp.json();
    if (s.hf_download_concurrency != null)
      document.getElementById('dl-concurrency').value = s.hf_download_concurrency;
    if (s.download_timeout_hours != null)
      document.getElementById('dl-timeout').value = s.download_timeout_hours;
    if (s.bandwidth_limit_mbps != null)
      document.getElementById('dl-bandwidth').value = s.bandwidth_limit_mbps;
    if (s.sync_after_download != null)
      document.getElementById('dl-sync-s3').checked = s.sync_after_download;
  } catch (_) {}
}

async function saveSettings() {
  const btn = document.getElementById('dl-save-btn');
  btn.disabled = true;

  const concRaw = parseInt(document.getElementById('dl-concurrency').value, 10);
  const toRaw   = parseFloat(document.getElementById('dl-timeout').value);
  const bwRaw   = parseFloat(document.getElementById('dl-bandwidth').value);
  const syncS3  = document.getElementById('dl-sync-s3').checked;

  // null means "remove from yaml" (no limit); 0 for timeout means disabled
  const body = {
    hf_download_concurrency: (concRaw > 0)                    ? concRaw : null,
    download_timeout_hours:  (!isNaN(toRaw) && toRaw >= 0)    ? toRaw   : null,
    bandwidth_limit_mbps:    (!isNaN(bwRaw) && bwRaw > 0)     ? bwRaw   : null,
    sync_after_download:     syncS3,
  };

  try {
    const resp = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (resp.ok) {
      btn.textContent = '✓ Сохранено';
      btn.classList.add('saved');
      btn.classList.remove('error');
    } else {
      btn.textContent = '✗ Ошибка';
      btn.classList.add('error');
      btn.classList.remove('saved');
      console.error('saveSettings error:', data.error);
    }
  } catch (e) {
    btn.textContent = '✗ Ошибка';
    btn.classList.add('error');
    btn.classList.remove('saved');
    console.error('saveSettings fetch error:', e);
  }

  btn.disabled = false;
  // Reset button label after 2.5 sec
  setTimeout(() => {
    btn.textContent = 'Сохранить';
    btn.classList.remove('saved', 'error');
  }, 2500);
}
</script>
</body>
</html>
"""


# ─── Config Loading ────────────────────────────────────────────────────────────


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
            "disk_size_str": fmt_size(disk_size_bytes) if downloaded else "-",
        })

    return result


# ─── Security Helpers ─────────────────────────────────────────────────────────

# Mirrors the client-side JS validateDestDir regex.
# Allowed: lowercase letters, digits, hyphens, underscores; max 3 path levels.
# Examples: "misc", "llm/qwen", "embeddings/russian"
_DEST_DIR_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*(?:/[a-z0-9][a-z0-9_-]*){0,2}$")

# Allowed: author/model-name  (HuggingFace standard)
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.\-]+$")


def _validate_dest_dir(dest_dir: str, models_dir: Optional[Path] = None) -> None:
    """
    Validate dest_dir against path traversal attacks.

    1. Regex check — only safe characters / depth (client JS has the same rule).
    2. Path resolution check — ensures the resolved path stays inside models_dir.

    Raises ValueError with a descriptive message on any violation.
    """
    if not _DEST_DIR_RE.match(dest_dir):
        raise ValueError(
            f"Недопустимый dest_dir '{dest_dir}': разрешены строчные буквы, цифры, "
            "дефис, подчёркивание; максимум 3 уровня пути (напр. 'llm/qwen')"
        )
    if models_dir is not None:
        resolved = (models_dir / dest_dir).resolve()
        try:
            resolved.relative_to(models_dir.resolve())
        except ValueError:
            raise ValueError(
                f"dest_dir '{dest_dir}' выходит за пределы директории моделей — "
                "path traversal недопустим"
            )


def save_models(config_path: Path, updates: list[dict]) -> None:
    """
    Atomically update model entries in models.yaml.

    Each update dict must contain repo_id + filename as identifiers and may
    contain any combination of: enabled, tags, vram_gb, description, gated, dest_dir.

    If dest_dir changes and the model file exists at the old path, it is moved
    to the new path (parent dirs created automatically).
    """
    raw = load_yaml(config_path)
    settings: dict = raw.get("settings", {})
    models_dir = Path(settings.get("models_dir", "./models"))
    if not models_dir.is_absolute():
        models_dir = config_path.parent / models_dir

    update_map: dict[tuple[str, str], dict] = {
        (str(u.get("repo_id", "")), str(u.get("filename", ""))): u
        for u in updates
        if u.get("repo_id") and u.get("filename")
    }

    for item in raw.get("models", []) or []:
        if not isinstance(item, dict):
            continue
        key = (item.get("repo_id", ""), item.get("filename", ""))
        if key not in update_map:
            continue
        u = update_map[key]

        # dest_dir change → validate first, then move file if it exists
        if "dest_dir" in u:
            new_dir = str(u["dest_dir"]).strip() or "misc"
            _validate_dest_dir(new_dir, models_dir)  # raises ValueError on violation
            old_dir = str(item.get("dest_dir", "misc"))
            if old_dir != new_dir:
                old_path = models_dir / old_dir / item["filename"]
                new_path = models_dir / new_dir / item["filename"]
                if old_path.is_file():
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_path), str(new_path))
            item["dest_dir"] = new_dir

        if "enabled" in u:
            item["enabled"] = bool(u["enabled"])
        if "gated" in u:
            item["gated"] = bool(u["gated"])
        if "vram_gb" in u:
            try:
                item["vram_gb"] = float(u["vram_gb"])
            except (ValueError, TypeError):
                pass
        if "description" in u:
            item["description"] = str(u.get("description") or "")
        if "tags" in u:
            raw_tags = u["tags"]
            if isinstance(raw_tags, list):
                item["tags"] = [str(t) for t in raw_tags if t]
            elif isinstance(raw_tags, str):
                item["tags"] = [t for t in raw_tags.split() if t]

    _atomic_yaml_write(config_path, raw)


# ─── HuggingFace Search ────────────────────────────────────────────────────────


def _sibling_size(s) -> Optional[int]:
    """Return actual file size: LFS size for LFS-tracked files, blob size otherwise."""
    lfs = getattr(s, "lfs", None)
    if lfs is not None:
        sz = getattr(lfs, "size", None)
        if sz:
            return sz
    sz = getattr(s, "size", None)
    return sz if sz else None


def search_hf(
    query: Optional[str],
    author: Optional[str],
    file_regex: Optional[str],
    limit: int,
    token: Optional[str],
    pipeline_tag: Optional[str] = None,
    language: Optional[str] = None,
) -> list[dict]:
    """
    Поиск моделей на HuggingFace Hub.

    Использует m.siblings для фильтрации файлов без лишних API-запросов.
    Возвращает список dict с полями: repo_id, downloads, likes, gated,
    pipeline_tag, tags, files.
    """
    from huggingface_hub import HfApi

    limit = min(50, max(1, limit))

    compiled_re: Optional[re.Pattern] = None
    if file_regex:
        try:
            compiled_re = re.compile(file_regex, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Невалидный regex: {exc}") from exc

    api = HfApi(token=token)

    # Запрашиваем с запасом если есть regex (он отсеет часть)
    fetch_limit = limit * 4 if file_regex else limit

    kwargs: dict[str, Any] = {
        "limit": fetch_limit,
        "sort": "downloads",
        "expand": ["siblings", "cardData"],  # siblings — файлы; cardData — лицензия
    }
    if query:
        kwargs["search"] = query
    if author:
        kwargs["author"] = author
    if pipeline_tag:
        kwargs["pipeline_tag"] = pipeline_tag

    # Собираем тег-фильтры: язык + автоопределение формата по file_regex.
    # Без авто-фильтра формата list_models() возвращает топ-N глобально
    # (BERT, CLIP…) — у них нет .gguf/.safetensors → 0 результатов.
    _filter_tags: list[str] = []
    if language:
        _filter_tags.append(language)
    if file_regex and not query and not author:
        if re.search(r"\.gguf", file_regex, re.IGNORECASE):
            _filter_tags.append("gguf")
        elif re.search(r"\.safetensors", file_regex, re.IGNORECASE):
            _filter_tags.append("safetensors")
    if _filter_tags:
        kwargs["filter"] = _filter_tags if len(_filter_tags) > 1 else _filter_tags[0]

    models = list(api.list_models(**kwargs))

    results = []
    for m in models:
        # Фильтрация файлов через siblings (не требует отдельного запроса)
        AI_EXTS = ('.gguf', '.safetensors', '.bin', '.pt', '.pth', '.ckpt')
        if compiled_re is not None:
            siblings = m.siblings or []
            files = [
                {"name": s.rfilename, "size_bytes": _sibling_size(s)}
                for s in sorted(siblings, key=lambda x: x.rfilename)
                if compiled_re.search(s.rfilename)
            ]
            if not files:
                continue  # пропускаем репо без совпадений
        else:
            siblings = m.siblings or []
            files = [
                {"name": s.rfilename, "size_bytes": _sibling_size(s)}
                for s in sorted(siblings, key=lambda x: x.rfilename)
                if s.rfilename.lower().endswith(AI_EXTS)
            ]

        pipeline_val = getattr(m, "pipeline_tag", "") or ""
        likes = getattr(m, "likes", 0) or 0

        # License + auto-generated description from model name
        card = getattr(m, "card_data", None)
        license_val = ""
        if card is not None:
            if isinstance(card, dict):
                license_val = str(card.get("license", "") or "").strip()
            else:
                license_val = str(getattr(card, "license", "") or "").strip()
        repo_name = m.id.split("/")[-1]
        clean_name = re.sub(
            r"[-_]?(?:i\d+[-_]?)?GGUF$", "", repo_name, flags=re.IGNORECASE
        ).strip("-_ ")
        description = clean_name + (f" [{license_val}]" if license_val else "")

        # Build display tags from multiple sources (m.tags may be empty for GGUF repos)
        _skip_tags = {"transformers", "endpoints_compatible", "conversational",
                      "text-generation-inference", "has_space", "region:us"}
        display_tags: list[str] = []
        # 1. pipeline_tag first
        if pipeline_val:
            display_tags.append(pipeline_val)
        # 2. Non-namespaced HF tags
        for t in (m.tags or []):
            if ":" not in t and t not in _skip_tags and len(t) <= 20 and t not in display_tags:
                display_tags.append(t)
                if len(display_tags) >= 4:
                    break
        # 3. License as tag
        if license_val and len(license_val) <= 20:
            lic_tag = license_val.lower()
            if lic_tag not in display_tags:
                display_tags.append(lic_tag)
        # 4. Size extracted from repo name (e.g. "14B" → "14b")
        size_m = re.search(r"[-_](\d+(?:\.\d+)?)[Bb](?:[-_A-Z]|$)", m.id)
        if size_m:
            size_tag = size_m.group(1).rstrip("0").rstrip(".") + "b"
            if size_tag not in display_tags:
                display_tags.append(size_tag)
        tags = display_tags[:6]

        results.append({
            "repo_id": m.id,
            "downloads": getattr(m, "downloads", 0) or 0,
            "likes": likes,
            "gated": bool(getattr(m, "gated", False)),
            "pipeline_tag": pipeline_val,
            "tags": tags,
            "files": files,
            "description": description,
            "license": license_val,
        })

        if len(results) >= limit:
            break

    return results


# ─── Add models to YAML ────────────────────────────────────────────────────────


def add_models_to_yaml(
    config_path: Path,
    new_models: list[dict],
) -> dict:
    """
    Атомарно добавляет новые записи в models.yaml.

    Пропускает дубликаты по паре (repo_id, filename).
    Возвращает {"added": int, "skipped": list[str]}.
    """
    raw = load_yaml(config_path)
    existing_keys: set[tuple[str, str]] = {
        (m.get("repo_id", ""), m.get("filename", ""))
        for m in (raw.get("models", []) or [])
        if isinstance(m, dict)
    }

    added = 0
    skipped: list[str] = []

    for item in new_models:
        repo_id = str(item.get("repo_id", "")).strip()
        filename = str(item.get("filename", "")).strip()
        if not repo_id or not filename:
            continue

        # Server-side validation (client JS can be bypassed via curl/API)
        if not _REPO_ID_RE.match(repo_id):
            skipped.append(f"INVALID_REPO_ID:{repo_id}")
            continue
        if Path(filename).name != filename:
            skipped.append(f"INVALID_FILENAME:{filename}")
            continue

        raw_dest_dir = str(item.get("dest_dir", "misc") or "misc")
        try:
            _validate_dest_dir(raw_dest_dir)  # no models_dir: no file ops here
        except ValueError as exc:
            skipped.append(f"INVALID_DEST_DIR:{raw_dest_dir} ({exc})")
            continue

        key = (repo_id, filename)
        if key in existing_keys:
            skipped.append(f"{repo_id}::{filename}")
            continue

        # Безопасное приведение типов
        try:
            vram = float(item.get("vram_gb", 0) or 0)
        except (ValueError, TypeError):
            vram = 0.0

        raw_tags = item.get("tags", []) or []
        if isinstance(raw_tags, list):
            tags = [str(t) for t in raw_tags if t]
        else:
            tags = []

        entry: dict[str, Any] = {
            "repo_id": repo_id,
            "filename": filename,
            "dest_dir": raw_dest_dir,
            "enabled": bool(item.get("enabled", False)),
            "gated": bool(item.get("gated", False)),
            "tags": tags,
            "vram_gb": vram,
            "description": str(item.get("description", "") or ""),
        }

        if "models" not in raw or raw["models"] is None:
            raw["models"] = []
        raw["models"].append(entry)
        existing_keys.add(key)
        added += 1

    if added > 0:
        _atomic_yaml_write(config_path, raw)

    return {"added": added, "skipped": skipped}


def delete_model(
    config_path: Path,
    repo_id: str,
    filename: str,
) -> dict:
    """
    Удаляет модель из models.yaml и, если файл существует на диске, удаляет его.

    Возвращает {"removed_yaml": bool, "removed_file": bool, "file_path": str|None}.
    """
    # Filename must not contain path separators (prevents traversal via crafted filename)
    if Path(filename).name != filename:
        raise ValueError(
            f"Недопустимое имя файла '{filename}': разделители пути не разрешены"
        )

    raw = load_yaml(config_path)
    settings: dict = raw.get("settings", {})
    models_dir = Path(settings.get("models_dir", "./models"))
    if not models_dir.is_absolute():
        models_dir = config_path.parent / models_dir

    models_list = raw.get("models", []) or []
    new_list = []
    removed_yaml = False
    removed_file = False
    file_path_str: Optional[str] = None

    for item in models_list:
        if not isinstance(item, dict):
            new_list.append(item)
            continue
        if item.get("repo_id") == repo_id and item.get("filename") == filename:
            removed_yaml = True
            dest_dir = item.get("dest_dir", "misc")
            local_file = models_dir / dest_dir / filename
            file_path_str = str(local_file)
            if local_file.is_file():
                local_file.unlink()
                removed_file = True
        else:
            new_list.append(item)

    if not removed_yaml:
        raise ValueError(f"Запись не найдена: {repo_id}::{filename}")

    raw["models"] = new_list
    _atomic_yaml_write(config_path, raw)
    return {"removed_yaml": removed_yaml, "removed_file": removed_file, "file_path": file_path_str}


def _atomic_yaml_write(config_path: Path, data: dict) -> None:
    """Атомарная запись YAML через tempfile + os.replace()."""
    config_dir = config_path.parent
    fd, tmp_path = tempfile.mkstemp(dir=str(config_dir), suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        os.replace(tmp_path, str(config_path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── HTTP Handler ──────────────────────────────────────────────────────────────


class ModelBrowserHandler(BaseHTTPRequestHandler):
    """HTTP request handler for AI Model Browser."""

    config_path: Path       # set by make_handler()
    hf_token: Optional[str] # set by make_handler()

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        print(f"  {self.address_string()} {format % args}")

    def log_error(self, format: str, *args) -> None:  # noqa: A002
        """Suppress SSL handshake noise (browser sending HTTPS to HTTP server)."""
        msg = format % args
        if "Bad request version" in msg or "Bad request syntax" in msg:
            return  # HTTPS→HTTP mismatch: harmless, no need to show in terminal
        print(f"  [ERR] {self.address_string()} {msg}", file=sys.stderr)

    def _send_json(self, data: Any, status: int = 200) -> None:
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

    def _send_sse_headers(self) -> None:
        """Send SSE (Server-Sent Events) response headers, then keep connection open."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send_html(HTML_PAGE)

        elif path == "/api/settings":
            try:
                raw = load_yaml(self.config_path)
                s = raw.get("settings", {})
                self._send_json({
                    "hf_download_concurrency": s.get("hf_download_concurrency"),
                    "bandwidth_limit_mbps":    s.get("bandwidth_limit_mbps"),
                    "download_timeout_hours":  s.get("download_timeout_hours"),
                    "sync_after_download":     s.get("s3", {}).get("sync_after_download"),
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif path == "/api/models":
            try:
                models = get_models_json(self.config_path)
                self._send_json(models)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif path == "/api/search":
            qs = parse_qs(parsed.query)
            query        = qs.get("q", [None])[0] or None
            author       = qs.get("author", [None])[0] or None
            file_regex   = qs.get("file_regex", [None])[0] or None
            pipeline_tag = qs.get("pipeline_tag", [None])[0] or None
            language     = qs.get("language", [None])[0] or None
            try:
                limit = int(qs.get("limit", ["20"])[0])
            except (ValueError, TypeError):
                limit = 20

            try:
                results = search_hf(
                    query, author, file_regex, limit, self.hf_token,
                    pipeline_tag=pipeline_tag, language=language,
                )
                self._send_json({"results": results, "count": len(results)})
            except ValueError as e:
                self._send_json({"error": str(e), "results": []}, status=400)
            except Exception as e:
                self._send_json({"error": str(e), "results": []}, status=500)

        elif path == "/api/download/stream":
            self._send_sse_headers()
            # Defensive deadline: auto-close stream after 2 hours even if
            # running flag is never cleared (guards against worker crash edge cases).
            deadline = time.time() + 7200
            try:
                while time.time() < deadline:
                    with _dl_lock:
                        snapshot = {
                            "running": download_state["running"],
                            "cancelled": download_state["cancelled"],
                            "current": download_state["current"],
                            "progress": download_state["progress"],
                            "model_count": download_state["model_count"],
                            "done_count": download_state["done_count"],
                            "log": list(download_state["log"][-100:]),
                            "status_map": dict(download_state["status_map"]),
                        }
                    payload = json.dumps(snapshot, ensure_ascii=False)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    if not snapshot["running"]:
                        break
                    time.sleep(0.5)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Client disconnected — download continues in background

        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            length = min(int(self.headers.get("Content-Length", 0) or 0), 10 * 1024 * 1024)
        except (ValueError, TypeError):
            length = 0
        body = self.rfile.read(length)

        if path == "/api/download/cancel":
            with _dl_lock:
                download_state["cancelled"] = True
                proc = download_state.get("process")
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            self._send_json({"status": "cancelled"})
            return

        if path == "/api/download/start":
            # Parse optional throttle params from JSON body (sent by UI controls)
            try:
                params = json.loads(body) if body else {}
            except (json.JSONDecodeError, ValueError):
                params = {}
            try:
                max_concurrency: Optional[int] = int(params["max_concurrency"]) \
                    if params.get("max_concurrency") is not None else None
            except (ValueError, TypeError):
                max_concurrency = None
            try:
                download_timeout: Optional[float] = float(params["download_timeout"]) \
                    if params.get("download_timeout") is not None else None
            except (ValueError, TypeError):
                download_timeout = None
            try:
                _bw_raw = params.get("bandwidth_limit")
                bandwidth_limit_mbps: Optional[float] = float(_bw_raw) if _bw_raw is not None else None
                # Treat 0 as "no limit"
                if bandwidth_limit_mbps is not None and bandwidth_limit_mbps <= 0:
                    bandwidth_limit_mbps = None
            except (ValueError, TypeError):
                bandwidth_limit_mbps = None
            sync_s3: bool = bool(params.get("sync_s3", False))

            with _dl_lock:
                if download_state["running"]:
                    self._send_json({"error": "Download already running"}, status=409)
                    return
                # Count enabled models
                try:
                    models_list = get_models_json(self.config_path)
                    enabled_count = sum(1 for m in models_list if m.get("enabled", True))
                except Exception:
                    enabled_count = 0
                # Reset state
                download_state["running"] = True
                download_state["cancelled"] = False
                download_state["process"] = None
                download_state["log"] = []
                download_state["current"] = ""
                download_state["progress"] = 0
                download_state["model_count"] = enabled_count
                download_state["done_count"] = 0
                download_state["status_map"] = {}
            # Start background download thread with throttle params from UI
            t = threading.Thread(
                target=_download_worker,
                args=(self.config_path, max_concurrency, download_timeout, bandwidth_limit_mbps, sync_s3),
                daemon=True,
            )
            t.start()
            self._send_json({"status": "started", "model_count": enabled_count})
            return

        if path == "/api/settings":
            try:
                params = json.loads(body) if body else {}
            except (json.JSONDecodeError, ValueError):
                params = {}
            try:
                raw = load_yaml(self.config_path)
                s: dict = raw.setdefault("settings", {})
                # hf_download_concurrency: int or null (removes key → no limit)
                if "hf_download_concurrency" in params:
                    v = params["hf_download_concurrency"]
                    if v is None:
                        s.pop("hf_download_concurrency", None)
                    else:
                        s["hf_download_concurrency"] = int(v)
                # bandwidth_limit_mbps: float or null (removes key → unlimited)
                if "bandwidth_limit_mbps" in params:
                    v = params["bandwidth_limit_mbps"]
                    if v is None:
                        s.pop("bandwidth_limit_mbps", None)
                    else:
                        s["bandwidth_limit_mbps"] = float(v)
                # download_timeout_hours: float (0 = no timeout) or null (removes key)
                if "download_timeout_hours" in params:
                    v = params["download_timeout_hours"]
                    if v is None:
                        s.pop("download_timeout_hours", None)
                    else:
                        s["download_timeout_hours"] = float(v)
                # sync_after_download: bool — сохраняется в settings.s3.sync_after_download
                if "sync_after_download" in params:
                    s3_sec = s.setdefault("s3", {})
                    s3_sec["sync_after_download"] = bool(params["sync_after_download"])
                _atomic_yaml_write(self.config_path, raw)
                self._send_json({"status": "ok"})
            except (ValueError, TypeError) as e:
                self._send_json({"error": str(e)}, status=400)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            self._send_json({"error": f"Invalid JSON: {e}"}, status=400)
            return

        if path == "/api/save":
            try:
                updates = payload.get("updates", [])
                if not isinstance(updates, list):
                    raise ValueError("'updates' must be a list")
                save_models(self.config_path, updates)
                self._send_json({"status": "ok", "updated": len(updates)})
            except (ValueError, TypeError) as e:
                self._send_json({"error": str(e)}, status=400)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif path == "/api/add":
            try:
                models_list = payload.get("models", [])
                if not isinstance(models_list, list):
                    raise ValueError("'models' must be a list")
                result = add_models_to_yaml(self.config_path, models_list)
                self._send_json({"status": "ok", **result})
            except (ValueError, TypeError) as e:
                self._send_json({"error": str(e)}, status=400)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif path == "/api/delete":
            try:
                repo_id = str(payload.get("repo_id", "")).strip()
                filename = str(payload.get("filename", "")).strip()
                if not repo_id or not filename:
                    raise ValueError("repo_id и filename обязательны")
                result = delete_model(self.config_path, repo_id, filename)
                self._send_json({"status": "ok", **result})
            except (ValueError, TypeError) as e:
                self._send_json({"error": str(e)}, status=400)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        else:
            self._send_json({"error": "Not found"}, status=404)


# ─── Download Worker ──────────────────────────────────────────────────────────


def _download_worker(
    config_path: Path,
    max_concurrency: Optional[int] = None,
    download_timeout: Optional[float] = None,
    bandwidth_limit_mbps: Optional[float] = None,
    sync_s3: bool = False,
) -> None:
    """
    Background thread that runs download_models.py as a subprocess.

    Streams stdout lines into download_state['log'] and parses progress
    markers of the form:
      [repo_id] — начало модели
      [OK]  / [SKIP] / [ERR] — результат
    Updates download_state fields under _dl_lock.

    Args:
        max_concurrency:    --max-concurrency N  (HuggingFace Xet / HTTP parallel connections)
        download_timeout:   --download-timeout H (per-file timeout in hours, 0=unlimited)
        bandwidth_limit_mbps: --bandwidth-limit M (soft cap via concurrency compensation, Mbit/s)
        sync_s3:            --upload-s3  (upload downloaded models to S3 after download)
    """
    script = Path(__file__).parent / "download_models.py"
    cmd = [
        sys.executable, "-u", str(script),   # -u: force unbuffered stdout (critical when piped)
        "--config", str(config_path),
    ]
    if max_concurrency is not None:
        cmd += ["--max-concurrency", str(int(max_concurrency))]
    if download_timeout is not None:
        cmd += ["--download-timeout", str(float(download_timeout))]
    if bandwidth_limit_mbps is not None:
        cmd += ["--bandwidth-limit", str(float(bandwidth_limit_mbps))]
    if sync_s3:
        cmd += ["--upload-s3"]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr → stdout for single stream
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        with _dl_lock:
            download_state["process"] = proc

        import re as _re
        _tqdm_re = _re.compile(r'^\s*(\d+)%\|')  # matches "  52%|████..."

        done_count = 0
        model_count = download_state["model_count"] or 1  # avoid div-by-zero

        for raw_line in proc.stdout:  # type: ignore[union-attr]
            line = raw_line.rstrip()

            # tqdm refreshes via carriage return — keep only the last (most recent) value
            if "\r" in line:
                line = line.rsplit("\r", 1)[-1].strip()
            if not line:
                continue

            with _dl_lock:
                if download_state["cancelled"]:
                    break

                # [PROGRESS] idx/total — model-level marker, don't add to log
                if line.startswith("[PROGRESS]"):
                    try:
                        parts = line.split()[1].split("/")
                        done_count = int(parts[0]) - 1
                        model_count = int(parts[1])
                        download_state["model_count"] = model_count
                        download_state["done_count"] = done_count
                        # Per-file progress starts at 0 until tqdm lines arrive
                        base_pct = int(done_count * 100 / max(model_count, 1))
                        download_state["progress"] = min(base_pct, 99)
                    except (IndexError, ValueError):
                        pass
                    continue

                # [FILESIZE] N — expected total bytes for current file (from HF metadata)
                if line.startswith("[FILESIZE]"):
                    continue  # informational only, don't add to log

                # [FILEPROGRESS] current/total — file-size monitor (hf_xet has no tqdm)
                if line.startswith("[FILEPROGRESS]"):
                    try:
                        cur_s, tot_s = line.split()[1].split("/")
                        cur_b, tot_b = int(cur_s), int(tot_s)
                        if tot_b > 0:
                            _mc = max(model_count, 1)
                            file_pct = int(cur_b * 100 / tot_b)
                            base_pct = int(done_count * 100 / _mc)
                            slice_pct = int(100 / _mc)
                            scaled = base_pct + int(file_pct * slice_pct / 100)
                            download_state["progress"] = min(scaled, 99)
                    except (IndexError, ValueError):
                        pass
                    continue  # don't add to log

                # tqdm per-file progress line: "  52%|████..." → extract file %
                # (fallback for non-xet backends that do emit tqdm)
                m = _tqdm_re.match(line)
                if m:
                    file_pct = int(m.group(1))   # 0-100 within current file
                    _mc = max(model_count, 1)
                    # Scale file progress into the slice for this model: [base, base+slice)
                    base_pct = int(done_count * 100 / _mc)
                    slice_pct = int(100 / _mc)
                    scaled = base_pct + int(file_pct * slice_pct / 100)
                    download_state["progress"] = min(scaled, 99)
                    continue  # don't add tqdm lines to log

                download_state["log"].append(line)
                if len(download_state["log"]) > 500:
                    download_state["log"] = download_state["log"][-400:]

                # Parse model start: lines like "[ModelName]" from download_models.py
                if (line.startswith("[") and line.endswith("]")
                        and not any(line.startswith(p) for p in (
                            "[INFO]", "[WARN]", "[ERR", "[DOWNLOAD", "[SKIP", "[OK", "[FATAL", "[WORKER"))):
                    download_state["current"] = line[1:-1]

                # Parse result lines → advance done_count and progress
                for marker, status_key in (("[OK]", "DOWNLOAD"), ("[SKIP]", "SKIP"), ("[ERR]", "ERROR")):
                    if f"  {marker}" in line or line.startswith(marker):
                        cur = download_state["current"]
                        if cur:
                            download_state["status_map"][cur] = status_key
                        done_count += 1
                        download_state["done_count"] = done_count
                        _mc = download_state["model_count"] or 1
                        pct = int(done_count * 100 / _mc)
                        download_state["progress"] = min(pct, 99)
                        break

        rc = proc.wait()
        with _dl_lock:
            if rc != 0 and not download_state["cancelled"]:
                download_state["log"].append(f"[WORKER] download_models.py завершился с кодом {rc}")
            download_state["running"] = False
            download_state["progress"] = 100
            download_state["process"] = None

    except Exception as exc:
        with _dl_lock:
            download_state["log"].append(f"[WORKER ERROR] {exc}")
            download_state["running"] = False
            download_state["process"] = None


# ─── Server Factory ────────────────────────────────────────────────────────────


def make_handler(config_path: Path, hf_token: Optional[str] = None) -> type[ModelBrowserHandler]:
    """Create a handler class with config_path and hf_token bound."""

    class Handler(ModelBrowserHandler):
        pass

    Handler.config_path = config_path
    Handler.hf_token = hf_token
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
        type=int, default=9000, metavar="PORT",
        help="Порт HTTP-сервера (по умолчанию: 9000)",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1", metavar="HOST",
        help="Адрес для привязки (по умолчанию: 127.0.0.1)",
    )
    p.add_argument(
        "--config",
        default="models.yaml", metavar="FILE",
        help="Путь к models.yaml (по умолчанию: models.yaml)",
    )
    p.add_argument(
        "--creds",
        default="credentials.yaml", metavar="FILE",
        help="Файл с HF-токеном (по умолчанию: credentials.yaml)",
    )
    p.add_argument(
        "--open", "-o",
        action="store_true",
        help="Открыть браузер автоматически после запуска сервера",
    )
    return p


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)

    if not config_path.is_file():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        print("  Run from the project root or use --config to specify path.", file=sys.stderr)
        return 1

    try:
        load_yaml(config_path)
    except Exception as e:
        print(f"[ERROR] Cannot parse {config_path}: {e}", file=sys.stderr)
        return 1

    # Загрузка HF-токена (credentials.yaml → HF_TOKEN env)
    creds_path = Path(args.creds)
    hf_token = load_hf_token(creds_path)
    if hf_token:
        print("[INFO] HuggingFace token loaded (HF search enabled)")
    else:
        print("[INFO] No HF token — public search only (set HF_TOKEN or credentials.yaml)")

    handler_class = make_handler(config_path, hf_token)
    server = ThreadedHTTPServer((args.host, args.port), handler_class)

    url = f"http://localhost:{args.port}"
    print(f"[INFO] AI Model Browser")
    print(f"[INFO] Config : {config_path.resolve()}")
    print(f"[INFO] Listen : {args.host}:{args.port}")
    print(f"[INFO] URL    : {url}")
    print("[INFO] Press Ctrl+C to stop.")

    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
