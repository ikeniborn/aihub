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
    process: Optional[Any]          # subprocess.Popen or None
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
                  padding: 10px 24px; display: flex; align-items: center;
                  gap: 10px; flex-wrap: wrap; }
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

  /* ── HF Search panel ── */
  .hf-panel { padding: 20px 24px; }
  .hf-search-form {
    display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start;
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; margin-bottom: 14px;
  }
  .hf-field { display: flex; flex-direction: column; gap: 4px; }
  .hf-field label { font-size: 0.75rem; color: var(--text-muted); }
  .hf-field input {
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 6px;
    color: var(--text-primary); padding: 7px 12px; font-size: 0.875rem;
    outline: none; transition: border-color 0.15s;
  }
  .hf-field input:focus { border-color: var(--accent); }
  .hf-field input[type=text]   { width: 200px; }
  .hf-field input[type=number] { width: 80px; }

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
  .hf-field select:focus { border-color: var(--accent); }

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
  .select-cb { accent-color: var(--accent); width: 15px; height: 15px; cursor: pointer; }
  #hf-empty-msg { text-align: center; color: #5a6a82; padding: 40px; font-size: 0.9rem; display: none; }

  /* ── HF Add form ── */
  .hf-add-form {
    display: none; background: var(--bg-surface);
    border: 1px solid var(--accent); border-radius: 8px;
    padding: 18px; margin-top: 16px;
  }
  .hf-add-form h3 { font-size: 0.95rem; color: var(--text-secondary); margin-bottom: 14px; }
  .hf-add-fields { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 14px; }
  .hf-add-field { display: flex; flex-direction: column; gap: 4px; }
  .hf-add-field label { font-size: 0.75rem; color: var(--text-muted); }
  .hf-add-field input[type=text],
  .hf-add-field input[type=number] {
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 5px;
    color: var(--text-primary); padding: 6px 10px; font-size: 0.85rem;
    outline: none; transition: border-color 0.15s;
  }
  .hf-add-field input:focus { border-color: var(--accent); }
  .hf-add-field input[type=text]   { width: 200px; }
  .hf-add-field input[type=number] { width: 90px; }
  .hf-add-field.wide input[type=text] { width: 280px; }
  .hf-add-actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }

  /* ── Download Panel ── */
  .download-panel {
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 8px; margin: 10px 24px 0; padding: 14px 18px;
  }
  .download-panel-header {
    display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap;
  }
  .download-panel-header span { font-size: 0.82rem; color: var(--text-secondary); }
  .dl-current { font-size: 0.82rem; color: var(--accent); font-weight: 500; flex: 1;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .dl-progress-track {
    background: var(--bg-elevated); border-radius: 4px; height: 6px;
    margin-bottom: 10px; overflow: hidden;
  }
  .dl-progress-bar {
    background: var(--accent); height: 100%; border-radius: 4px;
    transition: width 0.4s ease; width: 0%;
  }
  .dl-log {
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 5px;
    color: var(--text-muted); font-family: 'SF Mono', Consolas, monospace; font-size: 0.72rem;
    height: 120px; overflow-y: auto; padding: 8px 10px; white-space: pre-wrap;
    word-break: break-all;
  }
  .dl-badge {
    display: inline-block; border-radius: 3px; padding: 1px 6px;
    font-size: 0.65rem; font-weight: 700; white-space: nowrap;
  }
  .dl-badge-dl   { background: #1a3a2a; color: var(--green); }
  .dl-badge-skip { background: #252a38; color: var(--text-muted); }
  .dl-badge-err  { background: #3a1a1a; color: var(--red); }
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
  <div class="controls-bar">
    <input type="text" id="search" placeholder="Поиск по модели, описанию..." oninput="applyFilters()">
    <div class="tag-filter" id="tag-filter"></div>
    <label class="toggle-label">
      <input type="checkbox" id="only-downloaded" onchange="applyFilters()">
      только скачанные
    </label>
    <button class="btn-primary" id="save-btn" onclick="saveChanges()" disabled style="margin-left:auto">Сохранить</button>
    <span class="status-msg" id="save-msg"></span>
    <button class="btn-secondary" id="dl-start-btn" onclick="dlStart()" style="margin-left:8px" title="Скачать все enabled модели">Скачать enabled</button>
    <button class="btn-secondary" id="dl-cancel-btn" onclick="dlCancel()" style="display:none">Отменить</button>
    <span class="status-msg" id="dl-status-msg"></span>
  </div>
  <div class="download-panel" id="download-panel" style="display:none">
    <div class="download-panel-header">
      <span>Загрузка:</span>
      <span class="dl-current" id="dl-current">—</span>
      <span id="dl-counter" style="font-size:0.78rem;color:var(--text-muted)"></span>
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
      <div class="hf-field">
        <label>Запрос</label>
        <input type="text" id="hf-query" placeholder="llama, qwen, deepseek..." onkeydown="if(event.key==='Enter')hfSearch()">
      </div>
      <div class="hf-field">
        <label>Автор</label>
        <input type="text" id="hf-author" placeholder="bartowski, unsloth..." onkeydown="if(event.key==='Enter')hfSearch()">
      </div>
      <div class="hf-field hf-field-regex">
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
        <label>Язык <span style="font-size:0.65rem;color:var(--text-muted)">(ISO 639-1)</span></label>
        <input type="text" id="hf-language" placeholder="ru, en, zh..." maxlength="20"
               onkeydown="if(event.key==='Enter')hfSearch()" style="width:120px">
        <div class="filter-chips" id="lang-chips">
          <span class="filter-chip" onclick="toggleLangChip(this,'ru')">🇷🇺 ru</span>
          <span class="filter-chip" onclick="toggleLangChip(this,'en')">🇬🇧 en</span>
          <span class="filter-chip" onclick="toggleLangChip(this,'zh')">🇨🇳 zh</span>
          <span class="filter-chip" onclick="toggleLangChip(this,'ja')">🇯🇵 ja</span>
          <span class="filter-chip" onclick="toggleLangChip(this,'de')">🇩🇪 de</span>
          <span class="filter-chip" onclick="toggleLangChip(this,'fr')">🇫🇷 fr</span>
        </div>
      </div>
      <div class="hf-field">
        <label>Лимит</label>
        <input type="number" id="hf-limit" value="20" min="1" max="50">
      </div>
      <div class="hf-field">
        <label>Макс. размер (GB)</label>
        <input type="number" id="hf-max-size" placeholder="∞" min="0" step="0.5" oninput="hfApplySizeFilter()">
      </div>
      <button class="btn-primary" id="hf-search-btn" onclick="hfSearch()">Искать</button>
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
        <div class="hf-add-fields">
          <div class="hf-add-field">
            <label>dest_dir</label>
            <input type="text" id="add-dest-dir" placeholder="misc" value="misc">
          </div>
          <div class="hf-add-field">
            <label>Теги (через пробел)</label>
            <input type="text" id="add-tags" placeholder="llm chat 8b">
          </div>
          <div class="hf-add-field">
            <label>VRAM (GB)</label>
            <input type="number" id="add-vram" placeholder="0" min="0" step="1" value="0">
          </div>
          <div class="hf-add-field wide">
            <label>Описание</label>
            <input type="text" id="add-description" placeholder="краткое описание модели">
          </div>
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
  const onlyDl = document.getElementById('only-downloaded').checked;
  const rows = document.getElementById('models-body').querySelectorAll('tr');
  let visible = 0;
  rows.forEach(row => {
    const m = allModels[parseInt(row.dataset.idx)];
    if (!m) return;
    let show = true;
    if (q && !(m.repo_id + ' ' + m.filename + ' ' + (m.description || '')).toLowerCase().includes(q)) show = false;
    if (show && onlyDl && !m.downloaded) show = false;
    if (show && activeTags.size > 0) {
      const mt = new Set(m.tags || []);
      for (const t of activeTags) { if (!mt.has(t)) { show = false; break; } }
    }
    row.style.display = show ? '' : 'none';
    if (show) visible++;
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

function renderModels(models) {
  const tbody = document.getElementById('models-body');
  tbody.innerHTML = '';
  models.forEach((m, i) => {
    const key = modelKey(m);
    const enabled = key in changes ? changes[key] : m.enabled;
    const tr = document.createElement('tr');
    tr.dataset.idx = i;
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

    tr.append(tdCb, tdStatus, tdName, tdTags, tdVram, tdSize, tdDesc);
    tbody.appendChild(tr);
  });
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

function toggleLangChip(chip, code) {
  const input = document.getElementById('hf-language');
  const active = chip.classList.contains('active');
  // Deactivate all chips
  document.querySelectorAll('#lang-chips .filter-chip').forEach(c => c.classList.remove('active'));
  if (active) {
    input.value = '';
  } else {
    chip.classList.add('active');
    input.value = code;
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Tab: Поиск HuggingFace
// ═══════════════════════════════════════════════════════════════════

let hfResults = [];
let hfSelected = new Map(); // "repo_id||filename" → {repo_id, filename, gated, tags}

async function hfSearch() {
  const q           = document.getElementById('hf-query').value.trim();
  const author      = document.getElementById('hf-author').value.trim();
  const fileRegex   = document.getElementById('hf-file-regex').value.trim();
  const pipelineTag = document.getElementById('hf-pipeline-tag').value;
  const language    = document.getElementById('hf-language').value.trim();
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
        tdRepo.innerHTML =
          `<a href="https://huggingface.co/${escHtml(r.repo_id)}" target="_blank">${escHtml(r.repo_id)}</a>` +
          (r.gated ? '<span class="gated-badge">GATED</span>' : '');
        tr.appendChild(tdRepo);
      }

      // filename + size
      const tdFile = document.createElement('td');
      tdFile.className = 'cell-hf-files';
      if (fname) {
        const sizeHtml = fsize != null
          ? `<span class="file-size">${fmtSize(fsize)}</span>`
          : '';
        tdFile.innerHTML = `<span class="file-item">${escHtml(fname)}${sizeHtml}</span>`;
      } else {
        tdFile.innerHTML = `<span style="color:var(--text-muted);font-style:italic">нет файлов по фильтру</span>`;
      }
      tr.appendChild(tdFile);

      // downloads — только на первой строке
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
    hfSelected.set(key, { repo_id: r.repo_id, filename: fname || '', gated: r.gated, tags: r.tags || [] });
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
      hfSelected.set(key, { repo_id, filename: fname, gated: r.gated, tags: r.tags || [] });
    } else {
      hfSelected.delete(key);
    }
  });
  document.getElementById('hf-add-btn').disabled = hfSelected.size === 0;
}

function hfApplySizeFilter() {
  const maxGbVal = document.getElementById('hf-max-size').value.trim();
  const maxBytes = maxGbVal !== '' ? parseFloat(maxGbVal) * 1024 * 1024 * 1024 : null;
  document.querySelectorAll('#hf-results-body tr').forEach(tr => {
    if (maxBytes === null || isNaN(maxBytes)) {
      tr.style.display = '';
      return;
    }
    const sizeBytes = tr.dataset.sizeBytes != null ? parseFloat(tr.dataset.sizeBytes) : null;
    // Hide only rows where size is known AND exceeds the limit
    if (sizeBytes != null && sizeBytes > maxBytes) {
      tr.style.display = 'none';
      // Uncheck and deselect if hidden
      const cb = tr.querySelector('.select-cb');
      if (cb && cb.checked) {
        cb.checked = false;
        hfSelected.delete(cb.dataset.key);
      }
    } else {
      tr.style.display = '';
    }
  });
  // Update add button state
  document.getElementById('hf-add-btn').disabled = hfSelected.size === 0;
}

function hfShowAddForm() {
  const n = hfSelected.size;
  const word = n === 1 ? 'модель' : n < 5 ? 'модели' : 'моделей';
  document.getElementById('hf-add-title').textContent = `Добавить ${n} ${word} в models.yaml`;
  document.getElementById('hf-add-form').style.display = 'block';
  document.getElementById('add-status').className = 'status-msg';
  document.getElementById('add-status').style.display = 'none';
  document.getElementById('hf-add-form').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hfCancelAdd() {
  document.getElementById('hf-add-form').style.display = 'none';
}

async function hfConfirmAdd() {
  const dest_dir   = document.getElementById('add-dest-dir').value.trim() || 'misc';
  const tagsRaw    = document.getElementById('add-tags').value.trim();
  const tags       = tagsRaw ? tagsRaw.split(/\s+/).filter(Boolean) : [];
  const vram_gb    = parseFloat(document.getElementById('add-vram').value) || 0;
  const description = document.getElementById('add-description').value.trim();
  const enabled    = document.getElementById('add-enabled').checked;

  const models = [];
  for (const [, item] of hfSelected.entries()) {
    if (!item.filename) {
      setAddStatus('Ошибка: у некоторых записей нет имени файла — используйте «Regex файлов» при поиске', true);
      return;
    }
    models.push({
      repo_id: item.repo_id, filename: item.filename,
      dest_dir, enabled, gated: item.gated,
      tags: tags.length > 0 ? tags : item.tags,
      vram_gb, description,
    });
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

// ═══════════════════════════════════════════════════════════════════
//  Download — SSE client
// ═══════════════════════════════════════════════════════════════════

let _dlEventSource = null;
let _dlLastData = null;

async function dlStart() {
  const startBtn  = document.getElementById('dl-start-btn');
  const cancelBtn = document.getElementById('dl-cancel-btn');
  const panel     = document.getElementById('download-panel');
  const statusMsg = document.getElementById('dl-status-msg');

  startBtn.disabled = true;
  statusMsg.style.display = 'none';
  try {
    const resp = await fetch('/api/download/start', { method: 'POST' });
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

  // Show UI
  startBtn.style.display = 'none';
  cancelBtn.style.display = '';
  cancelBtn.disabled = false;
  panel.style.display = '';
  _dlLastData = null;
  document.getElementById('dl-log').textContent = '';
  document.getElementById('dl-current').textContent = '—';
  document.getElementById('dl-progress-bar').style.width = '0%';
  document.getElementById('dl-counter').textContent = '';

  // Open SSE stream
  if (_dlEventSource) { _dlEventSource.close(); }
  _dlEventSource = new EventSource('/api/download/stream');

  _dlEventSource.onmessage = function(e) {
    let d;
    try { d = JSON.parse(e.data); } catch(_) { return; }

    // Update current model
    if (d.current) {
      document.getElementById('dl-current').textContent = d.current;
    }

    // Update progress bar
    document.getElementById('dl-progress-bar').style.width = (d.progress || 0) + '%';

    // Update counter
    if (d.model_count > 0) {
      document.getElementById('dl-counter').textContent =
        d.done_count + ' / ' + d.model_count;
    }

    // Update log (full replacement to avoid delta sync issues when log is bounded server-side)
    if (d.log && d.log.length > 0) {
      const logEl = document.getElementById('dl-log');
      logEl.textContent = d.log.join('\n');
      logEl.scrollTop = logEl.scrollHeight;
    }

    // Save last snapshot for completion summary
    _dlLastData = d;

    // Done?
    if (!d.running) {
      dlDone();
    }
  };

  _dlEventSource.onerror = function() {
    dlDone();
  };
}

function dlDone() {
  if (_dlEventSource) { _dlEventSource.close(); _dlEventSource = null; }
  const startBtn  = document.getElementById('dl-start-btn');
  const cancelBtn = document.getElementById('dl-cancel-btn');
  const panel     = document.getElementById('download-panel');

  document.getElementById('dl-progress-bar').style.width = '100%';
  cancelBtn.style.display = 'none';
  startBtn.style.display  = '';
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
  const cancelBtn = document.getElementById('dl-cancel-btn');
  cancelBtn.disabled = true;
  try {
    await fetch('/api/download/cancel', { method: 'POST' });
  } catch (_) {}
}

// Init
loadModels();
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


def save_models(config_path: Path, updates: list[dict]) -> None:
    """
    Atomically update 'enabled' field in models.yaml for given models.

    Updates is a list of {repo_id, filename, enabled} dicts.
    Uses tempfile + os.replace() for POSIX atomic write.
    """
    raw = load_yaml(config_path)
    update_map = {
        (u["repo_id"], u["filename"]): bool(u["enabled"])
        for u in updates
    }

    for item in raw.get("models", []) or []:
        if not isinstance(item, dict):
            continue
        key = (item.get("repo_id", ""), item.get("filename", ""))
        if key in update_map:
            item["enabled"] = update_map[key]

    _atomic_yaml_write(config_path, raw)


# ─── HuggingFace Search ────────────────────────────────────────────────────────


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
        "expand": ["siblings"],   # нужно для фильтрации файлов без доп. запросов
    }
    if query:
        kwargs["search"] = query
    if author:
        kwargs["author"] = author
    if pipeline_tag:
        kwargs["pipeline_tag"] = pipeline_tag
    if language:
        kwargs["filter"] = language  # язык передаётся как тег-фильтр (ISO 639-1 код)

    models = list(api.list_models(**kwargs))

    results = []
    for m in models:
        # Фильтрация файлов через siblings (не требует отдельного запроса)
        AI_EXTS = ('.gguf', '.safetensors', '.bin', '.pt', '.pth', '.ckpt')
        if compiled_re is not None:
            siblings = m.siblings or []
            files = [
                {"name": s.rfilename, "size_bytes": getattr(s, "size", None)}
                for s in sorted(siblings, key=lambda x: x.rfilename)
                if compiled_re.search(s.rfilename)
            ]
            if not files:
                continue  # пропускаем репо без совпадений
        else:
            siblings = m.siblings or []
            files = [
                {"name": s.rfilename, "size_bytes": getattr(s, "size", None)}
                for s in sorted(siblings, key=lambda x: x.rfilename)
                if s.rfilename.lower().endswith(AI_EXTS)
            ]

        tags_raw = m.tags or []
        tags = [t for t in tags_raw if ":" not in t][:8]

        results.append({
            "repo_id": m.id,
            "downloads": getattr(m, "downloads", 0) or 0,
            "likes": getattr(m, "likes", 0) or 0,
            "gated": bool(getattr(m, "gated", False)),
            "pipeline_tag": getattr(m, "pipeline_tag", "") or "",
            "tags": tags,
            "files": files,
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
            "dest_dir": str(item.get("dest_dir", "misc") or "misc"),
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
            try:
                while True:
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
                pass  # Client disconnected

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
            # Start background download thread
            t = threading.Thread(
                target=_download_worker,
                args=(self.config_path,),
                daemon=True,
            )
            t.start()
            self._send_json({"status": "started", "model_count": enabled_count})
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

        else:
            self._send_json({"error": "Not found"}, status=404)


# ─── Download Worker ──────────────────────────────────────────────────────────


def _download_worker(config_path: Path) -> None:
    """
    Background thread that runs download_models.py as a subprocess.

    Streams stdout lines into download_state['log'] and parses progress
    markers of the form:
      [repo_id] — начало модели
      [OK]  / [SKIP] / [ERR] — результат
    Updates download_state fields under _dl_lock.
    """
    script = Path(__file__).parent / "download_models.py"
    cmd = [
        sys.executable, str(script),
        "--config", str(config_path),
    ]

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

        done_count = 0
        model_count = download_state["model_count"] or 1  # avoid div-by-zero

        for raw_line in proc.stdout:  # type: ignore[union-attr]
            line = raw_line.rstrip()
            if not line:
                continue

            # Strip tqdm carriage-return lines (progress bar artifacts)
            if "\r" in line:
                # Keep only the last segment after the last \r
                line = line.rsplit("\r", 1)[-1].strip()
            if not line:
                continue

            with _dl_lock:
                if download_state["cancelled"]:
                    break

                download_state["log"].append(line)
                # Keep log bounded
                if len(download_state["log"]) > 500:
                    download_state["log"] = download_state["log"][-400:]

                # Parse model start: lines like "[ModelName]" from download_models.py
                if line.startswith("[") and line.endswith("]") and not line.startswith("[INFO]") \
                        and not line.startswith("[WARN]") and not line.startswith("[ERR") \
                        and not line.startswith("[DOWNLOAD") and not line.startswith("[SKIP") \
                        and not line.startswith("[OK"):
                    download_state["current"] = line[1:-1]

                # Parse result lines
                for marker, status_key in (("[OK]", "DOWNLOAD"), ("[SKIP]", "SKIP"), ("[ERR]", "ERROR")):
                    if f"  {marker}" in line or line.startswith(marker):
                        cur = download_state["current"]
                        if cur:
                            download_state["status_map"][cur] = status_key
                        done_count += 1
                        download_state["done_count"] = done_count
                        pct = int(done_count * 100 / model_count)
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
