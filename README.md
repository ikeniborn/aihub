# aihub

Инструменты для загрузки открытых AI-моделей на локальный диск и синхронизации в S3-хранилище.
Поддерживаются источники: **HuggingFace Hub** и **Ollama Registry**.

---

## Требования

- Python 3.9+
- `make setup` — создаёт `.venv` и устанавливает все зависимости автоматически

> Альтернатива без make: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`

---

## Быстрый старт

```bash
make setup                                      # создать .venv и установить зависимости
cp credentials.yaml.example credentials.yaml   # добавить токены
cp .env.example .env                            # добавить S3-параметры
cp models.yaml.example models.yaml             # создать список моделей
make ui                                         # открыть веб-интерфейс → http://localhost:9009
make download                                   # загрузка всех включённых моделей
```

---

## Конфигурация

### Секреты → `credentials.yaml` (gitignored)

```yaml
huggingface:
  token: "hf_..."          # для закрытых моделей (Llama, Gemma)
s3:
  access_key_id: "..."
  secret_access_key: "..."
proxy:                     # опционально; .env имеет приоритет
  enabled: false
  url: ""
```

### Параметры → `.env` (gitignored)

```bash
HF_HUB_ENABLE_HF_TRANSFER=1   # ускоренные загрузки HuggingFace

# Proxy (для корпоративных сетей и фаерволов)
PROXY_ENABLED=true
PROXY_URL=http://proxy.example.com:8080

S3_BUCKET=my-bucket
S3_REGION=us-east-1
S3_PREFIX=models
S3_ENDPOINT_URL=               # пусто для AWS; для Yandex/MinIO — endpoint URL
```

Приоритет: `credentials.yaml` → `.env` → переменные окружения.

---

## Веб-интерфейс (`model_browser.py`)

```bash
python scripts/model_browser.py                         # http://localhost:9009
python scripts/model_browser.py --port 8080 --host 0.0.0.0
make ui                                                 # с автооткрытием браузера
make ui PORT=8080 HOST=0.0.0.0
```

### Управление моделями

- Модели группируются по `dest_dir` в раскрываемую иерархию (до 3 уровней)
- Фильтрация по тексту, тегам, флагу «только скачанные»
- Чекбокс `enabled` + кнопка **Сохранить** — атомарная запись в `models.yaml`
- Кнопка ✎ — инлайн-редактирование атрибутов (теги, описание, `dest_dir`, gated); изменение `dest_dir` перемещает файл на диске
- Кнопка ✕ — удаление записи из `models.yaml` и файла с диска

### Поиск HuggingFace

- Поиск по запросу, автору, типу pipeline, языку, regex файлов
- Фильтр по размеру файла (МБ)
- Предзаполнение `dest_dir` / тегов / описания при добавлении в `models.yaml`

### Поиск Ollama

- Поиск по названию на `ollama.com`
- Отображение возможностей модели: `tools`, `thinking`, `vision`, `embedding`, `cloud`
- Фильтрация результатов по возможностям (с счётчиками по каждому тегу)
- **Пред-фильтры** (применяются до нажатия «Искать»): квантизация (Q4_K_M, Q8_0, F16 …), размер (≤4 ГБ, ≤8 ГБ …), контекст (8K+, 32K+, 128K+ …), тип входных данных (text, text+img, text+audio)
- **Плоский список вариантов**: все теги модели раскрываются автоматически; каждый вариант — отдельная строка с колонками Размер, Контекст, Input, Загрузки, Теги
- Размер загружается мгновенно из HTML-страницы тегов ollama.com (1 запрос на модель вместо 300+ запросов к OCI-реестру)
- Загрузка с поддержкой HTTP Resume: прерванные загрузки продолжаются с точки останова

### Загрузка

- Кнопка **Скачать** запускает фоновый процесс загрузки всех включённых моделей
- Прогресс-бар и лог в реальном времени (SSE stream)
- Кнопка **Отмена** для остановки

---

## Использование скачанных Ollama-моделей

aihub скачивает GGUF из OCI-реестра в `models/`. Чтобы использовать их в `ollama run`,
зарегистрируйте модели через веб-интерфейс или CLI:

```bash
# Через UI: кнопка →Ollama в строке модели (вкладка «Мои модели»)
make ui

# Через CLI: скачать и зарегистрировать в Ollama
python scripts/download_models.py --register-ollama
```

После регистрации локальный GGUF в `models/` можно удалить (кнопка **✕ → Только локально**).

Подробнее — см. [docs/GUIDES.md §11](docs/GUIDES.md#11-использование-скачанных-ollama-моделей).

---

## CLI-скрипты

### `download_models.py` — загрузка моделей

```bash
python scripts/download_models.py [OPTIONS]
make download
```

```
--list                  показать список моделей
--dry-run               проверить без загрузки
--model SUBSTRING       фильтр по имени модели
--tag TAG               фильтр по тегу
--force                 перезагрузить даже если файл есть
--include-disabled      включить отключённые модели
--upload-s3             загрузить в S3 после скачивания
--s3-only               только S3 (без постоянного хранения на диске)
--register-ollama       зарегистрировать Ollama-модели в Ollama после скачивания
--retries N             количество повторов (default: 3)
--retry-delay SECS      базовая задержка между повторами (default: 5s)
--delay SECS            пауза между моделями (default: 0)
--max-concurrency N     макс. параллельных TCP-соединений HuggingFace Xet
--bandwidth-limit MBIT  ограничение скорости в Mbit/s
--download-timeout HRS  таймаут на один файл в часах (default: 2)
--config FILE           путь к models.yaml
--creds FILE            путь к credentials.yaml
```

HTTP 429 (rate limit) обрабатывается автоматически: задержка ×10 от базовой.
Одновременно может работать только один процесс загрузки (PID lock `.download.lock`).

---

## `models.yaml` — список моделей

Создаётся из шаблона: `cp models.yaml.example models.yaml`
Файл gitignored — персональный список моделей не попадает в репозиторий.

```yaml
settings:
  models_dir: ./models
  update_policy: etag        # etag | skip | always
  retry_count: 3
  retry_delay: 5.0
  inter_download_delay: 60
  hf_download_concurrency: 1    # макс. параллельных TCP при Xet-протоколе (null = без ограничений)
  download_timeout_hours: 2     # таймаут на файл в часах; 0 = без ограничений
  bandwidth_limit_mbps: null    # лимит Mbit/s для стандартного HTTP (null = без ограничений)
  s3:
    sync_after_download: false  # автоматически синхронизировать в S3 после загрузки
```

### HuggingFace-запись

```yaml
models:
  - repo_id: bartowski/Qwen2.5-14B-Instruct-GGUF
    filename: Qwen2.5-14B-Instruct-Q4_K_M.gguf
    dest_dir: llm/qwen
    enabled: true
    gated: false
    tags: [llm, chat, 14b]
    description: "Qwen 2.5 14B — Alibaba, Apache 2.0"
```

### Ollama-запись

```yaml
models:
  - repo_id: ollama/qwen3
    filename: qwen3.gguf          # локальное имя файла (тег latest)
    dest_dir: llm/qwen
    enabled: true
    source: ollama
    ollama_model: qwen3           # модель без тега → загружается latest
    tags: [llm, chat, thinking]
    description: "Qwen 3 — Alibaba"

  - repo_id: ollama/qwen3
    filename: qwen3-7b.gguf       # конкретный вариант
    dest_dir: llm/qwen
    enabled: true
    source: ollama
    ollama_model: qwen3:7b        # формат model:tag
```

Ollama-модели загружаются через OCI-реестр `registry.ollama.ai` в формате GGUF.
Поле `ollama_model` задаёт тег: `model` (latest) или `model:tag` (конкретный вариант).

Автогенерируемые поля (не указывать вручную):
- `s3_synced: true` / `s3_key: "..."` — проставляются после синхронизации в S3

---

## Структура

```
aihub/
├── models.yaml                  # список моделей
├── requirements.txt             # зависимости
├── pyproject.toml               # конфигурация проекта (PEP 517)
├── Makefile                     # setup, download, ui, list, update, security-check
├── credentials.yaml.example     # шаблон секретов
├── .env.example                 # шаблон параметров
├── models.yaml.example          # шаблон списка моделей (скопировать в models.yaml)
├── scripts/
│   ├── utils.py                 # общие утилиты (fmt_size, ProxyConfig)
│   ├── download_models.py       # загрузка моделей → локально / S3
│   ├── model_browser.py         # веб-интерфейс (порт 9009)
│   └── ollama_hub.py            # Ollama OCI client: поиск, теги, загрузка GGUF
├── docs/
│   └── GUIDES.md                # пошаговое руководство пользователя
├── .venv/                       # виртуальное окружение (gitignored)
└── models/                      # скачанные модели (gitignored)
```

---

## Безопасность

| Файл | git | Содержимое |
|------|-----|-----------|
| `credentials.yaml` | **ignored** | токены, ключи — никогда не коммитить |
| `models.yaml` | **ignored** | персональный список моделей |
| `.env` | **ignored** | параметры конфигурации |
| `models/` | **ignored** | бинарные веса моделей |
| `*.etag` | **ignored** | состояние загрузки (дедупликация) |
| `.download.lock` | **ignored** | PID-файл защиты от параллельного запуска |

```bash
chmod 600 credentials.yaml   # обязательно после создания
make security-check   # проверить права, .gitignore (credentials.yaml, models.yaml, .env) и токены в коде
```

Пароли в `PROXY_URL` автоматически маскируются в логах: `http://user:***@host:port`.
