# aihub

Инструменты для загрузки открытых AI-моделей на локальный диск и синхронизации в S3-хранилище.

Подробное руководство: [docs/GUIDES.md](docs/GUIDES.md)

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
make list                                       # показать список моделей
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
```

### Параметры → `.env` (gitignored)

```bash
HF_HUB_ENABLE_HF_TRANSFER=1   # ускоренные загрузки
S3_BUCKET=my-bucket
S3_REGION=us-east-1
S3_PREFIX=models
S3_ENDPOINT_URL=               # пусто для AWS; для Yandex/MinIO — endpoint URL
```

Приоритет: `credentials.yaml` → `.env` → переменные окружения.

---

## Скрипты

### `download_models.py` — загрузка моделей

```
--list                  показать список моделей
--dry-run               проверить без загрузки
--model SUBSTRING       фильтр по имени модели
--tag TAG               фильтр по тегу
--force                 перезагрузить даже если файл есть
--include-disabled      включить отключённые модели
--upload-s3             загрузить в S3 после скачивания
--s3-only               только S3 (без постоянного хранения на диске)
--retries N             количество повторов (default: 3)
--retry-delay SECS      базовая задержка между повторами (default: 5s)
--delay SECS            пауза между моделями (default: 0)
--config FILE           путь к models.yaml
--creds FILE            путь к credentials.yaml
```

HTTP 429 (rate limit) обрабатывается автоматически: задержка ×10 от базовой.

### `browse_models.py` — поиск на HuggingFace Hub

```
--query TEXT            полнотекстовый поиск
--author NAME           фильтр по автору
--tags TAG ...          фильтр по тегам
--regex PATTERN         regex по model_id
--file-regex PATTERN    regex по именам файлов внутри репо
--show-files            показать файлы каждого репо
--yaml                  вывод YAML-фрагмента для models.yaml
--sort downloads|likes|lastModified
--limit N               макс. результатов (default: 20)
```

### `model_browser.py` — веб-интерфейс

```bash
python scripts/model_browser.py            # http://localhost:9000
python scripts/model_browser.py --port 8080 --host 0.0.0.0
```

Просмотр и управление моделями в браузере. Показывает размер файла на диске, фильтрация по тексту/тегам, чекбоксы `enabled`, сохранение в `models.yaml`.

---

## `models.yaml` — список моделей

```yaml
settings:
  models_dir: ./models
  update_policy: etag        # etag | skip | always
  retry_count: 3
  retry_delay: 5.0
  inter_download_delay: 0

models:
  - repo_id: bartowski/Qwen2.5-14B-Instruct-GGUF
    filename: Qwen2.5-14B-Instruct-Q4_K_M.gguf
    dest_dir: llm/qwen
    enabled: true
    gated: false
    tags: [llm, chat, russian, 14b]
    vram_gb: 9
    description: "Qwen 2.5 14B — Alibaba, Apache 2.0"
```

---

## Структура

```
aihub/
├── models.yaml                  # список моделей
├── requirements.txt             # зависимости (для совместимости с CI/CD)
├── pyproject.toml               # конфигурация проекта (PEP 517, uv-совместимый)
├── Makefile                     # команды: setup, download, browse, ui, list, update
├── .python-version              # версия Python (3.9)
├── credentials.yaml.example     # шаблон секретов
├── .env.example                 # шаблон параметров
├── scripts/
│   ├── utils.py                 # общие утилиты (fmt_size, load_hf_token)
│   ├── download_models.py       # загрузка моделей → локально / S3
│   ├── browse_models.py         # поиск на HuggingFace Hub
│   └── model_browser.py         # веб-интерфейс (порт 9000)
├── docs/
│   └── GUIDES.md                # пошаговое руководство пользователя
├── .venv/                       # виртуальное окружение (gitignored, создаётся make setup)
└── models/                      # скачанные модели (gitignored)
```

---

## Безопасность

| Файл | git | Содержимое |
|------|-----|-----------|
| `credentials.yaml` | **ignored** | токены, ключи — никогда не коммитить |
| `.env` | **ignored** | параметры конфигурации |
| `models/` | **ignored** | бинарные веса моделей |
| `*.etag` | **ignored** | файлы состояния загрузки |
