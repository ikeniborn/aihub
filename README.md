# aihub — локальное хранилище AI-моделей

Набор инструментов для скачивания открытых AI-моделей на локальный диск с опциональной выгрузкой в S3-совместимое хранилище.

Основан на исследовании открытой экосистемы моделей 2025–2026 гг. ([docs/](docs/)).

---

## Содержание

- [Требования](#требования)
- [Установка](#установка)
- [Настройка кредов](#настройка-кредов)
- [Быстрый старт](#быстрый-старт)
- [Использование](#использование)
- [S3 / объектное хранилище](#s3--объектное-хранилище)
- [Список моделей (models.yaml)](#список-моделей-modelsyaml)
- [Идемпотентность](#идемпотентность)
- [Структура репозитория](#структура-репозитория)
- [Включённые модели](#включённые-модели)
- [Безопасность](#безопасность)

---

## Требования

- Python 3.9+
- Достаточно места на диске (модели занимают от ~1 ГБ до ~43 ГБ)
- GPU/CPU с объёмом VRAM, указанным в `models.yaml` (для последующего запуска)
- Для закрытых моделей (Llama, Gemma) — токен HuggingFace

---

## Установка

```bash
pip install -r requirements.txt
```

Для ускоренного скачивания (опционально, Rust-based):

```bash
# Включить в credentials.yaml или .env:
# HF_HUB_ENABLE_HF_TRANSFER=1
```

> `hf_transfer` и `boto3` уже включены в `requirements.txt`.

---

## Настройка кредов

Используется чёткое разделение:

| Файл | Что хранить |
|------|-------------|
| `credentials.yaml` | **Секреты**: токены и ключи доступа |
| `.env` | **Параметры**: bucket, region, prefix, endpoint и прочие настройки |

### 1. Секреты → `credentials.yaml`

```bash
cp credentials.yaml.example credentials.yaml
```

```yaml
huggingface:
  token: "hf_ваш_токен"   # для закрытых моделей (Llama, Gemma и др.)

s3:
  access_key_id: "AKIAXXXXXXXXXXXXXXXX"
  secret_access_key: "ваш_секретный_ключ"
```

Токен HuggingFace — создать на: https://huggingface.co/settings/tokens

### 2. Параметры → `.env`

```bash
cp .env.example .env
```

```bash
# Ускоренные загрузки
HF_HUB_ENABLE_HF_TRANSFER=1

# S3: параметры подключения (не секреты)
S3_REGION=us-east-1
S3_BUCKET=my-ai-models
S3_PREFIX=models
S3_ENDPOINT_URL=                 # пусто для AWS; для Yandex/MinIO — см. .env.example
```

### Приоритет загрузки

```
credentials.yaml (секреты)  +  .env (конфиг)  →  переменные окружения (fallback для CI/CD)
```

Переменные окружения (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `HF_TOKEN`) работают как запасной вариант — удобно для CI/CD где монтировать файл не нужно.

---

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Настроить креды
cp credentials.yaml.example credentials.yaml
# отредактировать credentials.yaml

# 3. Посмотреть список моделей
python scripts/download_models.py --list

# 4. Проверить без скачивания
python scripts/download_models.py --dry-run

# 5. Скачать все включённые модели
python scripts/download_models.py
```

---

## Использование

### Просмотр списка моделей

```bash
python scripts/download_models.py --list
```

Выводит таблицу со статусом (enabled/disabled), тегами, требуемым VRAM и описанием.

### Скачать все включённые модели

```bash
python scripts/download_models.py
```

### Проверка без скачивания (dry-run)

Проверяет доступность репозиториев на HuggingFace без скачивания файлов:

```bash
python scripts/download_models.py --dry-run
```

### Фильтрация по имени модели

Substring-поиск по `repo_id` и `filename`:

```bash
python scripts/download_models.py --model Phi-4
python scripts/download_models.py --model Qwen2.5
python scripts/download_models.py --model bge-m3
```

### Фильтрация по тегу

```bash
python scripts/download_models.py --tag russian       # русскоязычные модели
python scripts/download_models.py --tag embeddings    # модели эмбеддингов
python scripts/download_models.py --tag reasoning     # reasoning-модели
python scripts/download_models.py --tag code          # модели для кода
```

### Принудительное перескачивание

Игнорирует ETag и перекачивает файлы заново:

```bash
python scripts/download_models.py --force
python scripts/download_models.py --force --model bge-m3
```

### Включить отключённые модели

```bash
python scripts/download_models.py --include-disabled
```

### Альтернативный конфиг

```bash
python scripts/download_models.py --config my-selection.yaml
```

### Альтернативный файл кредов

```bash
python scripts/download_models.py --creds prod-credentials.yaml
```

---

## Поиск моделей (browse_models.py)

Скрипт для поиска моделей на HuggingFace Hub по API с regex-фильтрацией.

```bash
# Поиск по ключевому слову
python scripts/browse_models.py --query "qwen"

# Все модели автора с regex-фильтром по имени
python scripts/browse_models.py --author bartowski --regex ".*14B.*"

# По тегам + фильтр по именам файлов
python scripts/browse_models.py --tags gguf --file-regex "Q4_K_M\.gguf$"

# Показать файлы внутри репозиториев
python scripts/browse_models.py --author unsloth --show-files

# Вывести YAML-фрагмент для вставки в models.yaml
python scripts/browse_models.py --query "embedding russian" --file-regex "\.gguf$" --yaml

# Сортировка и лимит
python scripts/browse_models.py --query "llama" --sort likes --limit 10
```

### Параметры `browse_models.py`

| Флаг | Описание |
|------|----------|
| `--query TEXT` | Полнотекстовый поиск |
| `--author NAME` | Фильтр по автору/организации |
| `--tags TAG ...` | Фильтр по тегам (например: `gguf text-generation`) |
| `--regex PATTERN` | Regex по model_id (после API-запроса) |
| `--file-regex PATTERN` | Regex по именам файлов внутри репо |
| `--show-files` | Показать список файлов каждого репо |
| `--sort` | Сортировка: `downloads` / `likes` / `lastModified` |
| `--limit N` | Максимум результатов (по умолчанию: 20) |
| `--yaml` | Вывод YAML-фрагмента для `models.yaml` |
| `--creds FILE` | Файл секретов (по умолчанию: `credentials.yaml`) |

---

## Веб-интерфейс (model_browser.py)

Браузер моделей с веб-интерфейсом для просмотра и управления списком в `models.yaml`.

```bash
python scripts/model_browser.py
```

Открыть в браузере: **http://localhost:9000**

### Возможности

- Таблица всех моделей из `models.yaml` с реальным размером файла на диске
- Визуальные индикаторы: зелёная точка — скачана, серая — не скачана, тусклая строка — отключена
- Фильтрация: поиск по тексту, фильтр по тегам, переключатель «только скачанные»
- Чекбоксы `enabled` для включения/отключения моделей
- Кнопка «Сохранить» — атомарно обновляет `models.yaml` (без прерывания сохранения)
- Без внешних зависимостей: только Python stdlib (`http.server`, `json`, `pathlib`)

### Параметры запуска

```bash
python scripts/model_browser.py                        # порт 9000, config models.yaml
python scripts/model_browser.py --port 8080            # другой порт
python scripts/model_browser.py --config my.yaml       # другой конфиг
```

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--port PORT` | `9000` | Порт HTTP-сервера |
| `--host HOST` | `0.0.0.0` | Адрес для привязки |
| `--config FILE` | `models.yaml` | Путь к конфигу |

### API

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/api/models` | JSON-список всех моделей с полем `disk_size_bytes` и `downloaded` |
| `POST` | `/api/save` | Обновить `enabled` в `models.yaml`; тело: `{"updates": [{repo_id, filename, enabled}]}` |

---

## Retry и rate-limit

Скрипт автоматически повторяет попытку при сетевых ошибках и HTTP 429.

### Настройка в `models.yaml`

```yaml
settings:
  retry_count: 3            # количество повторных попыток
  retry_delay: 5.0          # базовая задержка (сек); backoff: 5 → 10 → 20
                            # при HTTP 429 задержка x10 (т.е. 50s по умолчанию)
  inter_download_delay: 0   # пауза между моделями (секунды)
```

### Переопределение через CLI

```bash
# 5 попыток, 10с базовая задержка, 2с пауза между моделями
python scripts/download_models.py --retries 5 --retry-delay 10 --delay 2

# Без повторов и пауз (быстрый режим)
python scripts/download_models.py --retries 0 --delay 0
```

---

## S3 / объектное хранилище

Поддерживаемые провайдеры: **AWS S3, MinIO, Yandex Object Storage, Cloudflare R2, Backblaze B2**.

### Скачать локально и загрузить в S3

```bash
python scripts/download_models.py --upload-s3
```

Модели сохраняются на диск **и** копируются в S3. При повторном запуске уже загруженные в S3 объекты пропускаются (сравнение по размеру файла).

### Только в S3 (без постоянного хранения на диске)

```bash
python scripts/download_models.py --s3-only
```

Файлы скачиваются во временную директорию, после успешной выгрузки в S3 — удаляются. Удобно для CI/CD-сценариев где локальный диск не нужен.

### Комбинация флагов

```bash
# Загрузить только русские модели в S3
python scripts/download_models.py --upload-s3 --tag russian

# Принудительно перезалить все embeddings в S3
python scripts/download_models.py --upload-s3 --tag embeddings --force

# Dry-run для проверки S3-конфигурации
python scripts/download_models.py --upload-s3 --dry-run
```

### Формат ключей в S3

```
{prefix}/{dest_dir}/{filename}

Пример:
  models/llm/llama/Llama-3.1-8B-Instruct-Q4_K_M.gguf
  models/llm/qwen/Qwen2.5-14B-Instruct-Q4_K_M.gguf
  models/embeddings/bge-m3-Q8_0.gguf
```

### Пример для Yandex Object Storage

```yaml
# credentials.yaml
s3:
  access_key_id: "ваш_key_id"
  secret_access_key: "ваш_secret_key"
  region: "ru-central1"
  endpoint_url: "https://storage.yandexcloud.net"
  bucket: "ai-models-bucket"
  prefix: "models"
```

### Пример для MinIO

```yaml
# credentials.yaml
s3:
  access_key_id: "minioadmin"
  secret_access_key: "minioadmin"
  region: "us-east-1"
  endpoint_url: "http://localhost:9000"
  bucket: "ai-models"
  prefix: "models"
```

---

## Список моделей (`models.yaml`)

Редактируйте `models.yaml` для управления списком загрузки:

```yaml
settings:
  models_dir: ./models      # корневая директория хранения
  update_policy: etag       # etag | skip | always
  s3:
    sync_after_download: false   # автовыгрузка в S3 (или --upload-s3 в CLI)

models:
  - repo_id: bartowski/Qwen2.5-14B-Instruct-GGUF
    filename: Qwen2.5-14B-Instruct-Q4_K_M.gguf
    dest_dir: llm/qwen
    enabled: true             # false — пропускать без удаления из списка
    gated: false              # true — требует HF_TOKEN + принятия лицензии
    tags: [llm, chat, russian, 14b]
    vram_gb: 9
    description: "Qwen 2.5 14B — Alibaba, Apache 2.0"
```

### Поля модели

| Поле          | Обязательно | Описание |
|---------------|-------------|----------|
| `repo_id`     | Да          | HuggingFace идентификатор `owner/repo` |
| `filename`    | Да          | Точное имя файла в репозитории (обычно `*.gguf`) |
| `dest_dir`    | Нет         | Поддиректория внутри `models/` (по умолчанию: `misc`) |
| `enabled`     | Нет         | `false` — пропускать (по умолчанию: `true`) |
| `gated`       | Нет         | `true` — модель закрытая, требует токен и принятие лицензии |
| `tags`        | Нет         | Теги для фильтрации через `--tag` |
| `vram_gb`     | Нет         | Минимальный объём VRAM (информационно) |
| `description` | Нет         | Описание модели |

### Политики обновления (`update_policy`)

| Значение | Поведение |
|----------|-----------|
| `etag`   | Проверяет ETag через HTTP HEAD; перекачивает только если файл изменился (**рекомендуется**) |
| `skip`   | Никогда не перекачивает если файл есть (игнорирует обновления) |
| `always` | Всегда перекачивает (аналог `--force`) |

---

## Идемпотентность

Скрипт полностью идемпотентен — безопасно запускать повторно:

**Для локального хранилища:**
- При каждом запуске проверяется remote ETag через HTTP HEAD (без скачивания данных)
- ETag сохраняется в файл-спутник рядом с моделью: `model.gguf.etag`
- Если ETag совпадает — файл пропускается
- При обнаружении изменения ETag — перекачивается автоматически

**Для S3:**
- Проверяется наличие объекта через `HEAD Object`
- Сравнивается `ContentLength` объекта с размером локального файла
- При совпадении — выгрузка пропускается

**Атомарность загрузки:**
- `hf_hub_download()` пишет во временный файл и переименовывает по завершении
- Прерванная загрузка не оставляет битых файлов

---

## Структура репозитория

```
aihub/
├── models.yaml                  # Список моделей (редактировать для добавления/отключения)
├── requirements.txt             # Зависимости Python
├── credentials.yaml.example     # Шаблон секретов → скопировать в credentials.yaml
├── credentials.yaml             # Ваши токены и ключи (gitignored, НЕ коммитить!)
├── .env.example                 # Шаблон параметров конфигурации
├── .env                         # Ваши параметры (gitignored)
├── scripts/
│   ├── download_models.py       # Скачивание моделей (HuggingFace → локально/S3)
│   ├── browse_models.py         # Поиск моделей на HuggingFace Hub по API
│   └── model_browser.py         # Веб-интерфейс для просмотра локальных моделей (порт 9000)
├── docs/
│   └── *.md                     # Исследование экосистемы моделей
└── models/                      # Скачанные модели (gitignored)
    ├── llm/
    │   ├── llama/
    │   ├── qwen/
    │   ├── deepseek/
    │   ├── phi/
    │   ├── russian/
    │   └── code/
    ├── embeddings/
    └── image_gen/
```

---

## Включённые модели

Полный список с аннотациями — в `models.yaml`. По умолчанию включены:

| Модель | Категория | VRAM | Лицензия |
|--------|-----------|------|----------|
| Llama 3.1 8B Instruct Q4_K_M | LLM / чат | 5 ГБ | Llama Community |
| Qwen 2.5 14B Instruct Q4_K_M | LLM / чат, русский | 9 ГБ | Apache 2.0 |
| DeepSeek R1 Distill Qwen 14B Q4_K_M | LLM / рассуждение | 10 ГБ | MIT / DeepSeek |
| Phi-4 14B Q4_K_M | LLM / рассуждение + код | 8 ГБ | MIT |
| Saiga Llama3 8B Q4_K | LLM / русский | 5 ГБ | Apache 2.0 |
| T-Lite 7B Q4_K_M | LLM / русский | 5 ГБ | Apache 2.0 |
| Qwen2.5-Coder 14B Instruct Q4_K_M | LLM / код | 9 ГБ | Apache 2.0 |
| BGE-M3 Q8_0 | Эмбеддинги / RAG | 1 ГБ | Apache 2.0 |
| Nomic Embed Text v1.5 Q8_0 | Эмбеддинги / RAG | <1 ГБ | Apache 2.0 |

По умолчанию **отключены** (требуют ≥20 ГБ VRAM или предназначены для специфических задач):
Qwen 2.5 32B, DeepSeek R1 32B, Qwen2.5-Coder 32B, GigaChat3 10B MoE, FLUX.1-schnell, SDXL.

---

## Безопасность

| Файл | Статус в git | Содержимое |
|------|-------------|-----------|
| `credentials.yaml` | **gitignored** | Секреты: токены, ключи — **никогда не коммитить** |
| `credentials.yaml.example` | Коммитится | Шаблон секретов без реальных значений |
| `.env` | **gitignored** | Параметры конфигурации (bucket, region, prefix и др.) |
| `.env.example` | Коммитится | Шаблон параметров конфигурации |
| `models/` | **gitignored** | Бинарные файлы моделей |
| `*.gguf`, `*.safetensors` | **gitignored** | Веса моделей |
| `*.etag` | **gitignored** | ETag-файлы состояния загрузки |

> **Правило**: `credentials.yaml` и `.env` должны оставаться только локально. Проверить:
> ```bash
> git status  # эти файлы не должны появляться здесь
> ```
