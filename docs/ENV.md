# Переменные окружения (.env)

Скопируйте этот файл в `.env` в корень проекта и раскомментируйте нужные строки.

> **Секреты** (HF_TOKEN, AWS ключи) — в `credentials.yaml`, а не в `.env`.
> `.env` хранит только параметры подключения и feature flags.

---

## Proxy

Управляется двумя переменными. Прокси применяется ко всем исходящим запросам:
HuggingFace Hub (загрузка моделей, API поиска) и S3-хранилищам.

```dotenv
# Включить proxy: true / false (по умолчанию: false)
PROXY_ENABLED=true

# Адрес прокси-сервера
PROXY_URL=http://proxy.example.com:8080
```

**Поддерживаемые схемы `PROXY_URL`:**

| Схема | HuggingFace Hub | S3 (boto3) |
|-------|:-:|:-:|
| `http://host:port` | ✅ | ✅ |
| `https://host:port` | ✅ | ✅ |
| `http://user:pass@host:port` | ✅ | ✅ |
| `socks5://host:port` | ✅ | ✅ |
| `socks5h://host:port` | ✅ | ✅ |

> `socks5h://` — SOCKS5 с DNS-резолвингом на стороне прокси (remote DNS).
> `socks5s://` — не является стандартной схемой, не поддерживается.

**Альтернатива через `credentials.yaml`** (ниже приоритетом чем `.env`):
```yaml
proxy:
  enabled: true
  url: "http://proxy.example.com:8080"
```

---

## HuggingFace

```dotenv
# Максимальная скорость через hf-xet (протокол по умолчанию с 2025 г.)
# Включает агрессивные настройки параллельности и кэширования.
# Не устанавливайте, если хотите ограничить пропускную способность.
# HF_XET_HIGH_PERFORMANCE=1
```

> **Устаревшее:** `# HF_XET_HIGH_PERFORMANCE=1   # раскомментировать для максимальной скорости` — не имеет эффекта когда установлен hf-xet,
> и вызывает предупреждение в логе. Удалите эту строку из `.env`.
> Аналог для hf-xet: `HF_XET_HIGH_PERFORMANCE=1` (раскомментируйте выше при необходимости).

> **HF_TOKEN** — только в `credentials.yaml` (секция `huggingface.token`), не в `.env`.

---

## Контроль нагрузки на канал

> **Эти параметры НЕ задаются через `.env`** — они находятся в секции `settings:` файла `models.yaml`.
> В `.env` есть только `HF_HUB_ENABLE_HF_TRANSFER`. Остальное — ниже.

```yaml
# models.yaml → settings:
hf_download_concurrency: 4    # макс. параллельных TCP-соединений HuggingFace Xet
                               # null = без ограничений (может открыть 49+ соединений!)
download_timeout_hours: 2     # таймаут на один файл в часах; 0 = без ограничений
bandwidth_limit_mbps: null    # лимит скорости в Mbit/s для стандартного HTTP (не Xet)
```

Переопределить для одного запуска через CLI или через **веб-интерфейс** (панель перед кнопкой «Скачать enabled»):

```bash
python scripts/download_models.py --max-concurrency 2 --download-timeout 3
```

| Параметр | Где задаётся | Влияет на |
|---|---|---|
| `hf_download_concurrency` | `models.yaml` / `--max-concurrency` / UI | Xet-протокол (параллельные TCP) |
| `download_timeout_hours` | `models.yaml` / `--download-timeout` / UI | Таймаут на один файл |
| `bandwidth_limit_mbps` | `models.yaml` / `--bandwidth-limit` | Только стандартный HTTP, не Xet |

> Для жёсткого ограничения Xet используйте `hf_download_concurrency` или `wondershaper` на уровне ОС.
> Подробнее: [docs/GUIDES.md → раздел 9](GUIDES.md#9-контроль-нагрузки-на-канал)

---

## Поиск моделей (browse_models.py)

Переменные задают дефолты для `browse_models.py`; CLI-аргументы всегда имеют приоритет.

```dotenv
# Задача модели — что умеет делать модель (pipeline_tag на HuggingFace).
# Значения: text-generation | text-to-image | automatic-speech-recognition |
#           sentence-similarity | text-classification | text-to-speech |
#           translation | image-classification | zero-shot-classification
# CLI: --pipeline-tag
BROWSE_PIPELINE_TAG=text-generation

# Формат / библиотека — в каком виде хранится модель.
# Значения: gguf | safetensors | transformers | diffusers |
#           onnx | sentence-transformers | mlx | openvino
# CLI: --library
BROWSE_LIBRARY=gguf

# Язык модели (ISO 639-1). Пусто — без фильтра по языку.
# Значения: ru | en | zh | de | fr | ja | ko | ...
# CLI: --language
# BROWSE_LANGUAGE=ru

# Regex-фильтр по именам файлов внутри репозитория.
# CLI: --file-regex
# Примеры:
#   Q4_K_M\.gguf$          — только Q4_K_M квантизация
#   Q[458]_K_[MS]\.gguf$   — Q4/Q5/Q8 в вариантах K_M и K_S
#   \.safetensors$          — только safetensors
BROWSE_FILE_REGEX=Q4_K_M\.gguf$
```

### Пример: дефолты для работы с русскоязычными GGUF LLM

```dotenv
BROWSE_PIPELINE_TAG=text-generation
BROWSE_LIBRARY=gguf
BROWSE_LANGUAGE=ru
BROWSE_FILE_REGEX=Q4_K_M\.gguf$
```

После настройки достаточно:

```bash
python scripts/browse_models.py --author bartowski   # применит все 4 дефолта
python scripts/browse_models.py --query "saiga"      # то же самое
```

### Пример: дефолты для image generation

```dotenv
BROWSE_PIPELINE_TAG=text-to-image
BROWSE_LIBRARY=safetensors
# BROWSE_FILE_REGEX — оставить пустым, safetensors-репо часто без лишних файлов
```

---

## S3 / Object Storage

```dotenv
# Имя bucket (обязательно для --upload-s3 и --s3-only)
S3_BUCKET=my-ai-models

# Регион (по умолчанию: us-east-1)
S3_REGION=us-east-1

# Endpoint для S3-совместимых хранилищ (оставить пустым для AWS S3)
# S3_ENDPOINT_URL=https://storage.yandexcloud.net
# S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
# S3_ENDPOINT_URL=http://localhost:9000

# Префикс внутри bucket (по умолчанию: models)
S3_PREFIX=models

# Fallback-ключи S3 (предпочтительно — через credentials.yaml)
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
```

---

## Примеры готовых `.env`

### Локальная разработка, только GGUF LLM

```dotenv
# HF_XET_HIGH_PERFORMANCE=1   # раскомментировать для максимальной скорости

BROWSE_PIPELINE_TAG=text-generation
BROWSE_LIBRARY=gguf
BROWSE_FILE_REGEX=Q4_K_M\.gguf$
```

### Корпоративная сеть с proxy + Yandex Object Storage

```dotenv
PROXY_ENABLED=true
PROXY_URL=http://proxy.corp.example.com:3128

S3_BUCKET=ai-models-prod
S3_REGION=ru-central1
S3_ENDPOINT_URL=https://storage.yandexcloud.net

BROWSE_PIPELINE_TAG=text-generation
BROWSE_LIBRARY=gguf
BROWSE_LANGUAGE=ru
BROWSE_FILE_REGEX=Q4_K_M\.gguf$
```

### Image generation (ComfyUI / A1111)

```dotenv
# HF_XET_HIGH_PERFORMANCE=1   # раскомментировать для максимальной скорости

BROWSE_PIPELINE_TAG=text-to-image
BROWSE_LIBRARY=safetensors
```
