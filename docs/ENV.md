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

### Локальная разработка, минимальная конфигурация

```dotenv
# HF_XET_HIGH_PERFORMANCE=1   # раскомментировать для максимальной скорости
```

### Корпоративная сеть с proxy + Yandex Object Storage

```dotenv
PROXY_ENABLED=true
PROXY_URL=http://proxy.corp.example.com:3128

S3_BUCKET=ai-models-prod
S3_REGION=ru-central1
S3_ENDPOINT_URL=https://storage.yandexcloud.net
```

### AWS S3 с автосинхронизацией

```dotenv
S3_BUCKET=my-ai-models
S3_REGION=us-east-1
S3_PREFIX=models
```
