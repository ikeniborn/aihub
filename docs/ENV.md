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
# Токен для gated-моделей (предпочтительно — через credentials.yaml)
HF_TOKEN=hf_...

# Быстрые загрузки через Rust-библиотеку hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1
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

## Пример `.env` для корпоративной сети с proxy

```dotenv
PROXY_ENABLED=true
PROXY_URL=http://proxy.corp.example.com:3128

S3_BUCKET=ai-models-prod
S3_REGION=eu-central-1
S3_ENDPOINT_URL=https://storage.yandexcloud.net
```
