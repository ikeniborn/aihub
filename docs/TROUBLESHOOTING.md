# Устранение неполадок (Troubleshooting)

Частые ошибки при работе с `make download` и способы их устранения.

---

## Содержание

1. [404 Repository Not Found](#1-404-repository-not-found)
2. [401 Unauthorized — неверный токен](#2-401-unauthorized--неверный-токен)
3. [403 Forbidden — gated-модель без принятия лицензии](#3-403-forbidden--gated-модель-без-принятия-лицензии)
4. [hf_transfer: ложные 404 и зависания](#4-hf_transfer-ложные-404-и-зависания)
5. [Создание токена HuggingFace с нужными правами](#5-создание-токена-huggingface-с-нужными-правами)
6. [Файл скачивается, но пустой или обрезанный](#6-файл-скачивается-но-пустой-или-обрезанный)
7. [Rate limit 429 — слишком много запросов](#7-rate-limit-429--слишком-много-запросов)
8. [Credentials not found — токен не читается](#8-credentials-not-found--токен-не-читается)

---

## 1. 404 Repository Not Found

### Симптом

```
[WARN] Ошибка (попытка 1/4): 404 Client Error.
Repository Not Found for url: https://huggingface.co/.../resolve/main/model.gguf
```

### Причины и решения

**A. Имя файла изменилось в репозитории**

Авторы периодически переименовывают квантизации. Проверь актуальные имена файлов:

```bash
.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
for f in api.list_repo_files('bartowski/Llama-3.1-8B-Instruct-GGUF'):
    print(f)
"
```

Обнови `filename` в `models.yaml` под актуальное имя.

**B. Репозиторий перемещён или удалён**

Открой страницу репозитория в браузере:
```
https://huggingface.co/<repo_id>
```
Если страница не найдена — обнови `repo_id` в `models.yaml`.

**C. hf_transfer возвращает ложный 404**

Отключи быстрый загрузчик — см. раздел [4. hf_transfer](#4-hf_transfer-ложные-404-и-зависания).

---

## 2. 401 Unauthorized — неверный токен

### Симптом

```
401 Client Error: Unauthorized
Invalid credentials in Authorization header
```

### Решение

Проверь токен в `credentials.yaml`:

```yaml
huggingface:
  token: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Токен должен начинаться с `hf_`. Длина — обычно 37+ символов.

Сгенерируй новый токен: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → тип **Read**.

---

## 3. 403 Forbidden — gated-модель без принятия лицензии

### Симптом

```
403 Client Error: Forbidden
Access to model <repo_id> is restricted. You must accept the license.
```

### Причина

Модель помечена как **gated** — доступ только после принятия лицензии на сайте HuggingFace.

### Решение

1. Открой страницу модели в браузере под своим аккаунтом
2. Нажми **"Agree and access repository"** / **"Accept license"**
3. Убедись что в `models.yaml` стоит `gated: true`
4. Убедись что токен задан в `credentials.yaml`

```yaml
# models.yaml
- repo_id: meta-llama/Llama-3.1-8B-Instruct
  gated: true   # <-- обязательно
```

---

## 4. hf_transfer: ложные 404 и зависания

### Симптом

- Ошибка 404 на репозиторий, который точно существует
- Загрузка зависает без прогресса
- В логе: `[INFO] hf_transfer enabled (fast Rust-based downloads)`

### Причина

`hf_transfer` — Rust-библиотека для ускорения загрузок. Она обходит стандартный HTTP-стек Python и иногда:
- некорректно обрабатывает LFS-редиректы (GGUF-файлы почти всегда через LFS)
- не передаёт токен авторизации в нужном формате
- возвращает 404 вместо реальной ошибки сети

### Решение

Отключи `hf_transfer` в `.env`:

```bash
# .env
HF_HUB_ENABLE_HF_TRANSFER=0
```

После этого повтори загрузку:

```bash
make download
```

> Стандартная загрузка через Python медленнее (нет многопоточности),
> но надёжнее для большинства репозиториев.

Если скорость критична — включи обратно только после проверки, что модель скачивается без токена в браузере.

---

## 5. Создание токена HuggingFace с нужными правами

Перейди в [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **New token**.

### Классический токен (рекомендуется для простых случаев)

| Поле | Значение |
|------|---------|
| Token name | `aihub-read` |
| Type | **Read** |

Этого достаточно для:
- публичных моделей (без ограничений)
- gated-моделей (после принятия лицензии на сайте)

### Fine-grained токен (для production)

| Раздел | Разрешение |
|--------|-----------|
| **Repositories** | `Read access to contents of all public gated repos you can access` |
| Inference | не нужно |
| Webhooks | не нужно |
| User | не нужно |

### Запись токена в проект

```yaml
# credentials.yaml
huggingface:
  token: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> `credentials.yaml` добавлен в `.gitignore` — не попадёт в репозиторий.

---

## 6. Файл скачивается, но пустой или обрезанный

### Симптом

```
ValueError: Downloaded file is empty — possible network error
```

Или файл есть, но его размер значительно меньше ожидаемого.

### Решение

Принудительно перескачай файл:

```bash
make download -- --force
# или напрямую:
.venv/bin/python scripts/download_models.py --force
```

Флаг `--force` игнорирует ETag-кэш и скачивает заново.

---

## 7. Rate limit 429 — слишком много запросов

### Симптом

```
[WARN] Rate limit (429) — ждём 50s перед повтором ...
```

### Причина

HuggingFace ограничивает анонимные и low-tier запросы. Особенно при параллельных или частых загрузках.

### Решение

1. **Добавь токен** — авторизованные запросы имеют более высокий лимит
2. **Увеличь паузу между загрузками** в `models.yaml`:

```yaml
settings:
  inter_download_delay: 120   # 2 минуты между моделями
  retry_delay: 10.0           # базовая задержка перед повтором
```

---

## 8. Credentials not found — токен не читается

### Симптом

```
[INFO] No HF_TOKEN found — gated models will fail
```

### Причина

Файл `credentials.yaml` не найден или имеет неверную структуру.

### Решение

Создай файл по шаблону (не трогая `credentials.yaml.example` если он есть):

```bash
cp credentials.yaml.example credentials.yaml
```

Или создай вручную:

```yaml
# credentials.yaml
huggingface:
  token: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# s3:                      # раскомментируй если нужен S3
#   access_key_id: ...
#   secret_access_key: ...
```

Проверь что файл находится в корне проекта (рядом с `Makefile`).

Альтернатива — переменная окружения:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
make download
```
