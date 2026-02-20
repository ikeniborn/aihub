# Руководство пользователя aihub

## Содержание

1. [Первоначальная настройка](#1-первоначальная-настройка)
2. [Загрузка моделей](#2-загрузка-моделей)
3. [Управление списком через веб-интерфейс](#3-управление-списком-через-веб-интерфейс)
4. [Поиск новых моделей](#4-поиск-новых-моделей)
5. [Синхронизация в S3](#5-синхронизация-в-s3)
6. [Обновление моделей](#6-обновление-моделей)
7. [Работа в CI/CD](#7-работа-в-cicd)
8. [Типовые сценарии](#8-типовые-сценарии)

---

## 1. Первоначальная настройка

### Шаг 1 — установить зависимости

```bash
pip install -r requirements.txt
```

### Шаг 2 — создать файл секретов

```bash
cp credentials.yaml.example credentials.yaml
```

Открыть `credentials.yaml` и заполнить:

```yaml
huggingface:
  token: ""        # оставить пустым, если не нужны закрытые модели

s3:
  access_key_id: ""
  secret_access_key: ""
```

Токен HuggingFace нужен только для моделей с `gated: true` (Llama, Gemma и др.).
Получить токен: <https://huggingface.co/settings/tokens>

### Шаг 3 — создать файл параметров

```bash
cp .env.example .env
```

Минимальная конфигурация для локального использования без S3:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1   # ускоренные загрузки (рекомендуется)
```

### Шаг 4 — проверить конфигурацию

```bash
python scripts/download_models.py --list
```

Должна вывестись таблица моделей из `models.yaml`. Если вывод появился — настройка выполнена корректно.

---

## 2. Загрузка моделей

### Просмотр доступных моделей

```bash
python scripts/download_models.py --list
```

Колонки: `STATUS` (enabled/disabled), `REPO_ID / FILENAME`, `TAGS`, `VRAM`, `DESCRIPTION`.

### Проверка без загрузки

```bash
python scripts/download_models.py --dry-run
```

Скрипт проверяет доступность репозиториев на HuggingFace без скачивания файлов. Выводит те же статусы, что и при реальной загрузке.

### Загрузить все включённые модели

```bash
python scripts/download_models.py
```

Модели сохраняются в `./models/{dest_dir}/{filename}`.
Уже скачанные файлы с актуальным ETag **пропускаются автоматически**.

### Загрузить конкретную модель

```bash
# по подстроке в repo_id или filename
python scripts/download_models.py --model Phi-4
python scripts/download_models.py --model Qwen2.5-14B
python scripts/download_models.py --model bge-m3
```

### Загрузить модели по категории

```bash
python scripts/download_models.py --tag russian       # русскоязычные
python scripts/download_models.py --tag embeddings    # эмбеддинги для RAG
python scripts/download_models.py --tag reasoning     # reasoning-модели
python scripts/download_models.py --tag code          # модели для кода
```

### Включить отключённые модели

Модели с `enabled: false` в `models.yaml` пропускаются по умолчанию.
Чтобы загрузить конкретную отключённую модель:

```bash
python scripts/download_models.py --include-disabled --model Qwen2.5-32B
```

### Принудительная перезагрузка

```bash
# перезагрузить все модели, игнорируя ETag
python scripts/download_models.py --force

# перезагрузить одну модель
python scripts/download_models.py --force --model DeepSeek-R1
```

---

## 3. Управление списком через веб-интерфейс

### Запуск

```bash
python scripts/model_browser.py
```

Открыть в браузере: <http://localhost:9000>

Для доступа с другого хоста в локальной сети:

```bash
python scripts/model_browser.py --host 0.0.0.0 --port 9000
```

### Что можно сделать в интерфейсе

**Просмотр:**
- Таблица всех моделей из `models.yaml`
- Зелёная точка — файл скачан, серая — не скачан
- Тусклая строка — модель отключена (`enabled: false`)
- Реальный размер файла на диске

**Фильтрация:**
- Поле поиска — substring по `repo_id`, `filename`, `description`
- Кнопки-теги — фильтр по категориям
- Переключатель «только скачанные»

**Редактирование:**
1. Поставить или снять чекбокс `enabled` у нужных моделей
2. Нажать **Сохранить** — `models.yaml` обновится атомарно

### Добавить другой конфиг

```bash
python scripts/model_browser.py --config /path/to/custom.yaml
```

---

## 4. Поиск новых моделей

### Поиск по ключевому слову

```bash
python scripts/browse_models.py --query "llama"
python scripts/browse_models.py --query "embedding russian"
```

### Поиск по автору

```bash
python scripts/browse_models.py --author bartowski
python scripts/browse_models.py --author unsloth
python scripts/browse_models.py --author Qwen
```

### Поиск с regex-фильтром по имени модели

```bash
# только модели с 14B в названии
python scripts/browse_models.py --author bartowski --regex ".*14B.*"

# только Q4_K_M квантизации
python scripts/browse_models.py --query "instruct" --regex ".*Q4_K_M.*"
```

### Просмотр файлов в репозитории

```bash
# показать все файлы репозиториев
python scripts/browse_models.py --author bartowski --show-files

# только gguf-файлы с regex-фильтром
python scripts/browse_models.py --author bartowski --file-regex "Q4_K_M\.gguf$"
```

### Получить YAML для вставки в models.yaml

```bash
# найти и сформировать записи
python scripts/browse_models.py \
  --author bartowski \
  --file-regex "Q4_K_M\.gguf$" \
  --yaml
```

Вывод скопировать в раздел `models:` файла `models.yaml`.
После вставки заполнить поля `dest_dir`, `tags`, `vram_gb`, `description` и поставить `enabled: true`.

### Пример полного цикла: найти → добавить → загрузить

```bash
# 1. найти модель
python scripts/browse_models.py --query "mistral" --file-regex "Q4_K_M\.gguf$" --yaml

# 2. скопировать нужную запись в models.yaml, заполнить поля

# 3. проверить
python scripts/download_models.py --dry-run --model mistral

# 4. загрузить
python scripts/download_models.py --model mistral
```

---

## 5. Синхронизация в S3

### Предварительная настройка

В `credentials.yaml` заполнить секции `s3`:

```yaml
s3:
  access_key_id: "AKIAXXXXXXXXXXXXXXXX"
  secret_access_key: "ваш_секрет"
```

В `.env` указать параметры:

```bash
S3_BUCKET=my-ai-models
S3_REGION=us-east-1
S3_PREFIX=models

# для Yandex Object Storage:
S3_ENDPOINT_URL=https://storage.yandexcloud.net
S3_REGION=ru-central1

# для MinIO:
S3_ENDPOINT_URL=http://localhost:9000
S3_REGION=us-east-1
```

### Загрузить локально и отправить в S3

```bash
python scripts/download_models.py --upload-s3
```

Модели сохраняются на диск **и** копируются в S3.
Повторный запуск пропускает объекты, уже присутствующие в S3 с совпадающим размером.

### Только в S3 (без постоянного хранения на диске)

```bash
python scripts/download_models.py --s3-only
```

Файлы скачиваются во временную директорию, загружаются в S3, затем удаляются.
Временная директория удаляется автоматически даже при прерывании (`Ctrl+C`).

### Выборочная синхронизация в S3

```bash
# только русскоязычные модели
python scripts/download_models.py --upload-s3 --tag russian

# принудительно перезалить embeddings
python scripts/download_models.py --upload-s3 --tag embeddings --force
```

### Структура ключей в S3

```
{S3_PREFIX}/{dest_dir}/{filename}

Примеры:
  models/llm/qwen/Qwen2.5-14B-Instruct-Q4_K_M.gguf
  models/embeddings/bge-m3-Q8_0.gguf
  models/llm/russian/model-q4_k.gguf
```

---

## 6. Обновление моделей

### Политики обновления

В `models.yaml`:

```yaml
settings:
  update_policy: etag   # рекомендуется
```

| Политика | Поведение |
|----------|-----------|
| `etag`   | При каждом запуске сравнивает ETag через HTTP HEAD. Перекачивает только если файл изменился на сервере |
| `skip`   | Никогда не перекачивает, если файл есть. Подходит если обновления не нужны |
| `always` | Всегда перекачивает. Эквивалентно флагу `--force` |

### Проверить и обновить все модели

```bash
# etag-политика (default) — скачает только изменившиеся
python scripts/download_models.py

# принудительно обновить всё
python scripts/download_models.py --force
```

### Настройка задержек

```bash
# пауза 2 секунды между моделями, 5 повторов, 10 секунд базовая задержка
python scripts/download_models.py --delay 2 --retries 5 --retry-delay 10
```

Или в `models.yaml`:

```yaml
settings:
  retry_count: 3
  retry_delay: 5.0
  inter_download_delay: 30   # секунд между моделями
```

При HTTP 429 (rate limit) скрипт автоматически ждёт `retry_delay × 10` секунд перед повтором.

---

## 7. Работа в CI/CD

В CI/CD окружении файл `credentials.yaml` обычно недоступен — использовать переменные окружения.

### Переменные окружения для CI

```bash
# секреты
HF_TOKEN=hf_...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# параметры
S3_BUCKET=my-ai-models
S3_REGION=us-east-1
S3_PREFIX=models
S3_ENDPOINT_URL=
HF_HUB_ENABLE_HF_TRANSFER=1
```

### Пример pipeline (GitHub Actions / GitLab CI)

```bash
# установка
pip install -r requirements.txt

# только S3 — не занимать место на раннере
python scripts/download_models.py --s3-only

# или: загрузить артефакты для следующего шага
python scripts/download_models.py --tag embeddings
```

### Использовать альтернативный файл кредов

```bash
python scripts/download_models.py --creds /secrets/prod-credentials.yaml
```

---

## 8. Типовые сценарии

### Первая загрузка всего набора

```bash
pip install -r requirements.txt
cp credentials.yaml.example credentials.yaml
# заполнить credentials.yaml
cp .env.example .env
# заполнить .env (HF_HUB_ENABLE_HF_TRANSFER=1)

python scripts/download_models.py --dry-run   # убедиться что всё ок
python scripts/download_models.py             # загрузить
```

### Добавить новую модель вручную

1. Найти модель на <https://huggingface.co>
2. Скопировать `repo_id` (например, `bartowski/Phi-4-GGUF`) и точное имя файла
3. Добавить запись в `models.yaml`:

```yaml
  - repo_id: bartowski/Phi-4-GGUF
    filename: Phi-4-Q4_K_M.gguf
    dest_dir: llm/phi
    enabled: true
    gated: false
    tags: [llm, reasoning, code, 14b]
    vram_gb: 8
    description: "Phi-4 14B — Microsoft, MIT"
```

4. Загрузить:

```bash
python scripts/download_models.py --model Phi-4
```

### Управлять набором через браузер

```bash
python scripts/model_browser.py   # открыть http://localhost:9000
# включить/отключить модели через чекбоксы → Сохранить
python scripts/download_models.py   # загрузить новые включённые
```

### Резервное копирование в S3

```bash
# скачать и залить всё в S3
python scripts/download_models.py --upload-s3

# плановое обновление с синхронизацией
python scripts/download_models.py --upload-s3 --delay 5
```

### Поиск и загрузка модели одной командой

```bash
# поиск → YAML → добавить в models.yaml → загрузить
python scripts/browse_models.py \
  --author unsloth \
  --file-regex "Llama.*Q4_K_M\.gguf$" \
  --yaml >> models_candidates.yaml

# вручную перенести нужные записи из models_candidates.yaml в models.yaml
python scripts/download_models.py --model Llama
```
