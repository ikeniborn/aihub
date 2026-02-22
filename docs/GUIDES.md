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
9. [Контроль нагрузки на канал](#9-контроль-нагрузки-на-канал)
10. [Безопасность](#10-безопасность)

---

## 1. Первоначальная настройка

### Шаг 1 — установить зависимости

```bash
make setup
```

Команда создаёт изолированное виртуальное окружение `.venv` и устанавливает все зависимости автоматически. Повторный запуск `make setup` безопасен.

> Альтернатива без make: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`

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
make ui                        # http://localhost:9000, открывает браузер автоматически
make ui PORT=9001              # другой порт
make ui HOST=0.0.0.0           # доступ из локальной сети
make ui CONFIG=custom.yaml     # другой конфиг
```

Или напрямую:

```bash
python scripts/model_browser.py --open
python scripts/model_browser.py --host 0.0.0.0 --port 9001 --open
```

Интерфейс содержит две вкладки: **Мои модели** и **Поиск HuggingFace**.

---

### Вкладка «Мои модели»

**Иерархия по `dest_dir`:**

Модели автоматически группируются по полю `dest_dir`:

```
▼ llm  (12)
    ▼ qwen  (3)
        ○  bartowski/Qwen2.5-14B-Instruct-GGUF  …
    ▼ deepseek  (2)
        ○  …
▼ embeddings  (2)
    ○  …
```

- Клик по заголовку группы — свернуть / развернуть
- Счётчик в заголовке показывает общее кол-во моделей в группе
- Структура автоматически обновляется после изменения `dest_dir`

**Просмотр:**
- Зелёная точка — файл скачан, серая — не скачан
- Тусклая строка — модель отключена (`enabled: false`)
- Реальный размер файла на диске
- Клик по `repo_id` открывает страницу модели на HuggingFace

**Фильтрация:**
- Поле поиска — substring по `repo_id`, `filename`, `description`
- Кнопки-теги — фильтр по категориям (поддерживается комбинирование)
- Переключатель «только скачанные»
- При активном фильтре пустые группы скрываются автоматически

**Включение / отключение:**
1. Поставить или снять чекбокс `enabled` у нужных моделей
2. Нажать **Сохранить** — `models.yaml` обновится атомарно

**Инлайн-редактирование атрибутов (кнопка ✎):**
1. Нажать ✎ в строке модели — откроется форма редактирования
2. Доступные поля: теги, VRAM (GB), описание, `dest_dir`, gated
3. Нажать **Сохранить** — изменения записываются в `models.yaml`
4. Если изменился `dest_dir` и файл скачан — файл автоматически перемещается в новую директорию

> Поле `dest_dir` — путь относительно `models_dir`. Формат: `a-z 0-9 _ -`, разделитель `/`, от 1 до 3 уровней.
> Примеры: `misc`  `llm`  `llm/qwen`  `llm/qwen/chat`

**Удаление модели (кнопка ✕):**
- Удаляет запись из `models.yaml`
- Если файл скачан — удаляет и файл с диска (предварительный confirm)

---

### Вкладка «Поиск HuggingFace»

Позволяет искать модели прямо в интерфейсе и добавлять их в `models.yaml` без редактирования файла вручную.

**Поля поиска:**

| Поле | Что ищет | Пример |
|------|----------|--------|
| Запрос | Полнотекстовый поиск по названию | `llama`, `deepseek` |
| Автор | Фильтр по организации | `bartowski`, `Qwen` |
| Regex файлов | Фильтр по именам файлов в репо | `Q4_K_M\.gguf$` |
| Лимит | Макс. кол-во результатов (1–50) | `20` |

**Процесс добавления модели:**

1. Ввести параметры поиска → нажать **Искать**
2. В таблице результатов отметить нужные файлы чекбоксами
   - В таблице видны: pipeline_tag, лайки, описание, размер файла, теги
3. Нажать **Добавить выбранные**
4. Откроется таблица с отдельной строкой на каждую выбранную модель:
   - `dest_dir`, теги, VRAM, описание — предзаполняются автоматически из результатов поиска
   - `dest_dir` определяется по типу задачи и названию модели (напр. `llm/qwen`, `embeddings`)
   - VRAM оценивается из размера модели и типа квантизации
   - Все поля можно скорректировать перед добавлением
5. Нажать **Добавить в models.yaml**

> **Формат `dest_dir`:** только `a-z 0-9 _ -`, разделитель `/`, от 1 до 3 уровней.
> Примеры: `misc`  `llm`  `llm/qwen`  `llm/qwen/chat`
> Обратный слеш `\` и заглавные буквы — автоматически исправляются при вводе.

> Дубликаты пропускаются автоматически — при повторном добавлении уже существующей модели она будет пропущена, остальные добавятся.

После добавления список в вкладке «Мои модели» обновляется автоматически.

**Пример: найти все Q4_K_M модели от bartowski и добавить выбранные:**

1. Автор: `bartowski`, Regex файлов: `Q4_K_M\.gguf$` → Искать
2. Выбрать нужные файлы → Добавить выбранные
3. Проверить/исправить строки: `dest_dir: llm/llama`, теги: `llm chat 8b`, VRAM: `5`
4. Нажать Добавить в models.yaml → Переключиться на «Мои модели»
5. В терминале: `python scripts/download_models.py --model Llama`

---

## 4. Поиск новых моделей

Поиск доступен двумя способами: через **веб-интерфейс** (вкладка «Поиск HuggingFace», см. раздел 3) или через **командную строку** (`browse_models.py`).

### Фильтры поиска

| Аргумент | Переменная `.env` | Что фильтрует |
|---|---|---|
| `--query TEXT` | — | Полнотекстовый поиск по названию и описанию |
| `--author NAME` | — | Автор или организация |
| `--pipeline-tag TASK` | `BROWSE_PIPELINE_TAG` | Задача модели (text-generation, text-to-image, …) |
| `--library LIB` | `BROWSE_LIBRARY` | Формат / библиотека (gguf, safetensors, transformers, …) |
| `--language LANG` | `BROWSE_LANGUAGE` | Язык модели (ru, en, zh, …) |
| `--regex PATTERN` | — | Regex по `repo_id` (после API-запроса) |
| `--file-regex PATTERN` | `BROWSE_FILE_REGEX` | Regex по именам файлов внутри репозитория |
| `--tags TAG …` | — | Теги HuggingFace |
| `--limit N` | — | Макс. кол-во результатов (по умолчанию: 20) |
| `--sort` | — | Сортировка: `downloads` / `likes` / `lastModified` |

Переменные из `.env` задают дефолты — CLI-аргументы всегда имеют приоритет.

---

### Поиск по задаче модели (`--pipeline-tag`)

```bash
# все LLM для генерации текста
python scripts/browse_models.py --pipeline-tag text-generation --limit 30

# модели для работы с изображениями
python scripts/browse_models.py --pipeline-tag text-to-image --library safetensors

# модели распознавания речи
python scripts/browse_models.py --pipeline-tag automatic-speech-recognition --limit 20

# модели для эмбеддингов / семантического поиска
python scripts/browse_models.py --pipeline-tag sentence-similarity --file-regex "\.gguf$"
```

Популярные значения `pipeline_tag`:

| Значение | Задача |
|---|---|
| `text-generation` | LLM, чат, генерация текста |
| `text-to-image` | Генерация изображений (Stable Diffusion, FLUX) |
| `automatic-speech-recognition` | Распознавание речи (Whisper) |
| `text-to-speech` | Синтез речи |
| `sentence-similarity` | Эмбеддинги, семантический поиск |
| `text-classification` | Классификация текста |
| `translation` | Перевод |
| `image-classification` | Классификация изображений |

---

### Поиск по формату / библиотеке (`--library`)

```bash
# только GGUF-модели
python scripts/browse_models.py --query "llama" --library gguf

# safetensors (для ComfyUI, A1111)
python scripts/browse_models.py --pipeline-tag text-to-image --library safetensors

# ONNX-модели
python scripts/browse_models.py --query "whisper" --library onnx

# модели для transformers
python scripts/browse_models.py --author mistralai --library transformers
```

---

### Поиск по языку (`--language`)

```bash
# русскоязычные LLM в формате GGUF
python scripts/browse_models.py --pipeline-tag text-generation --library gguf --language ru

# китайские модели
python scripts/browse_models.py --pipeline-tag text-generation --language zh --limit 20

# многоязычные модели от конкретного автора
python scripts/browse_models.py --author Qwen --language ru
```

---

### Поиск по автору с фильтром размера

HuggingFace API не фильтрует по числу параметров напрямую — используйте `--regex`:

```bash
# только 14B-модели от bartowski
python scripts/browse_models.py --author bartowski --regex "14[Bb]"

# 7B и 8B модели
python scripts/browse_models.py --author unsloth --regex "[78][Bb]"

# модели 32B+
python scripts/browse_models.py --pipeline-tag text-generation --library gguf --regex "3[2-9][Bb]|[4-9][0-9][Bb]"
```

---

### Просмотр файлов репозитория

```bash
# показать все файлы
python scripts/browse_models.py --author bartowski --show-files

# только Q4_K_M GGUF
python scripts/browse_models.py --author bartowski --file-regex "Q4_K_M\.gguf$"

# только safetensors
python scripts/browse_models.py --pipeline-tag text-to-image --file-regex "\.safetensors$"
```

---

### Дефолты поиска через `.env`

Если вы всегда ищете GGUF-модели с Q4_K_M квантизацией, настройте дефолты один раз:

```dotenv
# .env
BROWSE_PIPELINE_TAG=text-generation
BROWSE_LIBRARY=gguf
BROWSE_FILE_REGEX=Q4_K_M\.gguf$
```

После этого достаточно указать только что искать:

```bash
# применяются дефолты из .env: pipeline_tag=text-generation, library=gguf, file_regex=Q4_K_M.gguf$
python scripts/browse_models.py --author bartowski
python scripts/browse_models.py --query "deepseek"
python scripts/browse_models.py --language ru
```

Переопределить дефолт для одного запроса:

```bash
# искать safetensors, игнорируя BROWSE_FILE_REGEX из .env
python scripts/browse_models.py --query "flux" --file-regex "\.safetensors$"
```

---

### Получить YAML для вставки в models.yaml

```bash
python scripts/browse_models.py \
  --author bartowski \
  --file-regex "Q4_K_M\.gguf$" \
  --yaml
```

Вывод скопировать в раздел `models:` файла `models.yaml`.
После вставки заполнить поля `dest_dir`, `tags`, `description` и поставить `enabled: true`.

---

### Полный цикл через CLI: найти → добавить → загрузить

```bash
# 1. найти модель
python scripts/browse_models.py \
  --pipeline-tag text-generation \
  --library gguf \
  --language ru \
  --file-regex "Q4_K_M\.gguf$" \
  --yaml

# 2. скопировать нужные записи в models.yaml, заполнить поля

# 3. проверить доступность
python scripts/download_models.py --dry-run --model saiga

# 4. загрузить
python scripts/download_models.py --model saiga
```

### Полный цикл через веб-интерфейс: найти → добавить → загрузить

```bash
# запустить браузер
make ui

# 1. вкладка «Поиск HuggingFace»:
#    Автор: bartowski, Regex файлов: Q4_K_M\.gguf$ → Искать
#    выбрать нужные файлы → Добавить выбранные
#    заполнить форму → Добавить в models.yaml

# 2. вкладка «Мои модели»:
#    убедиться что модели появились

# 3. в терминале — загрузить новые модели:
python scripts/download_models.py --model bartowski
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
    dest_dir: llm/phi        # 1–3 уровня: a-z 0-9 _ - через /
    enabled: true
    gated: false
    tags: [llm, reasoning, code, 14b]
    description: "Phi-4 14B — Microsoft, MIT"
```

> `dest_dir` определяет место в иерархии веб-интерфейса и путь на диске:
> `models/{dest_dir}/{filename}`. Допустимые форматы: `misc`, `llm/phi`, `llm/phi/instruct`.

4. Загрузить:

```bash
python scripts/download_models.py --model Phi-4
```

### Управлять набором через браузер

```bash
make ui   # открывает http://localhost:9000 в браузере автоматически
# Вкладка «Мои модели»: включить/отключить чекбоксы → Сохранить
# Вкладка «Поиск HuggingFace»: найти новые → выбрать → добавить в models.yaml
python scripts/download_models.py   # загрузить новые включённые
```

### Резервное копирование в S3

```bash
# скачать и залить всё в S3
python scripts/download_models.py --upload-s3

# плановое обновление с синхронизацией
python scripts/download_models.py --upload-s3 --delay 5
```

### Поиск и загрузка модели одной командой (CLI)

```bash
# поиск → YAML → добавить в models.yaml → загрузить
python scripts/browse_models.py \
  --author unsloth \
  --file-regex "Llama.*Q4_K_M\.gguf$" \
  --yaml >> models_candidates.yaml

# вручную перенести нужные записи из models_candidates.yaml в models.yaml
python scripts/download_models.py --model Llama
```

### Поиск всех русскоязычных LLM в GGUF

```bash
# все русские text-generation GGUF Q4_K_M
python scripts/browse_models.py \
  --pipeline-tag text-generation \
  --library gguf \
  --language ru \
  --file-regex "Q4_K_M\.gguf$" \
  --yaml
```

### Подобрать модель для image generation

```bash
# FLUX и SDXL в safetensors
python scripts/browse_models.py \
  --pipeline-tag text-to-image \
  --library safetensors \
  --limit 30

# только от конкретных авторов
python scripts/browse_models.py \
  --pipeline-tag text-to-image \
  --author black-forest-labs \
  --file-regex "\.safetensors$"
```

---

## 9. Контроль нагрузки на канал

HuggingFace использует **Xet-протокол** (Rust, content-addressed chunks): по умолчанию открывает до 49 параллельных TCP-соединений и может занять весь входящий канал (90+ Mbit/s). Для контроля используйте следующие механизмы.

### Ограничение параллельных соединений (главный рычаг)

```yaml
# models.yaml
settings:
  hf_download_concurrency: 4    # 4 вместо ~50 → снижение нагрузки в 10–12 раз
```

Или через CLI (для разового запуска):

```bash
python scripts/download_models.py --max-concurrency 4
python scripts/download_models.py --max-concurrency 1   # самый щадящий режим
```

### Таймаут на зависший download (защита от «призрака»)

Если загрузка зависла и не завершается, скрипт завершится через заданное время:

```yaml
# models.yaml
settings:
  download_timeout_hours: 2     # убить процесс через 2 часа без результата
```

```bash
python scripts/download_models.py --download-timeout 1.0   # таймаут 1 час
python scripts/download_models.py --download-timeout 0     # без таймаута
```

При срабатывании таймаута скрипт выводит `[FATAL]` и завершается с кодом `3` — без молчаливого зависания.

### PID-блокировка — один процесс за раз

Одновременно может работать только один `download_models.py`. При попытке запустить второй:

```
[ERROR] Another download_models.py is already running (PID 13951).
        If no process is running, delete: /path/to/.download.lock
```

Если процесс завис и блокировка не снялась:

```bash
rm .download.lock
```

### Ограничение на уровне ОС (для Xet и любых протоколов)

```bash
# установить wondershaper
sudo apt install wondershaper

# ограничить входящий канал до 20 Mbit/s (на интерфейсе enp1s0)
sudo wondershaper enp1s0 20480 20480

# снять ограничение
sudo wondershaper clear enp1s0
```

### Запуск загрузки в нерабочее время

```bash
# загружать каждую ночь в 2:00
# crontab -e
0 2 * * * cd /path/to/aihub && make download
```

---

## 10. Безопасность

### Первоначальная настройка прав

```bash
cp credentials.yaml.example credentials.yaml
chmod 600 credentials.yaml    # обязательно — только владелец может читать
```

### Проверка безопасности проекта

```bash
make security-check
```

Команда проверяет:
- Права `credentials.yaml` (должны быть `600` или `400`)
- Наличие секретных файлов в `.gitignore`
- Отсутствие токенов HuggingFace в коде скриптов

Пример успешного вывода:

```
[OK] credentials.yaml permissions: 600

=== Проверка .gitignore ===
[OK]   credentials.yaml в .gitignore
[OK]   .env в .gitignore
[OK]   .download.lock в .gitignore

=== Поиск credentials в коде ===
[OK]   HF tokens не найдены в коде

Security check complete.
```

### Прокси с аутентификацией

Пароли в `PROXY_URL` автоматически маскируются во всех логах:

```
# Что записывается в лог:
[INFO] Proxy enabled: http://user:***@proxy.corp.com:3128

# Исходное значение в credentials.yaml:
url: "http://user:secret@proxy.corp.com:3128"
```

### Path Traversal защита (WebUI)

Поле `dest_dir` проверяется на сервере — даже если клиентская валидация обойдена через `curl` или прямой API-запрос:
- Допустимые символы: `a-z`, `0-9`, `-`, `_`, разделитель `/`
- Максимум 3 уровня вложенности
- Разрешённый диапазон: только внутри `models_dir`

Попытка `../../../../etc/passwd` вернёт HTTP 400.
