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
11. [Использование скачанных Ollama-моделей](#11-использование-скачанных-ollama-моделей)

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

### Шаг 4 — создать список моделей

```bash
cp models.yaml.example models.yaml
```

Открыть `models.yaml` и при необходимости отредактировать список моделей или оставить примеры из шаблона.
Файл gitignored — персональные настройки не попадут в репозиторий.

### Шаг 5 — проверить конфигурацию

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

Скрипт проверяет доступность репозиториев (HuggingFace: запрос `repo_info`; Ollama: запрос манифеста) без скачивания файлов. Выводит те же статусы, что и при реальной загрузке.

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

Интерфейс содержит три вкладки: **Мои модели**, **Поиск HuggingFace** и **Поиск Ollama**.

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
2. Доступные поля: теги, описание, `dest_dir`, gated
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
   - `dest_dir`, теги, описание — предзаполняются автоматически из результатов поиска
   - `dest_dir` определяется по типу задачи и названию модели (напр. `llm/qwen`, `embeddings`)
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
3. Проверить/исправить строки: `dest_dir: llm/llama`, теги: `llm chat 8b`
4. Нажать Добавить в models.yaml → Переключиться на «Мои модели»
5. В терминале: `python scripts/download_models.py --model Llama`

---

### Вкладка «Поиск Ollama»

Позволяет искать модели на `ollama.com` и добавлять их в `models.yaml` в формате Ollama-записи.

**Пред-фильтры (применяются до нажатия «Искать»):**

Перед поиском можно сразу задать фильтры, которые скроют ненужные варианты из результатов:

| Группа фильтров | Варианты |
|-----------------|---------|
| **Квантизация** | Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, IQ4_NL … |
| **Размер файла** | ≤2 ГБ, ≤4 ГБ, ≤8 ГБ, ≤16 ГБ, ≤32 ГБ |
| **Контекст (мин.)** | 8K+, 16K+, 32K+, 64K+, 128K+, 200K+ |
| **Тип входных данных** | text, text+img, text+audio |

**Поиск:**

Ввести название модели (напр. `qwen3`, `llama3.2`) → нажать **Искать**. Отображаются: число загрузок (pulls), теги возможностей, число вариантов, описание.

**Возможности модели (теги):**

| Тег | Значение |
|-----|---------|
| `tools` | Поддержка function calling / tool use |
| `thinking` | Режим расширенного рассуждения (chain-of-thought) |
| `vision` | Анализ изображений |
| `embedding` | Генерация векторных эмбеддингов |
| `cloud` | Облачная / API-модель (не локальная) |

Кнопки-фильтры над таблицей фильтруют результаты по выбранным возможностям (с счётчиком по каждому).

**Плоский список вариантов:**

После нажатия «Искать» каждая найденная модель раскрывается автоматически: вместо кнопки «▾ выбрать вариант» все теги отображаются в виде отдельных строк с колонками:

| Колонка | Содержимое |
|---------|-----------|
| Вариант | Тег (напр. `7b-instruct-q4_K_M`) |
| Размер | Размер GGUF-файла (загружается мгновенно из страницы тегов ollama.com) |
| Контекст | Максимальная длина контекста (из описания модели) |
| Input | Типы входных данных: `text`, `text+img`, `text+audio` |
| Загрузки | Число загрузок варианта |
| Теги | Дополнительные теги |

Пред-фильтры скрывают строки, не соответствующие выбранным критериям.

**Процесс добавления Ollama-модели:**

1. Задать пред-фильтры (квантизация, размер, контекст, input)
2. Ввести название → нажать **Искать**
3. В раскрытом списке вариантов отметить нужные чекбоксами
4. Нажать **Добавить выбранные**
5. Заполнить форму: `dest_dir`, теги, описание → **Добавить в models.yaml**

Ollama-модели загружаются из OCI-реестра `registry.ollama.ai` в формате GGUF с поддержкой HTTP Resume (прерванные загрузки продолжаются с точки останова).

---

## 4. Поиск новых моделей

Поиск новых моделей доступен через **веб-интерфейс** (`make ui`) во вкладках:

- **Поиск HuggingFace** — поиск по `huggingface.co` (см. [Вкладка «Поиск HuggingFace»](#вкладка-поиск-huggingface) в разделе 3)
- **Поиск Ollama** — поиск по `ollama.com` с выбором варианта квантизации (см. [Вкладка «Поиск Ollama»](#вкладка-поиск-ollama) в разделе 3)

### Полный цикл: найти → добавить → загрузить

**HuggingFace:**

```bash
make ui
# 1. вкладка «Поиск HuggingFace»: задать параметры → Искать → выбрать файлы
# 2. Добавить выбранные → заполнить форму → Добавить в models.yaml
# 3. вкладка «Мои модели»: убедиться что модели появились
python scripts/download_models.py
```

**Ollama:**

```bash
make ui
# 1. вкладка «Поиск Ollama»: выбрать пред-фильтры (квантизация, размер, контекст, input)
# 2. ввести название → Искать (варианты раскрываются автоматически)
# 3. отметить нужные варианты чекбоксами → Добавить выбранные → заполнить форму → Добавить в models.yaml
python scripts/download_models.py
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
cp models.yaml.example models.yaml
# заполнить/дополнить models.yaml

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
# Вкладка «Поиск Ollama»: найти модель → выбрать вариант → добавить в models.yaml
python scripts/download_models.py   # загрузить новые включённые
```

### Резервное копирование в S3

```bash
# скачать и залить всё в S3
python scripts/download_models.py --upload-s3

# плановое обновление с синхронизацией
python scripts/download_models.py --upload-s3 --delay 5
```

### Найти и добавить модель через веб-интерфейс

```bash
make ui
# HuggingFace: вкладка «Поиск HuggingFace» → задать параметры → Искать
#              → выбрать файлы → Добавить выбранные → заполнить форму
# Ollama:      вкладка «Поиск Ollama» → ввести название → выбрать вариант
#              → Добавить выбранные → заполнить форму
python scripts/download_models.py   # загрузить добавленные модели
```

### Добавить Ollama-модель с конкретной квантизацией

```bash
make ui
# вкладка «Поиск Ollama» → выбрать пред-фильтр квантизации «Q4_K_M»
# ввести «qwen3» → Искать (варианты раскрываются автоматически)
# выбрать строку «8b-instruct-q4_K_M» → отметить чекбоксом → Добавить выбранные
# dest_dir: llm/qwen, теги: llm thinking tools 8b → Добавить в models.yaml
python scripts/download_models.py --model qwen3
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
[OK]   models.yaml в .gitignore
[OK]   .env в .gitignore
[OK]   .download.lock в .gitignore

=== Поиск credentials в коде (ложных позитивов быть не должно) ===
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

---

## 11. Использование скачанных Ollama-моделей

aihub скачивает GGUF-файлы Ollama из OCI-реестра напрямую и сохраняет их на диск.
Чтобы использовать эти файлы в `ollama`, нужно зарегистрировать их через `ollama create`.

### Проблема дублирования данных

По умолчанию `ollama create FROM /path/file.gguf` **копирует** GGUF в
`$OLLAMA_MODELS/blobs/sha256-{hash}` (~46 ГБ → ~92 ГБ на диске).

aihub решает эту проблему двумя способами.

---

### Вариант 1 — простая регистрация через `ollama create` (с копированием)

Используйте этот вариант, если файлы aihub и Ollama хранятся на **разных дисках**
или когда место не ограничено. После регистрации можно удалить оригиналы aihub.

```bash
# Зарегистрировать все GGUF из папки models/misc в Ollama
for f in /path/to/models/misc/*.gguf; do
    name=$(basename "$f" .gguf)
    echo "FROM $f" | ollama create "$name" -f -
done

# Проверить, что модели появились
ollama list

# После проверки — удалить оригиналы (блобы уже в ollama/blobs/)
# rm /path/to/models/misc/*.gguf
```

> **Примечание:** `ollama_model` в `models.yaml` имеет вид `model:tag` (например, `qwen3:7b`).
> Для создания имени в Ollama используйте именно это значение как `$name`.

---

### Вариант 2 — zero-copy через `--register-ollama` (без дублирования)

aihub автоматически создаёт **hardlink** (жёсткую ссылку) от скачанного GGUF
в `$OLLAMA_MODELS/blobs/sha256-{hash}` перед вызовом `ollama create`.
Ollama находит уже существующий блоб → только записывает манифест → **данные не копируются**.

```bash
# Скачать Ollama-модели и сразу зарегистрировать в Ollama без дублирования
python scripts/download_models.py --register-ollama

# Если $OLLAMA_MODELS не ~/.ollama — указать явно
python scripts/download_models.py --register-ollama --ollama-models-dir /data/ollama

# Зарегистрировать уже скачанные модели (без повторного скачивания)
python scripts/download_models.py --register-ollama
# (download пропустит уже скачанные, но регистрация выполнится для SKIP-статусов тоже)
```

**Как это работает:**

1. Ollama хранит файлы по SHA256: `$OLLAMA_MODELS/blobs/sha256-{hexhash}`.
2. aihub сохраняет SHA256 в `.etag`-сайдкаре рядом с GGUF при скачивании.
3. `--register-ollama` читает `.etag` → создаёт hardlink с правильным именем → запускает
   `ollama create model:tag -f Modelfile`.
4. `ollama create` находит блоб → пишет только манифест → нет копирования.

**Требования:**
- `ollama` должен быть установлен и доступен в `PATH` (`ollama --version`).
- Hardlink работает только если GGUF и `$OLLAMA_MODELS` на **одном разделе/диске**.
  При разных файловых системах автоматически создаётся симлинк
  (удаление GGUF сломает симлинк — держите файлы, пока используете модель в Ollama).

**Проверка результата:**

```bash
ollama list                      # модель появится в списке
ollama run qwen3:7b              # запустить и проверить
du -sh ~/.ollama/blobs/          # размер не вырастет — hardlink не занимает места
```
