SHELL := /bin/bash

VENV_PYTHON := .venv/bin/python
VENV_PIP    := .venv/bin/pip

.PHONY: setup download ui list update check-creds-perms security-check

## setup — создать .venv и установить зависимости
setup:
	python3 -m venv .venv
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@echo "Setup complete. Virtual environment ready at .venv/"

## download — загрузить все включённые модели
download:
	$(VENV_PYTHON) scripts/download_models.py

## list — показать список моделей
list:
	$(VENV_PYTHON) scripts/download_models.py --list

## ui — запустить веб-интерфейс и открыть браузер
##      PORT=9000  HOST=127.0.0.1  CONFIG=models.yaml
##      Пример: make ui PORT=9001
PORT   ?= 9000
HOST   ?= 127.0.0.1
CONFIG ?= models.yaml

ui:
	$(VENV_PYTHON) scripts/model_browser.py --port $(PORT) --host $(HOST) --config $(CONFIG)

## update — обновить все пакеты до последних версий
update:
	$(VENV_PIP) install --upgrade -r requirements.txt
	@echo "Packages updated."

## check-creds-perms — проверить права доступа к credentials.yaml (должны быть 600 или 400)
check-creds-perms:
	@if [ -f credentials.yaml ]; then \
	  PERMS=$$(stat -c '%a' credentials.yaml 2>/dev/null || stat -f '%A' credentials.yaml); \
	  if [ "$$PERMS" = "600" ] || [ "$$PERMS" = "400" ]; then \
	    echo "[OK] credentials.yaml permissions: $$PERMS"; \
	  else \
	    echo "[WARN] credentials.yaml permissions: $$PERMS — рекомендуется 600"; \
	    echo "       Исправить: chmod 600 credentials.yaml"; \
	    exit 1; \
	  fi \
	else \
	  echo "[SKIP] credentials.yaml не найден"; \
	fi

## security-check — комплексная проверка безопасности проекта
security-check: check-creds-perms
	@echo ""
	@echo "=== Проверка .gitignore ==="
	@for secret in credentials.yaml models.yaml .env .download.lock; do \
	  if grep -qF "$$secret" .gitignore 2>/dev/null; then \
	    echo "[OK]   $$secret в .gitignore"; \
	  else \
	    echo "[WARN] $$secret НЕ в .gitignore — риск утечки"; \
	  fi; \
	done
	@echo ""
	@echo "=== Поиск credentials в коде (ложных позитивов быть не должно) ==="
	@grep -rn "hf_[a-zA-Z0-9]\{10,\}" scripts/ 2>/dev/null && echo "[WARN] Возможный HF token в коде" || echo "[OK]   HF tokens не найдены в коде"
	@echo ""
	@echo "Security check complete."
