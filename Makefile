SHELL := /bin/bash

VENV_PYTHON := .venv/bin/python
VENV_PIP    := .venv/bin/pip

.PHONY: setup download browse ui list update

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

## browse — поиск моделей на HuggingFace Hub
browse:
	$(VENV_PYTHON) scripts/browse_models.py

## ui — запустить веб-интерфейс (http://localhost:9000)
ui:
	$(VENV_PYTHON) scripts/model_browser.py

## update — обновить все пакеты до последних версий
update:
	$(VENV_PIP) install --upgrade -r requirements.txt
	@echo "Packages updated."
