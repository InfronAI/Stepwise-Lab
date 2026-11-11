PYTHON ?= python3.11
VENV := .venv/bin

.PHONY: setup doctor demo test lint replay video
setup:
	$(PYTHON) -m venv .venv
	$(VENV)/python -m pip install -e '.[train,dev]'
doctor:
	$(VENV)/prm800k doctor
demo:
	$(VENV)/prm800k demo --output experiments/runs/demo
test:
	$(VENV)/pytest -q
lint:
	$(VENV)/ruff check prm800k scripts tests
replay:
	$(VENV)/prm800k index --output .cache/scored.sqlite
	$(VENV)/prm800k evaluate --database .cache/scored.sqlite --population 1860 --ns 1,10,25,50,100,200,500,1000,1860 --method prm,orm --trials 400 --output experiments/runs/official
video:
	$(VENV)/python scripts/record_demo.py
