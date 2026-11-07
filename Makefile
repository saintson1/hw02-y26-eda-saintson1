.PHONY: install run analyze check

PYTHON ?= python3
VENV ?= .venv

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install -r solution/requirements.txt

run:
	$(VENV)/bin/python -m streamlit run solution/app.py

analyze:
	$(VENV)/bin/python solution/analyze.py

check:
	$(VENV)/bin/python solution/checks.py
