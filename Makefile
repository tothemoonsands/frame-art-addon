.PHONY: ha-context setup test

ha-context:
	ha-context --sync-config-from-git

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip

$(PYTHON):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install requests pillow rapidfuzz "samsungtvws[async] @ git+https://github.com/NickWaterton/samsung-tv-ws-api.git@fe95ef1d784cd32f49bf9a07ec479576574eea07"

setup: $(PYTHON)

test: $(PYTHON)
	$(PYTHON) -m unittest discover -s tests -v
