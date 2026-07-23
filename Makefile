PYTHON ?= python3
MPLCONFIGDIR ?= $(CURDIR)/.matplotlib-cache

.PHONY: all figures verify checksums

all: figures verify

figures:
	MPLCONFIGDIR="$(MPLCONFIGDIR)" $(PYTHON) scripts/make_all.py

verify:
	$(PYTHON) scripts/verify_release.py

checksums:
	$(PYTHON) scripts/verify_release.py --write-checksums
	$(PYTHON) scripts/verify_release.py
