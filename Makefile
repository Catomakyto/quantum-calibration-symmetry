.PHONY: reproduce quick test clean help

PYTHON ?= python3

help:
	@echo "make reproduce  Run the full pipeline + extras (~4 minutes)"
	@echo "make quick      Run a 30-second smoke version with looser CIs"
	@echo "make test       Run the five unit tests on the baselines"
	@echo "make clean      Remove regenerated outputs"

reproduce:
	$(PYTHON) scripts/reproduce.py --mode full
	$(PYTHON) scripts/extras.py
	$(PYTHON) scripts/multi_qubit_crosstalk.py

quick:
	$(PYTHON) scripts/reproduce.py --mode quick

test:
	$(PYTHON) -c "import sys; sys.path.insert(0, 'scripts'); from reproduce import unit_tests; unit_tests(verbose=True)"

clean:
	rm -f results/*.csv results/*.json
	rm -f figures/*.pdf figures/*.png
