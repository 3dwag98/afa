.PHONY: install test run clean

install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/pytest tests/

run:
	.venv/bin/python main.py

clean:
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf data/*.db
	rm -rf output/*.xlsx
	rm -rf logs/*.log
