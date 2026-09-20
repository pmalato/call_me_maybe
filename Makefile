LFLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

install:
	uv sync

run:
	uv run call-me-maybe.py

debug:

clean:
	rm -fr .mypy_cache
	rm -fr src/.mypy_cache
	rm -fr src/__pycache__
	clear

lint:
	python3 -m flake8 .
	python3 -m mypy . $(LFLAGS)

lint-strict:
	python3 -m flake8 .
	python3 -m mypy --strict .

.PHONY: install run debug clean lint lint-strict