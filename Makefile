export UV_CACHE_DIR := $(HOME)/sgoinfre/pecoelho/.cache/uv
export HF_HOME := &(HOME)/sgoinfre/pecoelho/.cache/hugginface

RUN_FLAG1 = --functions_definition data/input/functions_definition.json
RUN_FLAG2 = --input data/input/function_calling_tests.json
RUN_FLAG3 = --output data/output/function_calls.json
LFLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

all: run

install:
	uv sync

run:
	uv run python3 -m src $(RUN_FLAG1) $(RUN_FLAG2) $(RUN_FLAG3)

debug:
	uv run python3 -m pdb src

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