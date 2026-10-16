#!/usr/bin/env bash
# Eval slice: every title keyword-searchable first, then dense vectors by priority.
set -e
PY=venv/Scripts/python.exe
$PY -m regrag.ingest.pipeline --titles 1,21,29,40,49 --sparse-only
for t in 49 29 21 40; do $PY -m regrag.ingest.pipeline --titles $t; done
echo INGEST_DONE
