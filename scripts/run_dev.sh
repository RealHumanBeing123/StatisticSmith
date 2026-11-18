#!/usr/bin/env bash
# Run from source on Linux/macOS:  scripts/run_dev.sh [datafile]
set -e
cd "$(dirname "$0")/.."
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt; }
.venv/bin/python main.py "$@"
