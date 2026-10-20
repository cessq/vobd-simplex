#!/usr/bin/env bash
# Все проверки одной командой: стиль, тесты, пересчёт экспериментов.
set -euo pipefail
cd "$(dirname "$0")"
ruff check .
ruff format --check .
python3 -m pytest -q tests
python3 experiments/compare.py
git status --short experiments/results
