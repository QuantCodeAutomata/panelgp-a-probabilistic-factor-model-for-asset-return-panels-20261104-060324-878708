#!/usr/bin/env bash
# Runs the test suite and every experiment at the configuration described in
# results/RESULTS.md. Pass --full to any individual exp module for the
# paper's full configuration (see README.md).
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
python -m exp.exp5
python -m exp.exp6
