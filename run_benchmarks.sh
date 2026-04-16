#!/usr/bin/env bash
set -euo pipefail

EPOCHS=${1:-100}   # override with:  ./run_benchmarks.sh 200

echo "Running all benchmarks (epochs=${EPOCHS})"
echo

python main.py --scenario bsc            --epochs "$EPOCHS"
python main.py --scenario binary_feedback --epochs "$EPOCHS"
python main.py --scenario gaussian_ar1 --mode channel_ar --epochs "$EPOCHS"
python main.py --scenario gaussian_ar1 --mode source_ar  --epochs "$EPOCHS"

python main.py --scenario gaussian_ar1 --architecture lstm --epochs "$EPOCHS"

python main.py --estimator mine --scenario gaussian_ar1 --mode channel_ar --epochs "$EPOCHS"
python main.py --estimator mine --scenario gaussian_ar1 --mode source_ar  --epochs "$EPOCHS"