#!/bin/bash
# Item 8: mixed FastGAN. Small sizes (five draws) first, then large sizes (one draw).
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
R=experiments/revision/run_fastgan_rev.py
for ds in mnist fmnist; do
  $PY $R --mode mixed --dataset $ds --sizes 30 300 --draws 0 1 2 3 4
done
$PY $R --mode mixed --dataset mnist --sizes 1000 5000 10000 20000 --draws 0
$PY $R --mode mixed --dataset fmnist --sizes 1000 5000 10000 --draws 0
echo "QUEUE DONE"
