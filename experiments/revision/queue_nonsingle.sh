#!/bin/bash
# Remaining non-single-SP runs: MAE-FS on CelebA (both mask settings), then the
# second seed of the hierarchical CelebA SP ablation.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
REV=experiments/revision

for k in 0.85 0.15; do
  $PY $REV/run_mae_rev.py --keep $k --sizes 40 --draws 0 1 2 3 4
  $PY $REV/run_mae_rev.py --keep $k --sizes 300 1000 5000
done
$PY $REV/run_genspr_hier_rev.py --sizes 40 --draws 1 --variants nohebb noboost dense
echo "NONSINGLE QUEUE DONE"
