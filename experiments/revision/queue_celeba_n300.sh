#!/bin/bash
# CelebA n=300, draws 1-4 (draw 0 exists): baselines first, hierarchical GenSPR last.
# Waits for the 64-mask run so the two do not share the machine.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
REV=experiments/revision

while pgrep -f "run_genspr_hier_rev.py .*--reps 64" > /dev/null; do sleep 60; done

$PY $REV/run_baselines_rev.py --method vae --dataset celeba --sizes 300 --draws 1 2 3 4
$PY $REV/run_fastgan_rev.py --mode mixed --dataset celeba --sizes 300 --draws 1 2 3 4
$PY $REV/run_mae_rev.py --keep 0.15 --sizes 300 --draws 1 2 3 4
$PY $REV/run_mae_rev.py --keep 0.85 --sizes 300 --draws 1 2 3 4
$PY $REV/run_baselines_rev.py --method dcgan --dataset celeba --sizes 300 --draws 1 2 3 4
$PY $REV/run_genspr_hier_rev.py --sizes 300 --draws 1 2 3 4
echo "CELEBA N300 QUEUE DONE"
