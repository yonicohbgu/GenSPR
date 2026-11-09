#!/bin/bash
# Item 7 on hierarchical CelebA, after the large-n queue:
# 7a component ablation at n=40 (seed 0); 7b plateau test at n=5000 (boost 0/2/7, annealed).
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
REV=experiments/revision

while pgrep -f queue_celeba_large.sh >/dev/null; do sleep 60; done
$PY $REV/run_genspr_hier_rev.py --sizes 5000 --draws 0 --variants noboost boost2 boost7 anneal
$PY $REV/run_genspr_hier_rev.py --sizes 40 --draws 0 --variants nohebb noboost dense
echo "SP ABLATION QUEUE DONE"
