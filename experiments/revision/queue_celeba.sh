#!/bin/bash
# CelebA n=40 (curated): hierarchical GenSPR draw 0, then baselines, then GenSPR draws 1-4.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
REV=experiments/revision

$PY $REV/run_genspr_hier_rev.py --sizes 40 --draws 0
$PY $REV/run_baselines_rev.py --method vae --dataset celeba --sizes 40 --draws 0 1 2 3 4
$PY $REV/run_fastgan_rev.py --mode mixed --dataset celeba --sizes 40 --draws 0 1 2 3 4
$PY $REV/run_baselines_rev.py --method dcgan --dataset celeba --sizes 40 --draws 0 1 2
$PY $REV/run_genspr_hier_rev.py --sizes 40 --draws 1 2 3 4
echo "CELEBA QUEUE DONE"
