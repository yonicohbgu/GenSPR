#!/bin/bash
# CelebA n=300/1000/5000, draw 0: hierarchical GenSPR, then VAE, DCGAN, mixed FastGAN.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
REV=experiments/revision

$PY $REV/run_genspr_hier_rev.py --sizes 300 1000 5000 --draws 0
$PY $REV/run_baselines_rev.py --method vae --dataset celeba --sizes 300 1000 5000
$PY $REV/run_baselines_rev.py --method dcgan --dataset celeba --sizes 300 1000 5000
$PY $REV/run_fastgan_rev.py --mode mixed --dataset celeba --sizes 300 1000 5000
echo "CELEBA LARGE QUEUE DONE"
