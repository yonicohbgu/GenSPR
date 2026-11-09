#!/bin/bash
# DCGAN reruns after the [0, 1] scaling fix; starts once the CelebA queue is done.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
REV=experiments/revision
RES=experiments/results/revision_2026_11

while pgrep -f queue_celeba.sh >/dev/null; do sleep 60; done
$PY $REV/run_baselines_rev.py --method dcgan --dataset mnist --sizes 30 300 --draws 0 1 2
$PY $REV/run_baselines_rev.py --method dcgan --dataset fmnist --sizes 30 300 --draws 0 1 2
$PY $REV/run_baselines_rev.py --method dcgan --dataset octmnist --sizes 40 300 --draws 0 1 2
$PY $REV/run_baselines_rev.py --method dcgan --dataset mnist --sizes 1000 5000 10000 20000
$PY $REV/run_baselines_rev.py --method dcgan --dataset fmnist --sizes 1000 5000 10000
$PY $REV/run_baselines_rev.py --method dcgan --dataset octmnist --sizes 1000 5000
$PY $REV/classifier.py --score $RES/dcgan/{mnist,fmnist,octmnist}/n*/draw*
echo "DCGAN RERUN DONE"
