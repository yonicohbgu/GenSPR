#!/bin/bash
# Mixed FastGAN on CelebA n=40 with the same 9,600-image replicated set as the other methods.
# Runs after the 64-mask run and the n=300 queue.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
REV=experiments/revision

while pgrep -f "queue_celeba_n300.sh|run_genspr_hier_rev.py .*--reps 64" > /dev/null; do sleep 60; done

$PY $REV/run_fastgan_rev.py --mode mixed --dataset celeba --sizes 40 --draws 0 1 2 3 4 --parity
echo "FASTGAN PARITY QUEUE DONE"
