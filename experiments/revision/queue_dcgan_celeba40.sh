#!/bin/bash
# DCGAN on CelebA n=40, draws 3-4 (draws 0-2 exist). Runs after the other revision queues.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
REV=experiments/revision

while pgrep -f "queue_celeba_n300.sh|queue_fastgan_parity.sh|run_genspr_hier_rev.py .*--reps 64" > /dev/null; do sleep 60; done

$PY $REV/run_baselines_rev.py --method dcgan --dataset celeba --sizes 40 --draws 3 4
echo "DCGAN CELEBA40 QUEUE DONE"
