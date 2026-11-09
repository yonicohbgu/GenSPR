#!/bin/bash
# Table 5 CelebA encoder replacements (random projection, MLP) at GenSPR's keep ratio.
# Runs after the other revision queues.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
REV=experiments/revision

while pgrep -f "queue_celeba_n300.sh|queue_fastgan_parity.sh|queue_dcgan_celeba40.sh|run_genspr_hier_rev.py .*--reps 64" > /dev/null; do sleep 60; done

for e in random_projection linear; do
  $PY $REV/run_mae_rev.py --encoder $e --keep 0.15 --sizes 40 --draws 0 1 2 3 4
  $PY $REV/run_mae_rev.py --encoder $e --keep 0.15 --sizes 5000
done
echo "ENCODER ABLATION QUEUE DONE"
