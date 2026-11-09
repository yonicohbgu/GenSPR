#!/bin/bash
# Reruns the runs that crashed on Nov 8 when clean-fid's Inception weights vanished from /tmp
# (64-mask CelebA n=40 and VAE CelebA n=300). Completed runs are skipped.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
REV=experiments/revision

while pgrep -f "queue_celeba_n300.sh|queue_fastgan_parity.sh|queue_dcgan_celeba40.sh|queue_encoder_ablation.sh" > /dev/null; do sleep 60; done

$PY $REV/run_baselines_rev.py --method vae --dataset celeba --sizes 300 --draws 1 2 3 4
$PY $REV/run_genspr_hier_rev.py --sizes 40 --draws 0 1 2 3 4 --reps 64
echo "RECOVER QUEUE DONE"
