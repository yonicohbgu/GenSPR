#!/bin/bash
# Runs after the mixed-FastGAN queue: coverage backfill, VAE, OCTMNIST FastGAN, DCGAN.
# Small sizes get five draws (DCGAN three), large sizes one. Finished runs are skipped on restart.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
REV=experiments/revision
RES=experiments/results/revision_2026_11

while pgrep -f queue_fastgan_mixed.sh >/dev/null; do sleep 60; done

$PY $REV/recompute_coverage.py $RES/fastgan_mixed/*/n*/draw*

for method in vae dcgan; do
  if [ $method = vae ]; then DRAWS="0 1 2 3 4"; else DRAWS="0 1 2"; fi
  $PY $REV/run_baselines_rev.py --method $method --dataset mnist    --sizes 30 300 --draws $DRAWS
  $PY $REV/run_baselines_rev.py --method $method --dataset fmnist   --sizes 30 300 --draws $DRAWS
  $PY $REV/run_baselines_rev.py --method $method --dataset octmnist --sizes 40 300 --draws $DRAWS
  if [ $method = vae ]; then
    $PY $REV/run_fastgan_rev.py --mode mixed --dataset octmnist --sizes 40 300 --draws 0 1 2 3 4
    $PY $REV/run_fastgan_rev.py --mode perclass --dataset octmnist --sizes 40 --draws 0
  fi
  $PY $REV/run_baselines_rev.py --method $method --dataset mnist    --sizes 1000 5000 10000 20000
  $PY $REV/run_baselines_rev.py --method $method --dataset fmnist   --sizes 1000 5000 10000
  $PY $REV/run_baselines_rev.py --method $method --dataset octmnist --sizes 1000 5000
done
$PY $REV/run_fastgan_rev.py --mode mixed --dataset octmnist --sizes 1000 5000
$PY $REV/run_fastgan_rev.py --mode perclass --dataset mnist  --sizes 30 --draws 0
$PY $REV/run_fastgan_rev.py --mode perclass --dataset fmnist --sizes 30 --draws 0
echo "BASELINE QUEUE DONE"
