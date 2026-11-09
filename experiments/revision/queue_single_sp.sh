#!/bin/bash
# Single-SP GenSPR: main table, then Items 3a, 7b, 10, 7a, then classifier scoring.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
REV=experiments/revision
RES=experiments/results/revision_2026_11
S="$PY $REV/run_genspr_single_rev.py"

# Main table (Items 4, 5, 9)
$S --dataset mnist --sizes 30 300 --draws 0 1 2 3 4
$S --dataset fmnist --sizes 30 300 --draws 0 1 2 3 4
$S --dataset octmnist --sizes 40 300 --draws 0 1 2 3 4
$S --dataset mnist --sizes 1000 5000 10000 20000
$S --dataset fmnist --sizes 1000 5000 10000
$S --dataset octmnist --sizes 1000 5000
echo "MAIN TABLE DONE"

# Item 3a: per-class vs mixed GenSPR
$S --dataset mnist --sizes 30 --draws 0 1 2 3 4 --mode perclass
$S --dataset fmnist --sizes 30 --draws 0 1 2 3 4 --mode perclass
$S --dataset octmnist --sizes 40 --draws 0 1 2 3 4 --mode perclass

# Item 7b: plateau test (full SP = boost 7 is the main-table run)
$S --dataset mnist --sizes 10000 20000 --variants noboost boost2 anneal
$S --dataset fmnist --sizes 10000 --variants noboost boost2 anneal

# Item 10: masks per original at n=30 (333 is the main-table run)
for m in 1 10 100; do
  $S --dataset mnist --sizes 30 --draws 0 1 2 --masks $m
  $S --dataset fmnist --sizes 30 --draws 0 1 2 --masks $m
done

# Item 7a: component ablation
V="nohebb noboost nohebb_noboost density05 density10 density20 localinh"
$S --dataset mnist --sizes 30 --draws 0 1 2 --variants $V
$S --dataset fmnist --sizes 30 --draws 0 1 2 --variants $V
$S --dataset mnist --sizes 5000 --variants $V
$S --dataset fmnist --sizes 5000 --variants $V

$PY $REV/classifier.py --score $(ls -d $RES/genspr_single*/{mnist,fmnist,octmnist}/n*/draw*)
echo "SINGLE SP QUEUE DONE"
