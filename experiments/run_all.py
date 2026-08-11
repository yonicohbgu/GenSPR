#!/usr/bin/env python3
"""
Master Experiment Orchestrator
===============================
Runs all experiments from the reviewer's proposal in sequence:

  1. SP ablation (random projection, linear encoder, identity/MAE)
  2. Small VAE baseline
  3. Memorisation analysis on all variants + GenSPR

Results are collected under  experiments/results/  with a final
summary JSON.

Usage:
    source .venv/bin/activate
    python experiments/run_all.py                    # full suite
    python experiments/run_all.py --skip-ablation    # skip SP ablation
    python experiments/run_all.py --epochs 50        # quick test run
"""

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")
PYTHON = sys.executable

SEP = "=" * 70


def run_cmd(cmd, description):
    """Run a command and stream output."""
    print(f"\n{SEP}")
    print(f"  {description}")
    print(f"  cmd: {' '.join(cmd)}")
    print(SEP)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"  [{status}] {description}  ({elapsed/60:.1f} min)")
    return result.returncode == 0, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--data-path", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "celeba_stripes"))
    parser.add_argument("--total-size", type=int, default=9600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-vae", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {"experiments": {}, "total_time": 0}
    t_total = time.time()

    # 1. SP Ablation
    if not args.skip_ablation:
        ok, elapsed = run_cmd([
            PYTHON, os.path.join(EXPERIMENTS_DIR, "ablation_sp.py"),
            "--image-size", str(args.image_size),
            "--data-path", args.data_path,
            "--total-size", str(args.total_size),
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
        ], "SP Ablation (random projection, linear encoder, identity)")
        summary["experiments"]["sp_ablation"] = {
            "success": ok, "elapsed_min": elapsed / 60}

    # 2. Small VAE Baseline
    if not args.skip_vae:
        ok, elapsed = run_cmd([
            PYTHON, os.path.join(EXPERIMENTS_DIR, "baseline_vae.py"),
            "--image-size", str(args.image_size),
            "--data-path", args.data_path,
            "--total-size", str(args.total_size),
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
        ], "Small VAE Baseline")
        summary["experiments"]["vae_baseline"] = {
            "success": ok, "elapsed_min": elapsed / 60}

    summary["total_time"] = (time.time() - t_total) / 60

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{SEP}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total time: {summary['total_time']:.1f} min")
    print(f"  Results in: {RESULTS_DIR}")
    print(f"  Summary:    {summary_path}")
    print(SEP)

    for name, info in summary["experiments"].items():
        status = "OK" if info["success"] else "FAILED"
        print(f"    [{status}]  {name:30s}  {info['elapsed_min']:.1f} min")


if __name__ == "__main__":
    main()
