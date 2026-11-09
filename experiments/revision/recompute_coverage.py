#!/usr/bin/env python3
"""Backfill subset coverage for finished runs from their saved samples.

    python experiments/revision/recompute_coverage.py <run dir> [<run dir> ...]
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def recompute(run_dir):
    path = os.path.join(run_dir, "metrics.json")
    with open(path) as f:
        m = json.load(f)
    if m.get("coverage_def") == f"naeem_k{C.COVERAGE_K}_equal_counts":
        return m
    files = sorted(glob.glob(os.path.join(run_dir, "generated", "*.png")))
    gen = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255
                                     for p in files]))[:, None]
    parts = run_dir.rstrip("/").split(os.sep)
    n, draw = int(parts[-2][1:]), int(parts[-1][4:])
    train, _ = C.load_subset(m["dataset"], C.draw_indices(m["dataset"], n, draw))
    m["coverage_tau_all_samples"] = m.pop("coverage_all_samples", m["coverage"])
    m["coverage"], m["coverage_std_over_subsets"] = C.naeem_coverage(train, gen, C.device())
    m["coverage_def"] = f"naeem_k{C.COVERAGE_K}_equal_counts"
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
    return m


if __name__ == "__main__":
    for r in sys.argv[1:]:
        m = recompute(r)
        print(r, f"coverage {m['coverage']:.3f} ± {m['coverage_std_over_subsets']:.3f} "
                 f"(old tau definition {m['coverage_tau_all_samples']:.2f})")
