#!/usr/bin/env python3
"""
FastGAN for the 2026-11 revision (Items 5 and 8).

  mixed     one generator trained on the whole class-balanced subset
  perclass  one generator per class on its n/10 images, samples pooled

Training settings are those of Appendix A (run_fastgan.train_fastgan_single_class,
200 epochs, DiffAugment). Each run writes metrics.json and timing to
experiments/results/revision_2026_11/fastgan_<mode>/<dataset>/n<n>/draw<d>/.

    python experiments/revision/run_fastgan_rev.py --mode mixed --dataset mnist \
        --sizes 30 300 1000 --draws 0 1 2 3 4
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.PROJECT_ROOT, "experiments"))
from run_fastgan import train_fastgan_single_class  # noqa: E402

TRAIN_SIZE = 32


def run(mode, dataset, n, draw, epochs, parity=False):
    method = f"fastgan_{mode}" + ("_r9600" if parity else "")
    out = C.run_dir(method, dataset, f"n{n}", f"draw{draw}")
    if os.path.exists(os.path.join(out, "metrics.json")):
        print(f"skip {out}")
        return
    dev = C.device()
    torch.manual_seed(draw)
    idx = C.draw_indices(dataset, n, draw)
    x, y = C.load_subset(dataset, idx)
    x32 = F.interpolate(x, size=(TRAIN_SIZE, TRAIN_SIZE), mode="bilinear", align_corners=False)

    # parity: the same min(240, 9600 // n) replicas per original as GenSPR, DCGAN, VAE, MAE-FS
    rep = {"replicate_to": 9600, "max_replicas": 240} if parity else {}
    t0 = time.time()
    if mode == "mixed" or y is None:
        gen = train_fastgan_single_class(x32, list(range(len(x32))), TRAIN_SIZE, dev,
                                         os.path.join(out, "ckpt"), epochs=epochs, n_gen=C.N_GEN,
                                         **rep)
    else:
        parts = []
        per = C.N_GEN // C.N_CLASSES[dataset]
        for c in range(C.N_CLASSES[dataset]):
            cls_idx = (y == c).nonzero().flatten().tolist()
            parts.append(train_fastgan_single_class(
                x32, cls_idx, TRAIN_SIZE, dev, os.path.join(out, "ckpt", f"class{c}"),
                epochs=epochs, n_gen=per))
        gen = torch.cat(parts)
    train_gen_s = time.time() - t0

    t1 = time.time()
    m = C.evaluate(gen, x, dataset, out, train_labels=y,
                   extra={"method": method, "draw": draw, "epochs": epochs, **rep,
                          "train_and_generate_s": train_gen_s})
    m["eval_s"] = time.time() - t1
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(f"{mode} {dataset} n={n} draw={draw}: FID {m['fid']:.1f} KID {m['kid_x1e3']:.2f} "
          f"NN {m['lpips_nn_mean']:.3f} cov {m['coverage']:.2f} "
          f"train {train_gen_s / 60:.1f} min")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mixed", "perclass"], required=True)
    p.add_argument("--dataset", choices=["mnist", "fmnist", "octmnist", "celeba"], required=True)
    p.add_argument("--sizes", type=int, nargs="+", required=True)
    p.add_argument("--draws", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--parity", action="store_true",
                   help="replicate to 9,600 like the other CelebA methods (output fastgan_<mode>_r9600)")
    a = p.parse_args()
    for n in a.sizes:
        for d in a.draws:
            run(a.mode, a.dataset, n, d, a.epochs, a.parity)


if __name__ == "__main__":
    main()
