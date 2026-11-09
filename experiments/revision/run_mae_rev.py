#!/usr/bin/env python3
"""
MAE-from-scratch baseline on CelebA 64x64 for the 2026-11 revision.

Reuses the original implementation (ablation_sp.IdentityEncoder / AblationDecoder /
train_ablation, as called by run_large_n.run_mae): the masked image is flattened, a bias-free
linear map adapts it to 1,024 dims, and the hierarchical CNN decoder reconstructs the image;
150 epochs max, early stopping on SSIM (patience 15). Each original is replicated
min(240, 9600 // n) times as for hierarchical GenSPR. A fresh mask is drawn per batch.

  --keep 0.85  the original code (IdentityEncoder mask_ratio=0.15 hides 15% of pixels)
  --keep 0.15  the visible fraction of hierarchical GenSPR, as the manuscript describes

--encoder selects the Table 5 encoder-replacement variant (same decoder and training):
  identity           MAE-FS (output mae_fs[_keepK])
  random_projection  frozen dense random linear map (output randproj[_keepK])
  linear             MLP encoder 4096-512-1024, trained jointly (output mlpenc[_keepK])

    python experiments/revision/run_mae_rev.py --keep 0.85 --sizes 40 --draws 0 1 2 3 4
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.PROJECT_ROOT, "experiments"))
from ablation_sp import (AblationDecoder, IdentityEncoder, LinearEncoder,  # noqa: E402
                         RandomProjectionEncoder, train_ablation)

ENCODERS = {
    "identity": ("mae_fs", lambda m: IdentityEncoder(64 * 64, 1024, mask_ratio=m)),
    "random_projection": ("randproj", lambda m: RandomProjectionEncoder(64 * 64, 1024, mask_ratio=m)),
    "linear": ("mlpenc", lambda m: LinearEncoder(64 * 64, 512, 1024, mask_ratio=m)),
}


def run(n, draw, keep, encoder="identity"):
    base, make_enc = ENCODERS[encoder]
    name = base if keep == 0.85 else f"{base}_keep{int(keep * 100)}"
    out = C.run_dir(name, "celeba", f"n{n}", f"draw{draw}")
    if os.path.exists(os.path.join(out, "metrics.json")):
        print(f"skip {out}")
        return
    dev = C.device()
    torch.manual_seed(draw)
    x, _ = C.load_subset("celeba", C.draw_indices("celeba", n, draw))
    reps = max(1, min(240, 9600 // n))
    g = torch.Generator().manual_seed(draw)
    loader = DataLoader(TensorDataset(x.repeat(reps, 1, 1, 1)), batch_size=256, shuffle=True,
                        drop_last=True, generator=g)

    t = time.time()
    enc = make_enc(1 - keep)
    dec = AblationDecoder(1024, 64, channels=1, inner_dim=256)
    train_ablation(enc, dec, loader, dev, epochs=150, patience=15, save_dir=out)
    ckpt = torch.load(os.path.join(out, "best_model.pt"), map_location=dev, weights_only=False)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    os.remove(os.path.join(out, "best_model.pt"))
    enc, dec = enc.to(dev).eval(), dec.to(dev).eval()
    with torch.no_grad():
        gen = torch.cat([dec(enc(torch.rand(500, 1, 64, 64, device=dev))).cpu()
                         for _ in range(C.N_GEN // 500)]).clamp(0, 1)
    extra = {"method": name, "encoder": encoder, "draw": draw, "keep_prob": keep,
             "replicas_per_original": reps,
             "best_epoch": ckpt["epoch"] + 1, "best_ssim": ckpt["best_ssim"],
             "train_and_generate_s": time.time() - t}

    t = time.time()
    m = C.evaluate(gen, x, "celeba", out, extra=extra)
    m["eval_s"] = time.time() - t
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(f"{name} celeba n={n} draw={draw}: FID {m['fid']:.1f} KID {m['kid_x1e3']:.2f} "
          f"NN {m['lpips_nn_mean']:.3f} cov {m['coverage']:.2f} "
          f"train {extra['train_and_generate_s'] / 60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keep", type=float, choices=[0.85, 0.15], default=0.85)
    p.add_argument("--sizes", type=int, nargs="+", default=[40])
    p.add_argument("--draws", type=int, nargs="+", default=[0])
    p.add_argument("--encoder", choices=list(ENCODERS), default="identity")
    a = p.parse_args()
    for n in a.sizes:
        for d in a.draws:
            run(n, d, a.keep, a.encoder)


if __name__ == "__main__":
    main()
