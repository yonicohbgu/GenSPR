#!/usr/bin/env python3
"""
Hierarchical GenSPR on CelebA 64x64 for the 2026-11 revision (Items 1, 5, 7).

Protocol of run_fix5_celeba.py / run_large_n.py (the runs behind the reported
numbers): each original is replicated min(240, 9600 // n) times, every replica
gets its own fixed mask (HTMModel mask_ratio 0.15), the 21 SPs learn for a few
passes, are frozen, and the CNN decoder is trained on the cached SDRs. Per seed,
the SP seed, masks, shuffling and decoder init all change. Stage timings are
recorded separately; 10,000 samples are evaluated with common.evaluate.

  n=40 is the curated subset (common.CELEBA_CURATED_40); larger n are random
  draws from the train partition.

    python experiments/revision/run_genspr_hier_rev.py --sizes 40 --draws 0 1 2 3 4
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

sys.path.insert(0, C.PROJECT_ROOT)
from models.cnn_model import CNNModel  # noqa: E402
from models.htm_model import HTMModel  # noqa: E402
from utils.config_manager import generate_config  # noqa: E402

SP_SEED_BASE = 42

# Item 7 SP variants: overrides applied to both layers, or per layer under "L0"/"L1".
# "anneal" decays each layer's boostStrength linearly to 0 over the SP passes.
VARIANTS = {
    "full": {},
    "nohebb": {"synPermActiveInc": 0.0, "synPermInactiveDec": 0.0},
    "noboost": {"boostStrength": 0.0},
    "nohebb_noboost": {"synPermActiveInc": 0.0, "synPermInactiveDec": 0.0, "boostStrength": 0.0},
    "dense": {"L0": {"localAreaDensity": 0.20}, "L1": {"localAreaDensity": 0.30}},
    "boost2": {"boostStrength": 2.0},
    "boost7": {"boostStrength": 7.0},
    "anneal": {},
}


class _Indexed(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], i


def _sp_epochs(n):
    if n <= 40:
        return 5
    return 2 if n >= 5000 else 3


def _config(total_size, sp_seed, save_path, variant="full"):
    cfg = generate_config(
        image_size=64, base_enc=4, split_factor=4, channels=1,
        data_path=C.CELEBA_DIR,
        cnn={"inner_dim": 64, "learning_rate": 0.001},
        training={"batch_size": 256, "total_size": total_size, "patience": 10},
        save_path=save_path,
    )
    over = VARIANTS[variant]
    for layer in ("L0", "L1"):
        sp = cfg.spatial_pooler[layer]
        sp["seed"] = sp_seed
        sp.update({k: v for k, v in over.items() if k not in ("L0", "L1")})
        sp.update(over.get(layer, {}))
    return cfg


def _set_boost(htm, cfg, frac):
    """Scale every SP's boostStrength to frac x its configured value."""
    for layer, poolers in enumerate(htm.spatial_poolers):
        base = cfg.spatial_pooler["L0" if layer == 0 else "L1"]["boostStrength"]
        for sp in poolers:
            sp.setBoostStrength(base * frac)


def _column_usage(sdrs):
    """Distinct top-layer output columns active across the generated set, and usage entropy."""
    counts = (sdrs > 0).flatten(1).sum(0).double()
    p = counts[counts > 0] / counts.sum()
    return {"active_columns": int((counts > 0).sum()), "total_columns": counts.numel(),
            "column_usage_entropy": float(-(p * p.log()).sum()),
            "column_usage_entropy_max": float(np.log(counts.numel()))}


def train_decoder(cnn, sdrs, images, dev, out, seed, epochs=200, patience=10):
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3, betas=(0.5, 0.999))
    mse_fn = torch.nn.MSELoss()
    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=1)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(sdrs, images), batch_size=256, shuffle=True,
                        drop_last=True, generator=g)
    best, bad, best_epoch = -float("inf"), 0, 0
    ckpt = os.path.join(out, "best_cnn.pt")
    for epoch in range(epochs):
        cnn.train()
        ssims = []
        for s, img in loader:
            s, img = s.to(dev), img.to(dev)
            pred = cnn(s)
            ssim = ssim_fn(pred, img)
            loss = 0.15 * mse_fn(pred, img) + 0.85 * (1.0 - ssim)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ssims.append(ssim.item())
        avg = float(np.mean(ssims))
        if avg > best + 1e-3:
            best, bad, best_epoch = avg, 0, epoch + 1
            torch.save(cnn.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                break
    cnn.load_state_dict(torch.load(ckpt, map_location=dev))
    os.remove(ckpt)
    return {"decoder_best_epoch": best_epoch, "decoder_epochs_run": epoch + 1,
            "decoder_best_ssim": best}


def _sp_footprint(htm):
    """Synapse counts over all poolers; htm.core stores each permanence as a float32."""
    poolers = [sp for layer in htm.spatial_poolers for sp in layer]
    syn = sum(sp.connections.numSynapses() for sp in poolers)
    conn = 0
    for sp in poolers:
        counts = np.zeros(sp.getNumColumns(), dtype=np.uint32)
        sp.getConnectedCounts(counts)
        conn += int(counts.sum())
    return {"n_poolers": len(poolers), "potential_synapses": int(syn),
            "connected_synapses": int(conn), "permanence_bytes": int(4 * syn)}


def run(n, draw, variant="full", reps=None):
    method = "genspr_hier" if variant == "full" else f"genspr_hier_{variant}"
    if reps is not None:
        method += f"_reps{reps}"
    out = C.run_dir(method, "celeba", f"n{n}", f"draw{draw}")
    if os.path.exists(os.path.join(out, "metrics.json")):
        print(f"skip {out}")
        return
    dev = C.device()
    torch.manual_seed(draw)
    np.random.seed(draw)
    timing = {}

    t = time.time()
    idx = C.draw_indices("celeba", n, draw)
    x, _ = C.load_subset("celeba", idx)
    if reps is None:
        reps = max(1, min(240, 9600 // n))
    data = x.repeat(reps, 1, 1, 1)
    g = torch.Generator().manual_seed(draw)
    loader = DataLoader(_Indexed(data), batch_size=256, shuffle=True, drop_last=True,
                        num_workers=0, generator=g)
    timing["data_s"] = time.time() - t

    sp_seed = SP_SEED_BASE + draw
    cfg = _config(len(data), sp_seed, f"rev_hier_n{n}_d{draw}", variant)
    t = time.time()
    htm = HTMModel(config=cfg, mask_ratio=0.15, dataset_size=len(data), device=dev,
                   use_cache=False)
    timing["sp_init_s"] = time.time() - t

    t = time.time()
    sp_epochs = _sp_epochs(n)
    htm.learning_enabled = True
    for ep in range(sp_epochs):
        if variant == "anneal":
            _set_boost(htm, cfg, 1 - ep / max(1, sp_epochs - 1))
        for batch, bidx in loader:
            htm.process_batch(batch, bidx)
        print(f"  SP pass {ep + 1}/{sp_epochs} done ({(time.time() - t) / 60:.1f} min)", flush=True)
    timing["sp_learn_s"] = time.time() - t

    t = time.time()
    htm.learning_enabled = False
    sdrs, imgs = [], []
    for batch, bidx in loader:
        sdrs.append(htm.process_batch(batch, bidx).cpu())
        imgs.append(batch)
    sdrs, imgs = torch.cat(sdrs), torch.cat(imgs)
    timing["sp_encode_s"] = time.time() - t

    t = time.time()
    cnn = CNNModel(config=cfg).to(dev)
    info = train_decoder(cnn, sdrs, imgs, dev, out, draw)
    timing["decoder_s"] = time.time() - t

    t = time.time()
    cnn.eval()
    gen, gen_sdrs = [], []
    with torch.no_grad():
        for s in range(0, C.N_GEN, 250):
            noise = torch.randint(0, 256, (250, 1, 64, 64), dtype=torch.float32) / 255.0
            sdr = htm.process_batch(noise, -torch.arange(250)).to(dev)
            gen.append(cnn(sdr).cpu())
            gen_sdrs.append(sdr.cpu())
    gen = torch.cat(gen).clamp(0, 1)
    timing["generate_10k_s"] = time.time() - t

    extra = {"method": method, "variant": variant, "sp_overrides": VARIANTS[variant],
             **_column_usage(torch.cat(gen_sdrs)), "sp_footprint": _sp_footprint(htm),
             "draw": draw, "sp_seed": sp_seed, "sp_epochs": sp_epochs,
             "replicas_per_original": reps, "effective_train_size": len(data),
             "mask_keep_prob": 0.15, **info, "timing": timing,
             "train_and_generate_s": sum(timing.values())}
    t = time.time()
    m = C.evaluate(gen, x, "celeba", out, extra=extra)
    m["eval_s"] = time.time() - t
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(f"{method} celeba n={n} draw={draw}: cols {extra['active_columns']} FID {m['fid']:.1f} KID {m['kid_x1e3']:.2f} "
          f"NN {m['lpips_nn_mean']:.3f} cov {m['coverage']:.2f} | SP {timing['sp_learn_s'] / 60:.1f} "
          f"+ enc {timing['sp_encode_s'] / 60:.1f} + dec {timing['decoder_s'] / 60:.1f} "
          f"+ gen {timing['generate_10k_s'] / 60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[40])
    p.add_argument("--draws", type=int, nargs="+", default=[0])
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["full"])
    p.add_argument("--reps", type=int, default=None,
                   help="replicas per original (default min(240, 9600 // n))")
    a = p.parse_args()
    for v in a.variants:
        for n in a.sizes:
            for d in a.draws:
                run(n, d, v, a.reps)


if __name__ == "__main__":
    main()
