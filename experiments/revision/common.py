"""
Shared data draws and evaluation for the 2026-11 revision.

Every method is trained on the same class-balanced subsets (draw seeds 0-4)
and evaluated with one pipeline:
  * FID and KID: clean-fid (mode="clean") on 10,000 generated samples against
    a fixed held-out reference set per dataset.
  * LPIPS (VGG) nearest-neighbour distance to the training subset.
  * Coverage (Naeem et al., 2020): fraction of training images whose k=5
    nearest-neighbour LPIPS ball contains a generated sample, with the
    generated set subsampled to the training-set size.

CelebA n=40 is a fixed curated subset (frontal faces, zero head tilt) shared by
every draw; larger CelebA sizes are random draws from the training partition.

Reference sets:
  * MNIST / Fashion-MNIST: the 10,000-image test split, 28x28.
  * OCTMNIST: the 1,000-image test split, 28x28.
  * CelebA: the first 10,000 images of the official test partition
    (182638.jpg onward), grayscale, 64x64, same preprocessing as training.
"""
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets
from torchvision.utils import save_image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "results", "revision_2026_11")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
CELEBA_DIR = os.path.join(DATA_ROOT, "celeba_stripes", "img_align_celeba")
CELEBA_TRAIN_LAST = 162770
CELEBA_TEST_FIRST = 182638

def _curated_40():
    """First 40 image IDs of celeba_filtered.csv (frontal faces, zero head tilt)."""
    import csv
    path = os.path.join(DATA_ROOT, "celeba_dataset", "celeba_filtered.csv")
    with open(path) as f:
        ids = sorted(r["image_id"] for r in csv.DictReader(f))
    return [int(i[:6]) - 1 for i in ids[:40]]


CELEBA_CURATED_40 = _curated_40()

NATIVE_SIZE = {"mnist": 28, "fmnist": 28, "octmnist": 28, "celeba": 64}
N_CLASSES = {"mnist": 10, "fmnist": 10, "octmnist": 4, "celeba": 1}
OCTMNIST_NPZ = os.path.join(DATA_ROOT, "medmnist", "octmnist.npz")


class _Split:
    """Minimal torchvision-like view (data uint8 [N,28,28], targets int64 [N])."""

    def __init__(self, data, targets):
        self.data, self.targets = torch.from_numpy(data), torch.from_numpy(targets.astype(np.int64))
N_GEN = 10_000
DRAWS = [0, 1, 2, 3, 4]


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _mnist_split(dataset, train):
    if dataset == "octmnist":
        d = np.load(OCTMNIST_NPZ)
        key = "train" if train else "test"
        return _Split(d[f"{key}_images"], d[f"{key}_labels"].ravel())
    cls = datasets.FashionMNIST if dataset == "fmnist" else datasets.MNIST
    return cls(DATA_ROOT, train=train, download=True)


def draw_indices(dataset, n, draw):
    """Class-balanced subset of the training split; identical for every method."""
    rng = np.random.default_rng(draw)
    if dataset == "celeba":
        if n == 40:
            return CELEBA_CURATED_40
        return sorted(rng.choice(CELEBA_TRAIN_LAST, size=n, replace=False).tolist())
    targets = _mnist_split(dataset, True).targets.numpy()
    per_class = n // N_CLASSES[dataset]
    idx = []
    for c in range(N_CLASSES[dataset]):
        pool = np.flatnonzero(targets == c)
        idx.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(idx)


def _load_celeba(i, size):
    path = os.path.join(CELEBA_DIR, f"{i + 1:06d}.jpg")
    img = Image.open(path).convert("L").resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]


def load_subset(dataset, indices, size=None):
    """Return (images[N,1,H,W] in [0,1], labels[N] or None)."""
    size = size or NATIVE_SIZE[dataset]
    if dataset == "celeba":
        return torch.stack([_load_celeba(i, size) for i in indices]), None
    ds = _mnist_split(dataset, True)
    x = ds.data[indices].float().div(255.0)[:, None]
    if size != 28:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x, ds.targets[indices].clone()


def reference_dir(dataset):
    """Materialise the held-out reference set as PNGs (once)."""
    out = os.path.join(RESULTS_ROOT, "_reference", dataset)
    if os.path.isdir(out) and len(os.listdir(out)) >= N_GEN:
        return out
    os.makedirs(out, exist_ok=True)
    if dataset == "celeba":
        imgs = [_load_celeba(CELEBA_TEST_FIRST - 1 + k, 64) for k in range(N_GEN)]
    else:
        ds = _mnist_split(dataset, False)
        imgs = list(ds.data[:N_GEN].float().div(255.0)[:, None])
    if os.path.isdir(out) and len(os.listdir(out)) >= len(imgs):
        return out
    for k, img in enumerate(imgs):
        save_image(img, os.path.join(out, f"{k:05d}.png"))
    return out


def save_generated(images, out_dir, dataset):
    """Save generated samples at the dataset's native resolution."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    size = NATIVE_SIZE[dataset]
    images = images.detach().float().cpu().clamp(0, 1)
    if images.shape[1] == 3:
        images = images.mean(1, keepdim=True)
    if images.shape[-1] != size:
        images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    for k, img in enumerate(images):
        save_image(img, os.path.join(out_dir, f"{k:05d}.png"))
    return images


def fid_kid(gen_dir, dataset):
    from cleanfid import fid
    # clean-fid loads its Inception weights from /tmp, which macOS clears; its own download fails behind the proxy
    weights = "/tmp/inception-2015-12-05.pt"
    backup = os.path.join(RESULTS_ROOT, "_cache", "inception-2015-12-05.pt")
    if not os.path.exists(weights) and os.path.exists(backup):
        shutil.copy(backup, weights)
    name = f"genspr_rev_{dataset}"
    dev = torch.device(os.environ.get("GENSPR_FID_DEVICE", device()))
    if not fid.test_stats_exists(name, mode="clean"):
        fid.make_custom_stats(name, reference_dir(dataset), mode="clean", device=dev)
    kw = dict(dataset_name=name, mode="clean", dataset_split="custom",
              verbose=False, device=dev, num_workers=0)
    return (float(fid.compute_fid(gen_dir, **kw)),
            float(fid.compute_kid(gen_dir, **kw)) * 1e3)


_LPIPS = None


def _lpips(dev):
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="vgg", verbose=False).eval().to(dev)
    return _LPIPS


def _to_lpips(x, dev):
    return (x.repeat(1, 3, 1, 1) * 2 - 1).to(dev)


@torch.no_grad()
def _lpips_rows(queries, refs, dev, chunk=2048):
    """Full LPIPS distance matrix [len(queries), len(refs)]."""
    net = _lpips(dev)
    out = torch.empty(len(queries), len(refs))
    r = _to_lpips(refs, dev)
    for i in range(len(queries)):
        q = _to_lpips(queries[i:i + 1], dev)
        for j in range(0, len(refs), chunk):
            rr = r[j:j + chunk]
            out[i, j:j + len(rr)] = net(q.expand(len(rr), -1, -1, -1), rr).flatten().cpu()
    return out


@torch.no_grad()
def _nn_lpips(queries, refs, dev, k=50, exact_limit=2_000_000):
    """LPIPS nearest-neighbour distance and index for each query.

    Exact when len(queries)*len(refs) <= exact_limit; otherwise the LPIPS
    search is restricted to the k nearest references in pixel L2.
    """
    if len(queries) * len(refs) <= exact_limit:
        d = _lpips_rows(queries, refs, dev)
        val, idx = d.min(1)
        return val.numpy(), idx.numpy()
    qf, rf = queries.flatten(1).to(dev), refs.flatten(1).to(dev)
    net = _lpips(dev)
    vals, idxs = np.empty(len(queries)), np.empty(len(queries), dtype=int)
    for s in range(0, len(queries), 256):
        cand = torch.cdist(qf[s:s + 256], rf).topk(min(k, len(refs)), largest=False).indices.cpu()
        for t, c in enumerate(cand):
            q = _to_lpips(queries[s + t:s + t + 1], dev).expand(len(c), -1, -1, -1)
            d = net(q, _to_lpips(refs[c], dev)).flatten().cpu()
            j = int(d.argmin())
            vals[s + t], idxs[s + t] = float(d[j]), int(c[j])
    return vals, idxs


@torch.no_grad()
def _tau(train, dev, max_pairs=5000, seed=0):
    n = len(train)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = [pairs[p] for p in rng.choice(len(pairs), max_pairs, replace=False)]
    net = _lpips(dev)
    a = torch.stack([train[i] for i, _ in pairs])
    b = torch.stack([train[j] for _, j in pairs])
    d = []
    for s in range(0, len(pairs), 1024):
        d.append(net(_to_lpips(a[s:s + 1024], dev), _to_lpips(b[s:s + 1024], dev)).flatten().cpu())
    return float(torch.cat(d).median())


COVERAGE_K = 5


@torch.no_grad()
def _real_knn_radius(train, dev, k=COVERAGE_K, cand=50, exact_limit=2_000_000):
    """LPIPS distance from each training image to its k-th nearest other training image."""
    n = len(train)
    if n * n <= exact_limit:
        d = _lpips_rows(train, train, dev)
        d.fill_diagonal_(float("inf"))
        return d.sort(1).values[:, k - 1].numpy()
    flat = train.flatten(1).to(dev)
    net = _lpips(dev)
    rad = np.empty(n)
    for s in range(0, n, 256):
        c = torch.cdist(flat[s:s + 256], flat).topk(cand + 1, largest=False).indices.cpu()
        for t, row in enumerate(c):
            row = row[row != s + t][:cand]
            q = _to_lpips(train[s + t:s + t + 1], dev).expand(len(row), -1, -1, -1)
            rad[s + t] = float(net(q, _to_lpips(train[row], dev)).flatten().cpu().sort().values[k - 1])
    return rad


def naeem_coverage(train_images, gen, dev, seed=0):
    """Coverage of Naeem et al. (2020) with k=5 and equal real/generated counts.

    The generated set is subsampled to the training-set size (at most 10,000),
    averaged over random subsets; with all 10,000 samples against a small
    training set every method saturates at 1.0.
    """
    rad = _real_knn_radius(train_images, dev)
    m = min(len(train_images), len(gen))
    repeats = 10 if len(train_images) < 1000 else 3
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(repeats):
        sub = gen[rng.choice(len(gen), m, replace=False)]
        d, _ = _nn_lpips(train_images, sub, dev)
        vals.append(float((d <= rad).mean()))
    return float(np.mean(vals)), float(np.std(vals))


def evaluate(gen_images, train_images, dataset, out_dir, extra=None, train_labels=None):
    """Save samples, compute every metric, write metrics.json. Returns the dict."""
    dev = device()
    gen_dir = os.path.join(out_dir, "generated")
    gen = save_generated(gen_images, gen_dir, dataset)
    fid_v, kid_v = fid_kid(gen_dir, dataset)
    if len(gen) * len(train_images) <= 2_000_000:
        d = _lpips_rows(train_images, gen, dev)
        nn_d, nn_i = (t.numpy() for t in d.min(0))
        rev_d = d.min(1).values.numpy()
    else:
        nn_d, nn_i = _nn_lpips(gen, train_images, dev)
        rev_d, _ = _nn_lpips(train_images, gen, dev)
    tau = _tau(train_images, dev)
    cov_mean, cov_std = naeem_coverage(train_images, gen, dev)
    m = {
        "dataset": dataset,
        "n_train": len(train_images),
        "n_gen": len(gen),
        "fid": fid_v,
        "kid_x1e3": kid_v,
        "lpips_nn_mean": float(nn_d.mean()),
        "lpips_nn_frac_below": {t: float((nn_d < t).mean()) for t in (0.05, 0.10, 0.15)},
        "coverage_tau_all_samples": float((rev_d <= tau).mean()),
        "coverage": cov_mean,
        "coverage_std_over_subsets": cov_std,
        "coverage_def": f"naeem_k{COVERAGE_K}_equal_counts",
        "tau": tau,
    }
    if extra:
        m.update(extra)
    np.save(os.path.join(out_dir, "lpips_nn.npy"), nn_d)
    if train_labels is not None:
        np.save(os.path.join(out_dir, "nn_train_label.npy"), train_labels.numpy()[nn_i])
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    return m


def run_dir(*parts):
    path = os.path.join(RESULTS_ROOT, *[str(p) for p in parts])
    os.makedirs(path, exist_ok=True)
    return path
