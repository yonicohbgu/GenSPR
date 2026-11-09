#!/usr/bin/env python3
"""
Independent classifiers for Item 3b (class assignment and class consistency).

One small CNN per dataset, trained on the full 60,000-image training split;
it never sees the small training subsets or any generated sample.

    python experiments/revision/classifier.py --train mnist fmnist
    python experiments/revision/classifier.py --score <run dir with generated/>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CONFIDENT = 0.9


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.drop(F.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)


def _path(dataset):
    return os.path.join(C.RESULTS_ROOT, "_classifiers", f"{dataset}.pt")


def train(dataset, epochs=None):
    """OCTMNIST's training split is imbalanced, so its loss is weighted by inverse class frequency."""
    dev = C.device()
    epochs = epochs or (20 if dataset == "octmnist" else 8)
    tr, te = C._mnist_split(dataset, True), C._mnist_split(dataset, False)
    xtr, ytr = tr.data.float().div(255)[:, None], tr.targets
    xte, yte = te.data.float().div(255)[:, None].to(dev), te.targets.to(dev)
    model = SmallCNN(C.N_CLASSES[dataset]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    counts = torch.bincount(ytr).float()
    weight = (counts.sum() / counts / len(counts)).to(dev) if dataset == "octmnist" else None
    best_acc, best_state = -1.0, None
    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=g)
        for s in range(0, len(xtr), 128):
            b = perm[s:s + 128]
            loss = F.cross_entropy(model(xtr[b].to(dev)), ytr[b].to(dev), weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = sum((model(xte[s:s + 1000]).argmax(1) == yte[s:s + 1000]).sum().item()
                      for s in range(0, len(xte), 1000)) / len(xte)
        print(f"{dataset} epoch {ep + 1}: test acc {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    os.makedirs(os.path.dirname(_path(dataset)), exist_ok=True)
    torch.save({"state": best_state, "test_acc": best_acc}, _path(dataset))
    return best_acc


def load(dataset):
    ck = torch.load(_path(dataset), map_location="cpu")
    m = SmallCNN(C.N_CLASSES[dataset])
    m.load_state_dict(ck["state"])
    return m.eval()


@torch.no_grad()
def probs(dataset, images):
    m = load(dataset)
    return torch.cat([F.softmax(m(images[s:s + 1000]), 1) for s in range(0, len(images), 1000)])


def class_metrics(dataset, images, nn_labels=None):
    k = C.N_CLASSES[dataset]
    p = probs(dataset, images)
    conf, label = p.max(1)
    hist = torch.bincount(label, minlength=k).float()
    q = hist / hist.sum()
    entropy = float(-(q[q > 0] * q[q > 0].log()).sum())
    out = {
        "class_hist": hist.int().tolist(),
        "class_entropy": entropy,
        "class_entropy_max": float(np.log(k)),
        "kl_from_uniform": float(np.log(k) - entropy),
        "mean_max_softmax": float(conf.mean()),
        "class_consistent_rate": float((conf > CONFIDENT).mean(dtype=torch.float)),
    }
    if nn_labels is not None:
        nn_labels = torch.as_tensor(nn_labels)
        confident = conf > CONFIDENT
        tab = torch.zeros(k, k, dtype=torch.int)
        for a, b in zip(label[confident], nn_labels[confident]):
            tab[a, b] += 1
        out["confident_label_vs_nn_label"] = tab.tolist()
        out["nn_mismatch_rate_all"] = float((label != nn_labels).mean(dtype=torch.float))
        out["nn_mismatch_rate_confident"] = float(
            (label[confident] != nn_labels[confident]).mean(dtype=torch.float)) if confident.any() else None
    return out


def score_run(run_dir):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    files = sorted(glob.glob(os.path.join(run_dir, "generated", "*.png")))
    x = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255
                                   for p in files]))[:, None]
    nn_path = os.path.join(run_dir, "nn_train_label.npy")
    nn_labels = np.load(nn_path) if os.path.exists(nn_path) else None
    m["classifier"] = class_metrics(m["dataset"], x, nn_labels)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    return m["classifier"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="*", default=[])
    p.add_argument("--score", nargs="*", default=[])
    a = p.parse_args()
    for d in a.train:
        train(d)
    for r in a.score:
        print(r, json.dumps(score_run(r)))
