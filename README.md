# GenSPR: Image Generation Without External Priors via Hierarchical Temporal Memory

Reference implementation of **GenSPR**, a generative architecture that produces
diverse, multi-class images from as few as 30 training images **without any
external pretraining**. GenSPR pairs a Hierarchical Temporal Memory (HTM)
Spatial Pooler encoder with a small neural decoder, and uses aggressive input
masking both as the training signal and as the source of generative variability.

This repository contains the model, all baselines, and the evaluation scripts
used to produce the tables and figures in the paper.

## Repository layout

| Path | Contents |
|------|----------|
| `models/` | `htm_model.py` (Spatial Pooler pyramid), `cnn_model.py` (decoder) |
| `dataloaders/` | MNIST, Fashion-MNIST, CelebA, and replicated/masked dataset wrappers |
| `configs/` | JSON configurations; hyperparameters match the paper's appendix |
| `experiments/` | Training runners, baselines, evaluation, and figure generation |
| `utils/` | Config management, HTM helpers, analysis, visualization |
| `scripts/` | Convenience entry points |
| `notebooks/` | Exploratory analysis and CelebA curation |

## Installation

Requires Python 3.11+ (developed on 3.13) and, for GPU acceleration on Apple
Silicon, a PyTorch build with the `mps` backend.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Installing htm.core

`htm.core` is **not available on PyPI** and must be built from source. It
provides the `SpatialPooler` and `ScalarEncoder` used throughout:

```bash
pip install git+https://github.com/htm-community/htm.core.git
```

If the build fails, follow the platform-specific instructions in the
[htm.core repository](https://github.com/htm-community/htm.core). A C++
toolchain and CMake are required.

## Data

MNIST and Fashion-MNIST download automatically on first use into `data/`.

CelebA must be obtained separately from the
[official source](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and is not
redistributed here. The hierarchical experiments use a curated 40-image subset
of frontal, landmark-aligned faces; `notebooks/celeba_eda_and_filter.ipynb`
documents the filtering and alignment procedure. Place the resulting images
under `data/celeba40/`.

Datasets, generated samples, cached Spatial Poolers, and trained checkpoints are
excluded from version control (see `.gitignore`).

## Reproducing the paper

All commands assume the virtual environment is active.

```bash
# GenSPR + VAE + MAE + encoder ablations on MNIST / Fashion-MNIST
python experiments/run_mnist_all.py --dataset both

# Hierarchical GenSPR on the curated 40-image CelebA subset
python experiments/run_genspr_full.py --image-size 64 --num-samples 40

# Baselines
python experiments/run_dcgan.py
python experiments/run_fastgan.py --sizes 30 300 1000 5000 10000 20000

# Evaluation: FID, LPIPS nearest-neighbour, coverage
python experiments/eval_metrics.py
python experiments/eval_memorization.py
python experiments/eval_fastgan.py

# Ablations
python experiments/run_mask_sweep.py      # mask-ratio sensitivity
python experiments/run_fix4_postproc.py   # post-processing ablation
```

Figures are written to the directory given by the `GENSPR_FIGURE_DIR`
environment variable:

```bash
GENSPR_FIGURE_DIR=./figures python experiments/make_fastgan_scaling_figures.py
```

Results are written under `experiments/results/<dataset>_n<size>/<method>/`.
See `experiments/README.md` for a per-script description.

## Configuration

Model hyperparameters live in `configs/*.json` and are loaded through
`utils/config_manager.py`, which derives the Spatial Pooler input dimensions
from the image size, encoder resolution, and split factor, then validates the
whole configuration before training.

`configs/e14_celeba40_s64.json` is the configuration used for the hierarchical
CelebA results reported in the paper.

## Citation

```bibtex
@article{omer2026genspr,
  title  = {GenSPR: Image Generation Without External Priors via
            Hierarchical Temporal Memory},
  author = {Omer, Yuval and Cohen, Yoni and Katz, Gilad},
  year   = {2026}
}
```

## Acknowledgements

Built on [htm.core](https://github.com/htm-community/htm.core), a maintained
fork of Numenta's NuPIC.
