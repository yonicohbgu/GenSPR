# Ablation & Baseline Experiments

Scripts implementing the experiments proposed in the paper review.

## Experiments

| Script | What it tests | Priority |
|--------|---------------|----------|
| `ablation_sp.py` | SP contribution: random projection, linear encoder, no-encoder (MAE-style) | Critical |
| `baseline_vae.py` | Small convolutional VAE trained from scratch on the same data | High |
| `memorization_analysis.py` | Nearest-neighbor distances + coverage to rule out memorization | High |
| `eval_metrics.py` | FID / LPIPS evaluation with configurable sample count (default 10K) | High |
| `run_all.py` | Orchestrator that runs everything and saves a summary | — |

## Usage

```bash
cd htm_multilayer_gen_models-master
source .venv/bin/activate
python experiments/run_all.py            # run everything
python experiments/ablation_sp.py        # just the SP ablation
python experiments/eval_metrics.py --num-samples 10000   # just metrics
```
