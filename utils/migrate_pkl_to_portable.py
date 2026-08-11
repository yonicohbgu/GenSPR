import pickle
import json
import os
from pathlib import Path


def migrate_pkl_to_portable(pkl_path, config, out_dir=None):
    """
    Load an existing HTM pickle (on the machine it was saved on),
    then re-save each SpatialPooler using htm.core's PORTABLE format.

    Args:
        pkl_path:  Path to existing .pkl file (e.g. "results/e11_.../trained_htm_epoch_10.pkl")
        config:    ConfigManager instance matching the pkl (needed to reconstruct on load)
        out_dir:   Destination directory for the portable checkpoint.
                   Defaults to pkl_path with .pkl stripped (e.g. "trained_htm_epoch_10/")
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"pkl not found: {pkl_path}")

    if out_dir is None:
        out_dir = pkl_path.with_suffix("")   # strip .pkl → directory name

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load on same architecture where it was saved (Mac ARM → Mac ARM)
    with open(pkl_path, "rb") as f:
        htm_pp = pickle.load(f)

    # Save each SP in portable cross-platform format
    poolers_per_layer = []
    for layer_idx, layer_sps in enumerate(htm_pp.spatial_poolers):
        poolers_per_layer.append(len(layer_sps))
        for sp_idx, sp in enumerate(layer_sps):
            sp.saveToFile(str(out_dir / f"L{layer_idx}_sp{sp_idx:02d}.htm"), fmt="PORTABLE")

    metadata = {
        "num_layers": len(htm_pp.spatial_poolers),
        "poolers_per_layer": poolers_per_layer,
        "learning_enabled": htm_pp.learning_enabled,
        "epoch_counter": getattr(htm_pp, "epoch_counter", 0),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    config.save(str(out_dir / "config.json"))
    print(f"Migrated {pkl_path} → {out_dir}/")
    return str(out_dir)


def migrate_all_in_dir(results_dir, config):
    """
    Batch-migrate all .pkl files found recursively under results_dir.

    Args:
        results_dir: Root results folder to scan (e.g. "results/e11_celeba_mixed_s128")
        config:      ConfigManager instance for this experiment
    """
    results_dir = Path(results_dir)
    pkls = sorted(results_dir.rglob("*.pkl"))
    if not pkls:
        print(f"No .pkl files found under {results_dir}")
        return

    print(f"Found {len(pkls)} pkl file(s) to migrate:")
    for p in pkls:
        print(f"  {p}")

    for pkl_path in pkls:
        migrate_pkl_to_portable(pkl_path, config)

    print("Migration complete. Original .pkl files have NOT been deleted.")
    print("Verify the portable directories load correctly before deleting .pkl files.")


if __name__ == "__main__":
    # Example usage — edit paths before running
    from utils import generate_config
    from utils.config_manager import ConfigManager

    config = ConfigManager("configs/e11_celeba_mixed_s128.json")
    migrate_all_in_dir("results/e11_celeba_mixed_s128", config)