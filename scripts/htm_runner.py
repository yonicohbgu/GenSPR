#!/usr/bin/env python3

import argparse
import os
import sys
import warnings

import torch

# Add project root to path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataloaders import get_replicated_dataloader
from models import CNNModel, CNNTrainer
from utils import generate_config, save_htm_model, get_htm_model

warnings.filterwarnings('ignore', category=UserWarning, module='htm')

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SEP = '=' * 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_samples_count(data_path):
    """Return number of images found in data_path."""
    files = [
        f for f in os.listdir(data_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    n = len(files)
    if n == 0:
        raise ValueError(f"No images found in {data_path}")
    return n


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='HTM + CNN training runner')
    parser.add_argument('--exp-number',      type=int,   default=13,           help='Experiment number (default: 12)')
    parser.add_argument('--image-size',      type=int,   default=128,          help='Input image size (default: 64)')
    parser.add_argument('--dataset-name',    type=str,   default='celeba_stripes', help='Dataset folder name under data/ (default: celeba_stripes)')
    parser.add_argument('--saved-htm-epoch', type=int,   default=15,           help='Epoch to resume HTM training from (default: 0)')
    parser.add_argument('--htm-train-until', type=int,   default=20,           help='Train HTM up to this epoch (default: 30)')
    parser.add_argument('--cnn-epochs',      type=int,   default=200,          help='Number of CNN training epochs (default: 200)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(42)

    # --- Config ---
    save_path = f'e{args.exp_number}_{args.dataset_name}_s{args.image_size}'
    data_path = os.path.join(PROJECT_ROOT, 'data', args.dataset_name)
    config = generate_config(
        image_size=args.image_size,
        base_enc=4,
        split_factor=4,
        channels=1,
        data_path=data_path,
        cnn={"inner_dim": 64, "learning_rate": 0.001},
        training={
            "batch_size": 256,
            "total_size": 9600,
            "patience": 10,
        },
        save_path=save_path,
    )

    config_path = os.path.join(PROJECT_ROOT, 'configs', save_path + '.json')
    config.save(path=config_path)
    print(f"Config saved to: {config_path}")

    # --- Dataloader ---
    dataloader = get_replicated_dataloader(
        dataset_name='celeba',
        sample_indices=list(range(get_samples_count(data_path))),
        total_size=config.training['total_size'],
        download=True,
        image_size=config.image['size'],
        root_dir=config.runtime['data_path'],
        batch_size=config.training['batch_size'],
    )
    dataloader.dataset.return_idx = True

    # --- Models ---
    results_dir = os.path.join(PROJECT_ROOT, 'results', save_path)
    htm_model_path = os.path.join(results_dir, f'trained_htm_epoch_{args.saved_htm_epoch}.pkl')
    htm_model = get_htm_model(htm_model_path, config)

    model = CNNModel(config=config)
    trainer = CNNTrainer(model=model, htm_model=htm_model, config=config)

    # --- HTM Training ---
    print(f"\n{SEP}")
    print(f"Training HTM from epoch {args.saved_htm_epoch} to {args.htm_train_until}")
    htm_model.learning_enabled = True
    last_saved_htm = args.saved_htm_epoch

    try:
        for i in range(args.saved_htm_epoch, args.htm_train_until):
            model_name = f'trained_htm_epoch_{i + 1}'
            print(f"{SEP}\nTrain HTM: [{i} → {i + 1}]")

            trainer.train(
                dataloader=dataloader,
                epochs=[i, i + 1],
                save_path=config.runtime['save_path'],
                idx_htm_only=i + 1,
            )
            save_htm_model(htm_model, results_dir, model_name)
            last_saved_htm = i + 1
    except KeyboardInterrupt:
        print(f"\nHTM training interrupted. Last saved epoch: {last_saved_htm}")
        try:
            answer = input("Continue to CNN training? [y/N] ").strip().lower()
        except EOFError:
            answer = 'n'  # non-interactive (nohup/background): skip CNN
        if answer != 'y':
            print("Exiting.")
            return
    except Exception as e:
        print(f"\nHTM training failed at epoch {last_saved_htm}: {e}")
        raise

    print(f"HTM training complete. Last saved epoch: {last_saved_htm}")

    # --- CNN Training ---
    print(f"\n{SEP}")
    print(f"Training CNN for {args.cnn_epochs} epochs")
    htm_model.learning_enabled = False
    try:
        trainer.train(
            dataloader=dataloader,
            epochs=[0, args.cnn_epochs],
            save_path=config.runtime['save_path'] + '_wo',
        )
    except KeyboardInterrupt:
        print("\nCNN training interrupted. Goodbye!")

if __name__ == '__main__':
    main()
