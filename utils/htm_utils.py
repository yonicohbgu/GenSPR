import pickle
import os
from models import HTMModel

def save_htm_model(htm_pp, save_path, model_name="trained_htm"):
    """
    Save the trained HTM preprocessor with all learned spatial pooler states.

    Args:
        htm_pp: The HTMModel instance with trained spatial poolers
        save_path: Directory to save the model (e.g., 'results/e7_celeba_10_large')
        model_name: Name for the saved model file
    """
    os.makedirs(save_path, exist_ok=True)

    model_path = os.path.join(save_path, f"{model_name}.pkl")

    # Save the entire HTM preprocessor (includes learned spatial poolers)
    with open(model_path, 'wb') as f:
        pickle.dump(htm_pp, f)

    print(f"HTM model saved to {model_path}")
    return model_path


def load_htm_model(model_path):
    """
    Load a previously trained HTM preprocessor.

    Args:
        model_path: Full path to the saved model file

    Returns:
        HTMModel instance with learned state restored
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        htm_pp = pickle.load(f)

    print(f"HTM model loaded from {model_path}")
    return htm_pp


def get_htm_model(htm_model_path, config):
    if os.path.exists(htm_model_path):
        htm_model = load_htm_model(htm_model_path)
    else:
        print("No existing HTM model found. Initializing new HTM preprocessor...")
        htm_model = HTMModel(
            config=config,
            mask_ratio=config.runtime['mask_ratio'],
            dataset_size=config.training['total_size'],
            device=config.runtime['device'],
            use_cache=True
        )
        print("HTM will be trained during the training process")
    return htm_model