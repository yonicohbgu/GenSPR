"""
Centralized configuration management for HTM+CNN pipeline.

Provides:
- ConfigManager: Main class for loading, validating, and accessing config
- DerivedDimensions: Immutable dataclass with auto-calculated dimensions
- generate_config(): Function to create configs with computed values
- ConfigValidationError: Exception for validation failures
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


@dataclass(frozen=True)
class DerivedDimensions:
    """
    Immutable dataclass containing all derived/computed dimensions.
    These are calculated from base config values and used throughout the pipeline.
    """
    # Image partitioning
    splits_per_dim: int      # Number of splits per dimension (e.g., 4 for 16 parts)
    part_size: int           # Size of each image part (e.g., 16 for 64/4)
    num_parts: int           # Total number of parts (e.g., 16)

    # HTM L0 dimensions
    L0_input_dim: Tuple[int, int]   # Input to L0 spatial pooler (e.g., (64, 64))
    L0_output_dim: Tuple[int, int]  # Output from L0 (columnDimensions)
    num_L0_poolers: int             # Number of L0 spatial poolers

    # HTM L1 dimensions
    L1_input_dim: Tuple[int, int]   # Input to L1 spatial pooler (e.g., (64, 64))
    L1_output_dim: Tuple[int, int]  # Output from L1 (columnDimensions)
    num_L1_poolers: int             # Number of L1 spatial poolers

    # CNN dimensions
    cnn_input_flat: int       # Flattened input to CNN (e.g., 1024 for 32x32)
    cnn_dense_output: int     # Dense layer output (e.g., 4096 for 64x64)
    cnn_output_shape: Tuple[int, int, int]  # Final output shape (C, H, W)

    # Final SDR
    final_sdr_dim: Tuple[int, int]  # Final SDR dimensions from HTM


def _calculate_dimensions(image_size: int, channels: int, base_enc: int,
                          split_factor: int, L0_col_dims: Tuple[int, int],
                          L1_col_dims: Tuple[int, int]) -> DerivedDimensions:
    """
    Calculate all derived dimensions from base config values.

    Args:
        image_size: Base image dimension (must be power of 2)
        channels: Number of image channels
        base_enc: Encoder grid size (e.g., 4 for 4x4=16 bits)
        split_factor: Power for image partitioning
        L0_col_dims: L0 spatial pooler column dimensions
        L1_col_dims: L1 spatial pooler column dimensions

    Returns:
        DerivedDimensions with all computed values
    """
    # Image partitioning
    splits_per_dim = 2 ** (split_factor // 2)
    part_size = image_size // splits_per_dim
    num_parts = splits_per_dim ** 2

    # HTM L0 dimensions
    # Each part is part_size x part_size, encoded with base_enc x base_enc per pixel
    # Formula: L0_input = part_size * base_enc (e.g., 16 * 4 = 64)
    L0_input_dim = (part_size * base_enc, part_size * base_enc)
    L0_output_dim = tuple(L0_col_dims)
    num_L0_poolers = num_parts

    # HTM L1 dimensions (chained dependency from L0)
    # Each layer's input = 2 × previous layer's output (due to 2×2 merge)
    # L1 receives merged 2x2 blocks of L0 outputs, so:
    # L1_input_dim = 2 * L0_output_dim = 2 * L0_col_dims
    # Layer 2 uses L1 params, so L2_input = 2 * L1_output = 2 * L1_col_dims
    # This only works correctly if L0_col_dims == L1_col_dims (validated separately)
    L1_input_dim = (2 * L0_col_dims[0], 2 * L0_col_dims[1])
    L1_output_dim = tuple(L1_col_dims)
    num_L1_poolers = num_parts // 4  # 2x2 merge reduces count by 4

    # CNN dimensions
    final_sdr_dim = tuple(L1_col_dims)
    cnn_input_flat = final_sdr_dim[0] * final_sdr_dim[1]
    cnn_dense_output = channels * image_size * image_size
    cnn_output_shape = (channels, image_size, image_size)

    return DerivedDimensions(
        splits_per_dim=splits_per_dim,
        part_size=part_size,
        num_parts=num_parts,
        L0_input_dim=L0_input_dim,
        L0_output_dim=L0_output_dim,
        num_L0_poolers=num_L0_poolers,
        L1_input_dim=L1_input_dim,
        L1_output_dim=L1_output_dim,
        num_L1_poolers=num_L1_poolers,
        cnn_input_flat=cnn_input_flat,
        cnn_dense_output=cnn_dense_output,
        cnn_output_shape=cnn_output_shape,
        final_sdr_dim=final_sdr_dim,
    )


class ConfigManager:
    """
    Central configuration manager for HTM+CNN pipeline.

    Loads config from JSON, calculates derived dimensions, and validates
    consistency. Provides clean access to all config sections.

    Usage:
        config = ConfigManager("configs/htm_default_config.json")
        # or
        config = ConfigManager(config_dict)

        # Access sections
        sp_params = config.spatial_pooler["L0"]
        inner_dim = config.cnn["inner_dim"]

        # Access derived dimensions
        input_flat = config.dims.cnn_input_flat
    """

    def __init__(self, config: Union[str, Dict[str, Any]]):
        """
        Initialize ConfigManager.

        Args:
            config: Path to JSON config file, or config dict directly

        Raises:
            ConfigValidationError: If config is invalid
            FileNotFoundError: If config file doesn't exist
        """
        if isinstance(config, str):
            self._config = self._load_from_file(config)
            self._config_path = config
        else:
            self._config = config
            self._config_path = None

        # Inject computed inputDimensions into spatial pooler configs
        self._inject_computed_dimensions()

        # Calculate derived dimensions
        self._dims = self._calculate_dims()

        # Validate immediately (fail-fast)
        self.validate()

    def _load_from_file(self, path: str) -> Dict[str, Any]:
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def _inject_computed_dimensions(self) -> None:
        """Inject computed inputDimensions into spatial pooler configs."""
        image_size = self._config["image"]["size"]
        split_factor = self._config["architecture"].get("split_factor", 4)
        base_enc = self._config["encoder"]["base_enc"]

        splits_per_dim = 2 ** (split_factor // 2)
        part_size = image_size // splits_per_dim

        # L0 input dimensions
        L0_input = (part_size * base_enc, part_size * base_enc)
        self._config["spatial_pooler"]["L0"]["inputDimensions"] = list(L0_input)

        # L1 input dimensions (2x2 merge of L0 outputs)
        L0_col_dims = self._config["spatial_pooler"]["L0"]["columnDimensions"]
        L1_input = (2 * L0_col_dims[0], 2 * L0_col_dims[1])
        self._config["spatial_pooler"]["L1"]["inputDimensions"] = list(L1_input)

    def _calculate_dims(self) -> DerivedDimensions:
        """Calculate derived dimensions from config."""
        return _calculate_dimensions(
            image_size=self._config["image"]["size"],
            channels=self._config["image"]["channels"],
            base_enc=self._config["encoder"]["base_enc"],
            split_factor=self._config["architecture"].get("split_factor", 4),
            L0_col_dims=tuple(self._config["spatial_pooler"]["L0"]["columnDimensions"]),
            L1_col_dims=tuple(self._config["spatial_pooler"]["L1"]["columnDimensions"]),
        )

    def validate(self) -> None:
        """
        Validate config consistency. Raises ConfigValidationError if invalid.

        Checks:
        - Image size is power of 2
        - Split factor produces valid partitioning
        - L0/L1 inputDimensions match computed values
        - CNN input matches HTM output
        - Loss weights sum to 1.0
        """
        errors = []

        # Check image size is power of 2
        image_size = self._config["image"]["size"]
        if image_size <= 0 or (image_size & (image_size - 1)) != 0:
            errors.append(f"image.size must be power of 2, got {image_size}")

        # Check split factor produces valid partitioning
        split_factor = self._config["architecture"].get("split_factor", 4)
        splits_per_dim = 2 ** (split_factor // 2)
        if image_size % splits_per_dim != 0:
            errors.append(
                f"split_factor={split_factor} produces {splits_per_dim} splits, "
                f"but image_size={image_size} is not divisible"
            )

        # Check L0 inputDimensions match computed
        L0_input = self._config["spatial_pooler"]["L0"]["inputDimensions"]
        expected_L0 = list(self._dims.L0_input_dim)
        if L0_input != expected_L0:
            errors.append(
                f"L0 inputDimensions mismatch: config has {L0_input}, "
                f"computed is {expected_L0}"
            )

        # Check L1 inputDimensions match computed
        L1_input = self._config["spatial_pooler"]["L1"]["inputDimensions"]
        expected_L1 = list(self._dims.L1_input_dim)
        if L1_input != expected_L1:
            errors.append(
                f"L1 inputDimensions mismatch: config has {L1_input}, "
                f"computed is {expected_L1}"
            )

        # Check L0_col_dims == L1_col_dims (required for layer 2 to use L1 params correctly)
        # The "L1" config is used for both layer 1 and layer 2, and each layer's input
        # depends on the previous layer's output dimensions.
        L0_col_dims = self._config["spatial_pooler"]["L0"]["columnDimensions"]
        L1_col_dims = self._config["spatial_pooler"]["L1"]["columnDimensions"]
        if L0_col_dims != L1_col_dims:
            errors.append(
                f"L0 and L1 columnDimensions must match (L1 config is used for both "
                f"layer 1 and layer 2). L0={L0_col_dims}, L1={L1_col_dims}"
            )

        # Check loss weights sum to 1.0
        if "cnn" in self._config and "loss_weights" in self._config["cnn"]:
            weights = self._config["cnn"]["loss_weights"]
            total = sum(weights.values())
            if abs(total - 1.0) > 0.001:
                errors.append(f"Loss weights must sum to 1.0, got {total}")

        if errors:
            raise ConfigValidationError(
                "Config validation failed:\n  - " + "\n  - ".join(errors)
            )

    # -------------------------------------------------------------------------
    # Property accessors for config sections
    # -------------------------------------------------------------------------

    @property
    def dims(self) -> DerivedDimensions:
        """Access derived dimensions."""
        return self._dims

    @property
    def image(self) -> Dict[str, Any]:
        """Access image config section."""
        return self._config["image"]

    @property
    def encoder(self) -> Dict[str, Any]:
        """Access encoder config section."""
        return self._config["encoder"]

    @property
    def architecture(self) -> Dict[str, Any]:
        """Access architecture config section."""
        return self._config["architecture"]

    @property
    def spatial_pooler(self) -> Dict[str, Dict[str, Any]]:
        """Access spatial pooler config section (L0, L1)."""
        return self._config["spatial_pooler"]

    @property
    def global_config(self) -> Dict[str, Any]:
        """Access global config section."""
        return self._config.get("global", {})

    @property
    def cnn(self) -> Dict[str, Any]:
        """Access CNN config section."""
        return self._config.get("cnn", {})

    @property
    def training(self) -> Dict[str, Any]:
        """Access training config section."""
        return self._config.get("training", {})

    @property
    def runtime(self) -> Dict[str, Any]:
        """Access runtime config section."""
        return self._config.get("runtime", {})

    @property
    def raw(self) -> Dict[str, Any]:
        """Access raw config dict for full flexibility."""
        return self._config

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=2)
        print(f"Config saved to {path}")

    def __repr__(self) -> str:
        source = self._config_path or "dict"
        return f"ConfigManager(source={source}, image_size={self.image['size']})"


def generate_config(
    image_size: int,
    base_enc: int = 4,
    split_factor: int = 4,
    channels: int = 1,
    save_path: str = "e11_celeba_mixed",
    data_path: str = "data/celeba_mixed",
    device: str = None,
    **overrides
) -> ConfigManager:
    """
    Generate complete config with all derived values calculated and injected.

    Args:
        image_size: Base image dimension (must be power of 2)
        base_enc: Encoder grid size (default 4 = 4x4 = 16 bits)
        split_factor: Power for image partitioning (default 4 = 16 parts)
        channels: Number of image channels (default 1 for grayscale)
        save_path: Path to save results.
        data_path: Relative path to the dataset directory (e.g. "data/celeba_mixed",
                   "data/celeba_vertical"). Matches the output of create_composite_celeba.py.
        device: Compute device; auto-detected (cuda > mps > cpu) if not specified.
        **overrides: Override any config section (e.g., cnn={"inner_dim": 512})

    Returns:
        ConfigManager instance with complete, validated config

    Example:
        config = generate_config(image_size=64)
        config = generate_config(image_size=64, col_dim=64, data_path="data/celeba_vertical")
    """
    import torch
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    # Calculate derived dimensions for inputDimensions
    splits_per_dim = 2 ** (split_factor // 2)
    part_size = image_size // splits_per_dim

    L0_input = part_size * base_enc
    L0_col = L0_input // 2          # L0 and L1 column dims must be equal (validated by ConfigManager)
    L1_input = 2 * L0_col    # auto-derived: L1 input = 2 * L0 output (2x2 merge)
    L1_col = L1_input // 2

    # Build base config
    config = {
        "image": {
            "size": image_size,
            "channels": channels
        },
        "encoder": {
            "base_enc": base_enc,
            "active_bits": 7,
            "minimum": 0,
            "maximum": 255,
            "clip_input": True
        },
        "architecture": {
            "num_layers": 3,
            "split_factor": split_factor
        },
        "spatial_pooler": {
            "L0": {
                "inputDimensions": [L0_input, L0_input],
                "columnDimensions": [L0_col, L0_col],
                "globalInhibition": True,
                "seed": 42,
                "spVerbosity": 1,
                "wrapAround": False,
                "potentialRadius": 30,
                "boostStrength": 7,
                "dutyCyclePeriod": 850,
                "localAreaDensity": 0.1,
                "minPctOverlapDutyCycle": 0.2,
                "potentialPct": 0.2,
                "stimulusThreshold": 6,
                "synPermActiveInc": 0.12,
                "synPermConnected": 0.6,
                "synPermInactiveDec": 0.02
            },
            "L1": {
                "inputDimensions": [L1_input, L1_input],
                "columnDimensions": [L1_col, L1_col],
                "globalInhibition": True,
                "seed": 42,
                "spVerbosity": 1,
                "wrapAround": False,
                "potentialRadius": 16,
                "boostStrength": 5,
                "dutyCyclePeriod": 850,
                "localAreaDensity": 0.15,
                "minPctOverlapDutyCycle": 0.2,
                "potentialPct": 0.2,
                "stimulusThreshold": 6,
                "synPermActiveInc": 0.12,
                "synPermConnected": 0.6,
                "synPermInactiveDec": 0.02
            }
        },
        "global": {
            "random_seed": 42,
            "learning_enabled": True,
            "merge_method": "concatenate",
            "use_skip_connections": False,
            "layer_weights": None,
            "staged_learning": False,
            "initial_stability_epochs": 5,
            "learning_schedule": {}
        },
        "cnn": {
            "inner_dim": 256,
            "conv_filters": [64, 128, 64],
            "conv_kernels": [5, 3, 3, 3],
            "leaky_relu_alpha": 0.2,
            "learning_rate": 0.0001,
            "betas": [0.5, 0.999],
            "loss_weights": {
                "mse": 0.1,
                "ssim": 0.8,
                "tv": 0.1
            },
            "checkpoint_frequency": 2
        },
        "training": {
            "batch_size": 256,
            "total_size": 9600,
            "patience": 10,
            "min_delta": 0.001
        },
        "runtime": {
            "save_path": save_path,
            "data_path": data_path,
            "mask_ratio": 0.15,
            "device": device
        }
    }

    # Apply overrides (deep merge)
    def deep_merge(base: Dict, override: Dict) -> Dict:
        """Recursively merge override into base."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    for section, values in overrides.items():
        if section in config and isinstance(values, dict):
            config[section] = deep_merge(config[section], values)
        else:
            config[section] = values

    return ConfigManager(config)
