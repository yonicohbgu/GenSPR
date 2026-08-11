import torch
import numpy as np
import os
import pickle
import hashlib
from typing import List, Dict, Optional, TYPE_CHECKING
from htm.encoders.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htm.algorithms import SpatialPooler
from htm.bindings.sdr import SDR
import time

if TYPE_CHECKING:
    from utils.config_manager import ConfigManager



class HTMModel:
    def __init__(self,
                 config: "ConfigManager",
                 mask_ratio: Optional[float] = None,
                 dataset_size: Optional[int] = None,
                 device: str = "cpu",
                 use_cache: bool = True,
                 cache_dir: str = "cache/htm"):
        """
        Initialize HTM preprocessor with multilayer spatial pooling.

        Args:
            config: ConfigManager instance with all configuration
            mask_ratio: Ratio of pixels to mask (0.0-1.0), None for no masking
            dataset_size: Size of the dataset for generating constant masks
            device: Device to use for tensor operations ("cpu", "cuda", "cuda:0", etc.)
            use_cache: Enable caching of spatial poolers
            cache_dir: Cache directory relative to project root
        """
        start_init = time.time()

        # Store ConfigManager reference
        self._config_manager = config

        # Extract values from config
        self.input_size = config.image["size"]
        self.color_channels = config.image["channels"]
        self.split_factor = config.architecture.get("split_factor", 4)
        self.mask_ratio = mask_ratio
        self.dataset_size = dataset_size
        self.device = device
        self.print_example = True

        # Store raw config dict for internal use (backward compatibility)
        self.config = config.raw

        # Initialize encoder from config
        encoder_config = config.encoder
        self.base_enc = encoder_config["base_enc"]
        params = ScalarEncoderParameters()
        params.size = self.base_enc * self.base_enc
        params.activeBits = encoder_config["active_bits"]
        params.minimum = encoder_config["minimum"]
        params.maximum = encoder_config["maximum"]
        params.clipInput = encoder_config["clip_input"]
        self.encoder = ScalarEncoder(params)

        # Prepare spatial pooler parameters directly from config.
        # Config already has correct inputDimensions injected by ConfigManager
        self.spatial_pooler_params_L0 = self._prepare_sp_params(
            config.spatial_pooler["L0"]
        )
        self.spatial_pooler_params_L1 = self._prepare_sp_params(
            config.spatial_pooler["L1"]
        )

        # Get architecture settings
        self.num_layers = config.architecture["num_layers"]

        # Determine quarter size (part size) from derived dimensions
        self.quarter_size = config.dims.part_size

        # Initialize HTM components
        self.spatial_poolers = []  # Will be [layer][sp_index]
        self.encodings = []  # Will be [sp_index] for first layer only
        self.columns = []  # Will be [layer][sp_index]

        # Initialize spatial poolers - try cache first
        cached = False
        if use_cache:
            cached = self._load_from_cache(cache_dir)

        if not cached:
            # Cache not found or disabled - create from scratch
            self._initialize_spatial_poolers()

            # Save to cache for next time
            if use_cache:
                self.save_cache(cache_dir)

        # Initialize encodings and columns
        self._initialize_encodings_and_columns()

        # Generate constant masks if needed
        self.constant_masks = None
        if mask_ratio is not None and dataset_size is not None:
            self._generate_constant_masks()

        # Initialize epoch counter
        self.epoch_counter = 0
        cache_status = "from cache" if cached else "from scratch"
        print(f"{time.time() - start_init:.2f}s\tHTM initialized {cache_status}")

    def _prepare_sp_params(self, config_params: Dict) -> Dict:
        """
        Prepare spatial pooler parameters from config.

        Args:
            config_params: Configuration dictionary for this layer
                          (inputDimensions already computed by ConfigManager)

        Returns:
            Dictionary of parameters is ready for SpatialPooler initialization
        """
        params = config_params.copy()

        # Convert list to tuple for dimensions (required by HTM library)
        if isinstance(params.get('inputDimensions'), list):
            params['inputDimensions'] = tuple(params['inputDimensions'])
        if isinstance(params.get('columnDimensions'), list):
            params['columnDimensions'] = tuple(params['columnDimensions'])

        # Ensure integer values for parameters that require them
        params['stimulusThreshold'] = int(round(params['stimulusThreshold']))
        params['dutyCyclePeriod'] = int(round(params['dutyCyclePeriod']))

        return params

    def _get_cache_key(self) -> str:
        """Generate unique cache key from structural configuration"""
        cache_params = {
            'input_size': self.input_size,
            'color_channels': self.color_channels,
            'split_factor': self.split_factor,
            'num_layers': self.num_layers,
            'sp_L0_params': self.spatial_pooler_params_L0,
            'sp_L1_params': self.spatial_pooler_params_L1
        }

        param_str = str(sorted(cache_params.items()))
        cache_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
        return f"htm_cache_{cache_hash}"

    def save_cache(self, cache_dir: str = "cache/htm") -> None:
        """Save spatial poolers to cache for faster initialization"""
        # Ensure cache_dir is relative to project root, not models/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_cache_dir = os.path.join(project_root, cache_dir)
        os.makedirs(full_cache_dir, exist_ok=True)

        cache_key = self._get_cache_key()
        cache_path = os.path.join(full_cache_dir, f"{cache_key}.pkl")

        cache_data = {
            'spatial_poolers': self.spatial_poolers,
            'cache_params': {
                'input_size': self.input_size,
                'color_channels': self.color_channels,
                'split_factor': self.split_factor,
                'num_layers': self.num_layers,
            }
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"HTM cache saved to {cache_path}")

    def _load_from_cache(self, cache_dir: str = "cache/htm") -> bool:
        """Try to load spatial poolers from cache. Returns True if successful."""
        # Ensure cache_dir is relative to project root, not models/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_cache_dir = os.path.join(project_root, cache_dir)

        cache_key = self._get_cache_key()
        cache_path = os.path.join(full_cache_dir, f"{cache_key}.pkl")

        if not os.path.exists(cache_path):
            return False

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Restore spatial poolers
            self.spatial_poolers = cache_data['spatial_poolers']

            print(f"HTM spatial poolers loaded from cache: {cache_path}")
            return True
        except Exception as e:
            print(f"Failed to load cache: {e}. Initializing from scratch.")
            return False

    @property
    def learning_enabled(self) -> bool:
        """Whether learning is currently enabled for spatial poolers"""
        return self.config.get("global", {}).get("learning_enabled", True)

    @learning_enabled.setter
    def learning_enabled(self, value: bool):
        """Set learning enabled state for all spatial poolers"""
        if "global" not in self.config:
            self.config["global"] = {}
        self.config["global"]["learning_enabled"] = value

    @property
    def merge_method(self) -> str:
        """Current method used for merging representations"""
        return self.config.get("global", {}).get("merge_method", "concatenate")

    @property
    def use_skip_connections(self) -> bool:
        """Whether skip connections are used when combining layer outputs"""
        return self.config.get("global", {}).get("use_skip_connections", False)

    @property
    def layer_weights(self) -> List[float]:
        """Weights applied to each layer's output when using skip connections"""
        return self.config.get("global", {}).get("layer_weights", None)

    @property
    def spatial_pooler_configs(self) -> List[Dict]:
        """Configuration for each layer's spatial poolers"""
        return self.config.get("layers", [])

    def _initialize_spatial_poolers(self):
        """Initialize the hierarchical structure of spatial poolers"""
        self.spatial_poolers = []

        # For each layer
        for layer in range(self.num_layers):
            layer_poolers = []

            # Number of spatial poolers in this layer
            num_poolers = 2 ** (self.split_factor - layer*2) if layer < self.num_layers - 1 else 1

            # Create spatial poolers for this layer
            for i in range(num_poolers):
                # Create the spatial pooler
                sp = SpatialPooler(**self.spatial_pooler_params_L0) if layer == 0 else SpatialPooler(**self.spatial_pooler_params_L1)
                layer_poolers.append(sp)
            self.spatial_poolers.append(layer_poolers)

    def _initialize_encodings_and_columns(self):
        """Initialize encodings for first layer and columns for all layers"""
        # Initialize columns for each layer
        self.columns = []
        for layer in range(self.num_layers):
            layer_columns = []
            num_poolers = 2 ** (self.split_factor - layer*2) if layer < self.num_layers - 1 else 1

            for i in range(num_poolers):
                # Get dimensions from spatial pooler
                column_sdr = SDR(self.spatial_pooler_params_L0['columnDimensions']) if layer==0 else SDR(self.spatial_pooler_params_L1['columnDimensions'])
                layer_columns.append(column_sdr)

            self.columns.append(layer_columns)

        # Initialize encodings for the first layer only
        self.encodings = []
        for i in range(2 ** self.split_factor):  # Number of parts in first layer
            # Get input dimensions for this encoding
            input_dims = SDR(self.spatial_pooler_params_L0['inputDimensions'])
            encoding_sdr = SDR(input_dims)
            self.encodings.append(encoding_sdr)

    def _generate_constant_masks(self):
        """Generate and store constant masks for the entire dataset"""
        h = w = self.input_size

        # Generate random masks for all samples
        self.constant_masks = (torch.rand(
            self.dataset_size, 1, h, w, device=self.device
        ) < self.mask_ratio).float()  # Convert to float for multiplication

    def _generate_various_masks(self):
        """Generate and store constant masks for the entire dataset"""
        h = w = self.input_size

        # Generate random ratios between self.mask_ratio and 0.5 for each sample
        random_ratio = torch.rand(1, device=self.device) * (0.5 - self.mask_ratio) + self.mask_ratio

        # Apply the per-sample random ratios
        return (torch.rand(1, h, w, device=self.device) < random_ratio).float()



## _______________________________________________________________________________________________________________##

    def process_batch(self, batch: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of images through the multilayer HTM pipeline

        Args:
            batch: Batch of images (B, C, H, W)
            batch_indices: Indices of the samples in the dataset

        Returns:
            Processed batch (B, output_dimension)
        """
        # Move tensors to the desired device
        batch = batch.to(self.device)
        batch_indices = batch_indices.to(self.device)
        batch_size = batch.size(0)
        final_sdr_dim = self._config_manager.dims.final_sdr_dim
        processed_batch = torch.zeros(batch_size, final_sdr_dim[0], final_sdr_dim[1],
                                       dtype=torch.float32, device=self.device)


        # Process each image in the batch sequentially
        for i in range(batch_size):
            img = batch[i]
            idx = batch_indices[i].item()

            # Apply masking if enabled
            if self.mask_ratio is not None:
                img = self._apply_mask_to_image(img, idx)

            # Process this image through the HTM pipeline
            processed_img = self._process_image(img, learn=self.learning_enabled)
            processed_batch[i] = torch.from_numpy(processed_img).to(self.device)

        # Stack results and convert to tensor
        return processed_batch


    def _apply_mask_to_image(self, img: torch.Tensor, idx: int) -> torch.Tensor:
        """Apply a pre-generated mask to a single image"""
        if self.constant_masks is None:
            return img
        # Get mask for this index
        mask = self.constant_masks[idx] if idx >=0 else  self._generate_various_masks()
        # Make sure mask is on the same device as image
        mask = mask.to(img.device)
        # Handle different channel dimensions
        if mask.shape[0] != img.shape[0]:
            # Expand mask to match image channels
            mask = mask.expand(img.shape[0], -1, -1)
        # Apply mask (element-wise multiplication)
        return img * mask

    def _split_image(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Split image into parts according to split_factor"""
        # Determine splits per dimension
        splits_per_dim = 2 ** (self.split_factor // 2)

        # Get image dimensions
        if len(img.shape) == 2:  # Grayscale without channel dimension
            h, w = img.shape
            img = img.unsqueeze(0)  # Add channel dimension
        elif len(img.shape) == 3:  # With channel dimension
            c, h, w = img.shape
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        # Calculate part size
        part_h = h // splits_per_dim
        part_w = w // splits_per_dim

        # Collect all parts
        parts = []
        for i in range(splits_per_dim):
            for j in range(splits_per_dim):
                # Extract this part
                part = img[:, i * part_h:(i + 1) * part_h, j * part_w:(j + 1) * part_w]
                parts.append(part)

        return parts
    # Helper methods for internal processing



    def _process_image(self, img: torch.Tensor, learn: bool) -> np.ndarray:
        """Process a single image through all layers"""
        # Split image into parts
        parts = self._split_image(img)

        preprocessed_parts = []
        for part in parts:
            # Flatten and convert to numpy
            part_np = part.cpu().numpy()

            # Scale to 0-255 range if not already
            if part_np.max() <= 1.0:
                part_np = (part_np * 255).astype(np.uint8)

            preprocessed_parts.append(part_np)
        # Process through all layers
        layer_outputs = []

        # Process base layer
        part_size = self.quarter_size
        base_outputs = []
        for i, part in enumerate(preprocessed_parts):
            # Set up encoding for this part
            # empty matrix for encoding each pixel:
            masked_part_encoded = np.zeros((part_size, part_size, self.base_enc, self.base_enc))
            tile_size = 2
            for j in range(0, masked_part_encoded.shape[0], tile_size):
                for k in range(0, masked_part_encoded.shape[1], tile_size):
                    cell = masked_part_encoded[j,k]
                    if part[0,j,k] != 0:
                        pixel_sdr = SDR(self.encoder.size)
                        self.encoder.encode(part[0,j,k], pixel_sdr)
                        cell[:] = np.reshape(pixel_sdr.dense, (self.base_enc, self.base_enc))

            self.encodings[i].dense = masked_part_encoded.reshape(part_size, part_size, self.base_enc, self.base_enc).swapaxes(1, 2).reshape(part_size*self.base_enc, part_size*self.base_enc)

            # Process through spatial pooler
            self.spatial_poolers[0][i].compute(
                self.encodings[i],
                learn,
                self.columns[0][i]
            )
            # Store output
            base_outputs.append(np.reshape(self.columns[0][i].dense, self.spatial_pooler_params_L0['columnDimensions']))

        layer_outputs.append(base_outputs)

        # Process higher layers
        for layer in range(1, self.num_layers):
            prev_outputs = layer_outputs[-1]  # outputs from previous layer
            num_prev = len(prev_outputs)

            # Figure out grid size from number of tiles
            grid_size = int(np.sqrt(num_prev)), int(np.sqrt(num_prev))

            # Merge tiles in spatial order (2x2 grouping)
            merged_tiles = self.merge_blocks(prev_outputs, grid_size, block_size=2)

            layer_result = []
            for i, merged in enumerate(merged_tiles):
                input_dims = self.spatial_pooler_params_L1['inputDimensions']
                input_sdr = SDR(input_dims)

                if self.merge_method == "concatenate":
                    input_sdr.dense = merged
                else:
                    if merged.shape != input_dims:
                        merged = np.resize(merged, input_dims)
                    input_sdr.dense = merged

                self.spatial_poolers[layer][i].compute(
                    input_sdr,
                    learn,
                    self.columns[layer][i]
                )
                layer_result.append(self.columns[layer][i].dense.copy().astype(np.float32))

            layer_outputs.append(layer_result)

        # Final layer will have only one output
        final_output = layer_outputs[-1][0]

        return final_output


    def merge_blocks(self, tiles, grid_size, block_size=2):
        """
        Merge tiles in a grid into larger blocks.

        Parameters
        ----------
        tiles : list of 2D arrays
            Each tile is assumed to have the same shape.
        grid_size : tuple (rows, cols)
            How the tiles are arranged (e.g., (4,4) for 16 tiles).
        block_size : int
            How many tiles per row/col to merge (default=2).

        Returns
        -------
        merged_tiles : list of 2D arrays
            Each array is a merged block of size (block_size * h, block_size * w).
            The number of merged tiles will be (rows/block_size) * (cols/block_size).
        """
        rows, cols = grid_size
        tile_h, tile_w = tiles[0].shape
        merged_tiles = []

        for r in range(0, rows, block_size):
            for c in range(0, cols, block_size):
                block = [
                    tiles[(r+i)*cols + (c+j)]
                    for i in range(block_size)
                    for j in range(block_size)
                ]
                merged = []
                # merge row by row
                for i in range(block_size):
                    merged_row = np.hstack(block[i*block_size:(i+1)*block_size])
                    merged.append(merged_row)
                merged = np.vstack(merged)
                merged_tiles.append(merged)

        return merged_tiles


    def _adjust_learning_parameters(self):
        """Adjust learning parameters according to current epoch and schedule"""
        # Check if staged learning is enabled
        if self.config.get("global", {}).get("staged_learning", False):
            # Get the number of initial stability epochs
            initial_stability = self.config.get("global", {}).get("initial_stability_epochs", 5)

            # For each layer, enable/disable learning based on epoch
            for layer in range(self.num_layers):
                # This layer starts learning after (layer * initial_stability) epochs
                should_learn = self.epoch_counter >= (layer * initial_stability)

                for sp in self.spatial_poolers[layer]:
                    # Set learning mode for this spatial pooler
                    sp.setLearningMode(should_learn)

                if self.epoch_counter == (layer * initial_stability):
                    print(f"Epoch {self.epoch_counter}: Enabling learning for layer {layer}")

        # Check for scheduled parameter adjustments
        learning_schedule = self.config.get("global", {}).get("learning_schedule", {})

        # Convert string keys to integers (JSON stores keys as strings)
        schedule = {int(k): v for k, v in learning_schedule.items() if k.isdigit()}

        # Check if current epoch is in the schedule
        if self.epoch_counter in schedule:
            adjustments = schedule[self.epoch_counter]
            self._apply_parameter_adjustments(adjustments)
            print(f"Epoch {self.epoch_counter}: Applied parameter adjustments - {adjustments}")


    def _extract_permanence_values(self, sp: SpatialPooler) -> Dict:
        """Extract permanence values from a spatial pooler for saving state"""
        # Note: The exact implementation depends on HTM library's API
        # This is a simplified example; adjust based on actual HTM library
        try:
            # Method 1: Using getters if available
            num_columns = sp.getNumColumns()
            num_inputs = sp.getNumInputs()
            permanences = np.zeros((num_columns, num_inputs), dtype=np.float32)

            # Extract permanences for each column
            for col in range(num_columns):
                column_permanences = np.zeros(num_inputs, dtype=np.float32)
                sp.getPermanence(col, column_permanences)
                permanences[col] = column_permanences

            return {
                "permanences": permanences,
                "dimensions": {
                    "input": sp.getInputDimensions(),
                    "column": sp.getColumnDimensions()
                }
            }
        except AttributeError:
            # Method 2: Accessing internal attributes if getters not available
            # This is less robust but might be necessary depending on HTM implementation
            return {
                "permanences": sp._permanences.toDense() if hasattr(sp, "_permanences") else None,
                "connectedSynapses": sp._connectedSynapses.toDense() if hasattr(sp, "_connectedSynapses") else None,
                "dimensions": {
                    "input": sp.getInputDimensions(),
                    "column": sp.getColumnDimensions()
                }
            }

    def _apply_permanence_values(self, sp: SpatialPooler, state: Dict):
        """Apply saved permanence values to a spatial pooler"""
        # Note: The exact implementation depends on HTM library's API
        try:
            # Method 1: Using setters if available
            if "permanences" in state and state["permanences"] is not None:
                permanences = state["permanences"]
                for col in range(len(permanences)):
                    sp.setPermanence(col, permanences[col])
        except AttributeError:
            # Method 2: Setting internal attributes if setters not available
            # This is less robust but might be necessary depending on HTM implementation
            if "permanences" in state and state["permanences"] is not None and hasattr(sp, "_permanences"):
                sp._permanences.fromDense(state["permanences"])

            if "connectedSynapses" in state and state["connectedSynapses"] is not None and hasattr(sp,
                                                                                                   "_connectedSynapses"):
                sp._connectedSynapses.fromDense(state["connectedSynapses"])

    def _apply_parameter_adjustments(self, adjustments: Dict):
        """Apply parameter adjustments to spatial poolers"""
        # For each parameter in the adjustments
        for param_name, param_value in adjustments.items():
            # Apply to all spatial poolers
            for layer in self.spatial_poolers:
                for sp in layer:
                    # Set parameter using appropriate method
                    if param_name == "synPermActiveInc":
                        sp.setSynPermActiveInc(param_value)
                    elif param_name == "synPermInactiveDec":
                        sp.setSynPermInactiveDec(param_value)
                    elif param_name == "localAreaDensity":
                        sp.setLocalAreaDensity(param_value)
                    # Add other parameters as needed




    # Private methods
    # Note: _load_htm_config has been removed - now using unified load_htm_config() function