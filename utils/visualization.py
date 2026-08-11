import matplotlib.pyplot as plt
import numpy as np
import torch
from htm.bindings.sdr import SDR

base_enc = 4

def run(dataset, htm_model, cnn_model=None, image_idx=0):
    """
    Visualize the HTM pipeline with a single function call.

    Args:
        dataset: CelebADataset instance with return_idx=True
        htm_model: HTMModel instance
        cnn_model: Optional CNNModel instance for reconstruction visualization
        image_idx: Index of image to visualize (default: 0)
    """
    # Get image
    image, idx = dataset[image_idx]

    # Ensure image is on CPU
    image = image.cpu()

    # Convert to grayscale if needed
    if image.shape[0] == 3:
        image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        image = image.unsqueeze(0)

    # Original image for display
    img_np = image.squeeze().numpy()
    img_display = img_np if img_np.max() <= 1.0 else img_np / 255.0

    # Apply masking if enabled
    masked_image = image.clone()
    if htm_model.mask_ratio is not None:
        masked_image = htm_model._apply_mask_to_image(image, idx).cpu()

    masked_np = masked_image.squeeze().numpy()
    masked_display = masked_np if masked_np.max() <= 1.0 else masked_np / 255.0

    # Split image
    parts = htm_model._split_image(masked_image)
    num_parts = len(parts)
    grid_size = int(np.sqrt(num_parts))

    # Create figure
    num_rows = 3 if cnn_model is None else 4
    fig = plt.figure(figsize=(16, 4 * num_rows))

    # Row 1: Input processing
    # 1. Original Image
    plt.subplot(num_rows, 4, 1)
    plt.imshow(img_display, cmap='gray', vmin=0, vmax=1)
    plt.title("Original Grayscale Image")
    plt.axis('off')

    # 2. Masked Image
    plt.subplot(num_rows, 4, 2)
    plt.imshow(masked_display, cmap='gray', vmin=0, vmax=1)
    if htm_model.mask_ratio is not None:
        plt.title(f"Masked Image ({htm_model.mask_ratio*100:.1f}% masked)")
    else:
        plt.title("Image (no masking)")
    plt.axis('off')

    # 3. Split Parts
    plt.subplot(num_rows, 4, 3)
    part_h = parts[0].shape[1]
    part_w = parts[0].shape[2]
    grid_img = np.zeros((grid_size * part_h, grid_size * part_w))

    for i in range(grid_size):
        for j in range(grid_size):
            part_idx = i * grid_size + j
            part_np = parts[part_idx].squeeze().numpy()
            if part_np.max() > 1.0:
                part_np = part_np / 255.0
            grid_img[i*part_h:(i+1)*part_h, j*part_w:(j+1)*part_w] = part_np

    plt.imshow(grid_img, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Split into {num_parts} Parts ({grid_size}×{grid_size})")
    for i in range(1, grid_size):
        plt.axhline(y=i*part_h - 0.5, color='red', linestyle='-', alpha=0.5, linewidth=1)
        plt.axvline(x=i*part_w - 0.5, color='red', linestyle='-', alpha=0.5, linewidth=1)
    plt.axis('off')

    # 4. Mask Pattern
    plt.subplot(num_rows, 4, 4)
    if htm_model.mask_ratio is not None:
        mask = htm_model.constant_masks[idx].squeeze().cpu().numpy()
        plt.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1)
        plt.title("Mask Pattern\n(Green=visible, Red=masked)")
        plt.colorbar(fraction=0.046, pad=0.04)
    else:
        plt.text(0.5, 0.5, 'No Masking Applied', ha='center', va='center', fontsize=12)
        plt.title("Mask Pattern")
    plt.axis('off')

    # Process through HTM layers
    batch = image.unsqueeze(0).to(htm_model.device)
    batch_idx = torch.tensor([idx]).to(htm_model.device)

    img = batch[0]
    if htm_model.mask_ratio is not None:
        img = htm_model._apply_mask_to_image(img, idx)

    parts = htm_model._split_image(img)

    # Preprocess parts
    preprocessed_parts = []
    for part in parts:
        part_np = part.cpu().numpy()
        if part_np.max() <= 1.0:
            part_np = (part_np * 255).astype(np.uint8)
        preprocessed_parts.append(part_np)

    # Process Layer 0
    layer_outputs = []
    base_outputs = []

    for i, part in enumerate(preprocessed_parts):
        masked_part_encoded = np.zeros((16, 16, base_enc, base_enc))
        tile_size = 4
        for j in range(0, masked_part_encoded.shape[0], tile_size):
            for k in range(0, masked_part_encoded.shape[1], tile_size):
                cell = masked_part_encoded[j, k]
                if part[0, j, k] != 0:
                    pixel_sdr = SDR(htm_model.encoder.size)
                    htm_model.encoder.encode(part[0, j, k], pixel_sdr)
                    cell[:] = np.reshape(pixel_sdr.dense, (base_enc, base_enc))

        htm_model.encodings[i].dense = masked_part_encoded.reshape(
            16, 16, base_enc, base_enc
        ).swapaxes(1, 2).reshape(16*base_enc, 16*base_enc)

        htm_model.spatial_poolers[0][i].compute(
            htm_model.encodings[i],
            False,
            htm_model.columns[0][i]
        )

        base_outputs.append(
            np.reshape(htm_model.columns[0][i].dense, (32, 32))
        )

    layer_outputs.append(base_outputs)

    # Row 2: Layer 0
    # 5. Layer 0 Outputs
    plt.subplot(num_rows, 4, 5)
    l0_grid = np.zeros((grid_size * 32, grid_size * 32))
    for i in range(grid_size):
        for j in range(grid_size):
            out_idx = i * grid_size + j
            l0_grid[i*32:(i+1)*32, j*32:(j+1)*32] = base_outputs[out_idx]

    plt.imshow(l0_grid, cmap='viridis')
    plt.title(f"Layer 0 SP Outputs\n({len(base_outputs)} spatial poolers)")
    plt.colorbar(fraction=0.046, pad=0.04)
    for i in range(1, grid_size):
        plt.axhline(y=i*32 - 0.5, color='white', linestyle='-', alpha=0.5, linewidth=1)
        plt.axvline(x=i*32 - 0.5, color='white', linestyle='-', alpha=0.5, linewidth=1)
    plt.axis('off')

    # 6. Layer 0 Sparsity
    plt.subplot(num_rows, 4, 6)
    sparsities = [np.sum(out) / out.size for out in base_outputs]
    plt.bar(range(len(sparsities)), sparsities)
    plt.title("Layer 0 Sparsity per SP")
    plt.xlabel("Spatial Pooler Index")
    plt.ylabel("Active Columns Ratio")
    plt.ylim([0, 0.3])
    plt.grid(True, alpha=0.3)

    # Process Layer 1
    if htm_model.num_layers > 1:
        layer1_outputs = []
        prev_outputs = layer_outputs[-1]
        num_prev = len(prev_outputs)
        grid_size_l1 = int(np.sqrt(num_prev))

        merged_tiles = htm_model.merge_blocks(
            prev_outputs, (grid_size_l1, grid_size_l1), block_size=2
        )

        for i, merged in enumerate(merged_tiles):
            input_sdr = SDR((64, 64))
            input_sdr.dense = merged

            htm_model.spatial_poolers[1][i].compute(
                input_sdr,
                False,
                htm_model.columns[1][i]
            )
            layer1_outputs.append(
                htm_model.columns[1][i].dense.copy().astype(np.float32).reshape(32, 32)
            )

        layer_outputs.append(layer1_outputs)

        # 7. Layer 1 Outputs
        plt.subplot(num_rows, 4, 7)
        if len(layer1_outputs) == 4:
            l1_grid = np.zeros((64, 64))
            l1_grid[:32, :32] = layer1_outputs[0]
            l1_grid[:32, 32:] = layer1_outputs[1]
            l1_grid[32:, :32] = layer1_outputs[2]
            l1_grid[32:, 32:] = layer1_outputs[3]

            plt.imshow(l1_grid, cmap='viridis')
            plt.title(f"Layer 1 SP Outputs\n({len(layer1_outputs)} spatial poolers)")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axhline(y=32 - 0.5, color='white', linestyle='-', alpha=0.5, linewidth=1)
            plt.axvline(x=32 - 0.5, color='white', linestyle='-', alpha=0.5, linewidth=1)
        else:
            plt.imshow(layer1_outputs[0], cmap='viridis')
            plt.title(f"Layer 1 SP Output\n(1 spatial pooler)")
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        # 8. Layer 1 Sparsity
        plt.subplot(num_rows, 4, 8)
        l1_sparsities = [np.sum(out) / out.size for out in layer1_outputs]
        plt.bar(range(len(l1_sparsities)), l1_sparsities)
        plt.title("Layer 1 Sparsity per SP")
        plt.xlabel("Spatial Pooler Index")
        plt.ylabel("Active Columns Ratio")
        plt.ylim([0, 0.5])
        plt.grid(True, alpha=0.3)

    # Process Layer 2
    if htm_model.num_layers > 2:
        prev_outputs = layer_outputs[-1]
        num_prev = len(prev_outputs)
        grid_size_l2 = int(np.sqrt(num_prev))

        merged_tiles = htm_model.merge_blocks(
            prev_outputs, (grid_size_l2, grid_size_l2), block_size=2
        )

        layer2_output = merged_tiles[0]
        input_sdr = SDR((64, 64))
        input_sdr.dense = layer2_output

        htm_model.spatial_poolers[2][0].compute(
            input_sdr,
            False,
            htm_model.columns[2][0]
        )

        final_output = htm_model.columns[2][0].dense.copy().astype(np.float32).reshape(32, 32)

        # Row 3: Layer 2
        # 9. Final Layer Output
        plt.subplot(num_rows, 4, 9)
        plt.imshow(final_output, cmap='viridis')
        plt.title("Layer 2 (Final) SP Output\n(1 spatial pooler)")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        # 10. Final Sparsity
        plt.subplot(num_rows, 4, 10)
        final_sparsity = np.sum(final_output) / final_output.size
        plt.bar([0], [final_sparsity])
        plt.title("Layer 2 Sparsity")
        plt.ylabel("Active Columns Ratio")
        plt.ylim([0, 0.5])
        plt.xticks([])
        plt.grid(True, alpha=0.3)

    # Row 4: CNN Reconstruction (if model provided)
    if cnn_model is not None:
        cnn_model.eval()
        with torch.no_grad():
            htm_output = htm_model.process_batch(batch, batch_idx)
            reconstruction = cnn_model(htm_output).cpu().squeeze().numpy()

            # 13. Reconstruction
            plt.subplot(num_rows, 4, num_rows*4 - 3)
            plt.imshow(reconstruction, cmap='gray', vmin=0, vmax=1)
            plt.title("CNN Reconstruction")
            plt.axis('off')

            # 14. Reconstruction Error
            plt.subplot(num_rows, 4, num_rows*4 - 2)
            diff = np.abs(img_display - reconstruction)
            plt.imshow(diff, cmap='hot')
            plt.title("Reconstruction Error")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')

            # 15. Histogram Comparison
            plt.subplot(num_rows, 4, num_rows*4 - 1)
            plt.hist(img_display.flatten(), bins=50, alpha=0.5, label='Original', density=True)
            plt.hist(reconstruction.flatten(), bins=50, alpha=0.5, label='Reconstructed', density=True)
            plt.title("Pixel Distribution")
            plt.xlabel("Pixel Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 16. Metrics
            mse = np.mean((img_display - reconstruction) ** 2)
            plt.subplot(num_rows, 4, num_rows*4)
            plt.text(0.5, 0.5,
                    f'MSE: {mse:.6f}\n\nPeak Diff: {diff.max():.4f}\n\nMean Diff: {diff.mean():.4f}',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.title("Reconstruction Metrics")
            plt.axis('off')

    plt.tight_layout()
    plt.show()
