from typing import Sequence, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from pytorch_msssim import SSIM
from htm.bindings.sdr import SDR

if TYPE_CHECKING:
    from utils.config_manager import ConfigManager


tqdm.pandas()


class CNNModel(nn.Module):
    def __init__(self, config: "ConfigManager"):
        """
        Initialize CNN model from config.

        Args:
            config: ConfigManager instance with all dimensions and hyperparameters
        """
        super().__init__()

        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

        # Get dimensions from config
        cnn_input_flat = config.dims.cnn_input_flat
        cnn_dense_output = config.dims.cnn_dense_output
        output_channels = config.image["channels"]
        image_size = config.image["size"]
        inner_dim = config.cnn.get("inner_dim", 256)
        conv_filters = config.cnn.get("conv_filters", [64, 128, 64])
        conv_kernels = config.cnn.get("conv_kernels", [5, 3, 3, 3])
        leaky_relu_alpha = config.cnn.get("leaky_relu_alpha", 0.2)

        # Store for forward pass reshape
        self._image_size = image_size
        self._output_channels = output_channels

        # Initial flatten layer
        self.flatten = nn.Flatten()

        # Dense layers with Batch Normalization
        self.dense_layers = nn.Sequential(
            nn.Linear(cnn_input_flat, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.LeakyReLU(leaky_relu_alpha),
            nn.Linear(inner_dim, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.LeakyReLU(leaky_relu_alpha),
            nn.Linear(inner_dim, cnn_dense_output),
            nn.BatchNorm1d(cnn_dense_output),
            nn.LeakyReLU(leaky_relu_alpha)
        )

        # Convolutional layers built dynamically from config.
        # conv_filters: intermediate out_channels; output layer always uses output_channels.
        # conv_kernels: one per layer including output layer; padding = kernel // 2 preserves size.
        all_out_channels = conv_filters + [output_channels]
        in_ch = output_channels  # after dense→reshape, channels = output_channels
        conv_list = []
        for i, (out_ch, k) in enumerate(zip(all_out_channels, conv_kernels)):
            conv_list.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k,
                                                padding=k // 2, bias=False))
            if i < len(conv_filters):  # intermediate layers: BN + activation
                conv_list.append(nn.BatchNorm2d(out_ch))
                conv_list.append(nn.LeakyReLU(leaky_relu_alpha))
            else:                      # output layer: sigmoid only
                conv_list.append(nn.Sigmoid())
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*conv_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input
        x = self.flatten(x)

        # Pass through dense layers
        x = self.dense_layers(x).to(self.device)

        # Reshape to (batch_size, channels, image_size, image_size)
        x = x.view(-1, self._output_channels, self._image_size, self._image_size)

        # Pass through convolutional layers
        x = self.conv_layers(x)

        return x

    def generate(self, num_samples: int = 1, htm_model=None, rand_mask: bool = True) -> torch.Tensor:
        """
        Generate new images using the full HTM+CNN pipeline.

        Args:
            num_samples (int): Number of images to generate.
            htm_model: HTM preprocessor for the full pipeline. Required for proper generation.
            rand_mask (bool): Whether to randomly mask out the images.
        Returns:
            torch.Tensor: Generated images of shape (num_samples, C, H, W).
        """
        self.eval()  # Set model to evaluation mode

        if htm_model is None:
            raise ValueError("htm_pp is required for generation. The CNN must receive HTM-processed inputs to match training distribution.")

        image_size = self._image_size
        channels = self._output_channels

        with torch.no_grad():
            # Generate random pixel values in range [0, 255], scaled to [0, 1]
            noise = torch.randint(0, 256, (num_samples, channels, image_size, image_size),
                                 device=self.device, dtype=torch.float32) / 255.0

            # Create dummy batch indices for HTM preprocessor
            dummy_indices = torch.arange(num_samples, device=self.device)
            if rand_mask:
                dummy_indices *= -1
            # Process through HTM preprocessor (buckets pixels and applies spatial pooling)
            htm_output = htm_model.process_batch(noise, dummy_indices).to(self.device)

            # Pass HTM activations through CNN to generate final images
            generated_images = self.forward(htm_output)

        return generated_images

    def generate_sdr(self, num_samples: int = 1, htm_model=None, rand_mask: bool = True) -> torch.Tensor:
        """
        Generate new images using the full HTM+CNN pipeline.

        Args:
            num_samples (int): Number of images to generate.
            htm_model: HTM preprocessor for the full pipeline. Required for proper generation.
            rand_mask (bool): Whether to randomly mask out the images.

        Returns:
            torch.Tensor: Generated SDRs of shape (num_samples, sdr_h, sdr_w).
        """

        if htm_model is None:
            raise ValueError("htm_pp is required for generation. The CNN must receive HTM-processed inputs to match training distribution.")

        image_size = self._image_size
        channels = self._output_channels

        # Generate random pixel values in range [0, 255], scaled to [0, 1]
        noise = torch.randint(0, 256, (num_samples, channels, image_size, image_size),
                             device=self.device, dtype=torch.float32) / 255.0

        # Create dummy batch indices for HTM preprocessor
        dummy_indices = torch.arange(num_samples, device=self.device)
        if rand_mask:
            dummy_indices *= -1
        # Process through HTM preprocessor (buckets pixels and applies spatial pooling)
        htm_output = htm_model.process_batch(noise, dummy_indices)

        return htm_output


    def generate_from_SDR(self, htm_output: torch.Tensor=None) -> torch.Tensor:
        """
        Generate new images using the full HTM+CNN pipeline.

        Args:
            num_samples (int): Number of images to generate.
            htm_pp: HTM preprocessor for the full pipeline. Required for proper generation.

        Returns:
            torch.Tensor: Generated images of shape (num_samples, 1, 64, 64).
            :param htm_output:
        """
        self.eval()  # Set model to evaluation mode

        with torch.no_grad():

            # Pass HTM activations through CNN to generate final images
            generated_images = self.forward(htm_output).to(self.device)

        return generated_images

    def plot_images(self, images, num_cols=5, save_path=None):
        """
        Plot images in a grid. Optionally save the figure to disk.

        Args:
            images (torch.Tensor or list): Generated images, shape (N, 1, H, W)
            num_cols (int): Number of columns in the grid
            save_path (str, optional): File path to save the figure. If None, only shows.
        """

        num_images = len(images)
        num_rows = (num_images + num_cols - 1) // num_cols  # Calculate rows needed

        plt.figure(figsize=(num_cols * 2, num_rows * 2))  # Adjust figure size
        for i, img in enumerate(images):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')  # Squeeze to remove single channel
            plt.axis('off')  # Hide axes

        if save_path:
            plt.savefig(save_path)
            print(f"Generated images saved at {save_path}")
            plt.close()
        else:
            plt.show()



class CNNTrainer:
    def __init__(self, model: CNNModel, htm_model: None, config: "ConfigManager"):
        """
        Initialize CNN trainer from config.

        Args:
            model: CNNModel instance
            htm_model: HTMModel instance
            config: ConfigManager instance with all hyperparameters
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.htm_model = htm_model

        # Get hyperparameters from config
        learning_rate = config.cnn.get("learning_rate", 0.0001)
        betas = tuple(config.cnn.get("betas", [0.5, 0.999]))
        loss_weights = config.cnn.get("loss_weights", {"mse": 0.15, "ssim": 0.85})
        channels = config.image["channels"]

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=betas
        )

        # Loss functions - Combined MSE + SSIM + TV loss
        self.mse_loss = nn.MSELoss()
        self.ssim_metric = SSIM(data_range=1.0, size_average=True, channel=channels)

        # Loss combination weights from config
        self.mse_weight = loss_weights.get("mse", 0.15)
        self.ssim_weight = loss_weights.get("ssim", 0.85)
        self.tv_weight = loss_weights.get("tv", 0.0)

        # Training history
        self.losses = []
        self.ssim_scores = []
        self.dataloader = None

    def total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Penalizes sharp intensity changes between adjacent pixels."""
        tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return tv_h + tv_w

    def train_step(self, batch, batch_idx):
        # Move batch to device
        batch = batch.to(self.device)

        # Process images through HTM to get column activations
        htm_columns = self.htm_model.process_batch(batch, batch_idx).to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(htm_columns)

        # Calculate individual losses
        mse = self.mse_loss(outputs, batch)
        ssim_value = self.ssim_metric(outputs, batch)

        # Combined loss for backpropagation
        # Convert SSIM (higher=better) to loss (lower=better) by doing (1 - SSIM)
        ssim_loss = 1.0 - ssim_value
        tv = self.total_variation_loss(outputs)
        combined_loss = self.mse_weight * mse + self.ssim_weight * ssim_loss + self.tv_weight * tv

        # Backward pass and optimize
        combined_loss.backward()
        self.optimizer.step()

        # Return MSE (for monitoring) and SSIM value (for early stopping)
        return mse.item(), ssim_value.item()

    def train(self,
              dataloader,
              epochs: list[int],
              save_path: str = "default",
              patience: int = None,
              min_delta: float = None,
              idx_htm_only: int = 0):
        """
        Train the CNN model with early stopping

        Args:
            dataloader: Input data loader
            epochs: Maximum number of epochs to train
            save_path: Folder name inside 'results/' to save model checkpoints (e.g., "e6_16")
            patience: Number of epochs to wait for improvement before stopping (from config if None)
            min_delta: Minimum change in loss to qualify as an improvement (from config if None)
            idx_htm_only: Minimum htm-only cycles of training
        """
        # Use config defaults if not specified
        if patience is None:
            patience = self.config.training.get("patience", 10)
        if min_delta is None:
            min_delta = self.config.training.get("min_delta", 0.001)
        # Ensure save_path is inside results/ folder
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join("results", save_path)
        save_path = os.path.join(project_root, save_path)
        os.makedirs(save_path, exist_ok=True)

        self.dataloader = dataloader


        print("Starting training...")
        print(f"Total batches per epoch: {len(self.dataloader)}")
        print(f"Dataset size: {len(self.dataloader.dataset)}")
        print(f"Early stopping patience: {patience}")

        # Training history
        self.losses = []
        self.ssim_scores = []
        best_ssim = -float('inf')  # SSIM: higher is better
        best_model_state = None
        best_optimizer_state = None
        patience_counter = 0

        dataloader.dataset.return_idx = True
        for epoch in range(*epochs):
            epoch_losses = []
            epoch_ssim_scores = []

            with tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{epochs[1]}") as pbar:
                for (batch, batch_idx) in pbar:
                    # Training step returns both MSE loss and SSIM
                    if epoch < idx_htm_only:
                        batch = batch.to(self.device)

                        # Process images through HTM to get column activations
                        self.htm_model.process_batch(batch, batch_idx).to(self.device)
                        continue

                    loss, ssim = self.train_step(batch, batch_idx)
                    epoch_losses.append(loss)
                    epoch_ssim_scores.append(ssim)

                    # Update progress bar
                    current_mean_loss = np.mean(epoch_losses)
                    current_mean_ssim = np.mean(epoch_ssim_scores)
                    pbar.set_postfix({
                        'mse': f'{loss:.4f}',
                        'avg_mse': f'{current_mean_loss:.4f}',
                        'ssim': f'{ssim:.4f}',
                        'avg_ssim': f'{current_mean_ssim:.4f}'
                    })
            if epoch < idx_htm_only:
                print(f'Epoch {epoch + 1}: \t Trained HTM Only')
                continue
            # Calculate epoch statistics
            avg_loss = np.mean(epoch_losses)
            avg_ssim = np.mean(epoch_ssim_scores)
            self.losses.append(avg_loss)
            self.ssim_scores.append(avg_ssim)

            # Check for improvement (SSIM: higher is better)
            if avg_ssim > best_ssim + min_delta:
                improvement = avg_ssim - best_ssim
                best_ssim = avg_ssim
                patience_counter = 0
                best_model_state = self.model.state_dict()
                best_optimizer_state = self.optimizer.state_dict()
                print(f"\nEpoch {epoch + 1} - MSE: {avg_loss:.4f}, SSIM: {avg_ssim:.4f} (SSIM improved by {improvement:.4f})")
            else:
                patience_counter += 1
                print(
                    f"\nEpoch {epoch + 1} - MSE: {avg_loss:.4f}, SSIM: {avg_ssim:.4f} (No improvement for {patience_counter} epochs)")

            # Save model periodically
            checkpoint_freq = self.config.cnn.get("checkpoint_frequency", 5)
            if (epoch + 1) % checkpoint_freq == 0:
                save_path_epoch = os.path.join(save_path, f"cnn_model_epoch_{epoch + 1}.pt")
                try:
                    torch.save({
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': avg_loss,
                        'best_ssim': best_ssim,
                        'loss_history': self.losses
                    }, save_path_epoch)
                    print(f"Saved checkpoint to {save_path_epoch}")
                    # Generate and save images
                    self.generate_and_plot_checkpoint(epoch, num_samples=10, save_dir=os.path.join(save_path, "generated_samples"))
                except Exception as e:
                    print(f"Error saving sample: {str(e)}")

            # Early stopping check
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                # Restore best model
                self.model.load_state_dict(best_model_state)
                self.optimizer.load_state_dict(best_optimizer_state)
                break

        # Save final best model
        try:
            torch.save({
                'model_state': best_model_state,
                'optimizer_state': best_optimizer_state,
                'epoch': epoch,
                'best_ssim': best_ssim,
                'loss_history': self.losses,
                'ssim_history': self.ssim_scores
            }, os.path.join(save_path, "best_model.pt"))
            print(f"Saved best model to {save_path}/best_model.pt")
        except Exception as e:
            print(f"Error saving best model: {str(e)}")

        # Print final training summary
        if self.ssim_scores:
            print("\nTraining completed!")
            print(f"Best SSIM achieved: {best_ssim:.4f}")
            print(f"Final SSIM: {self.ssim_scores[-1]:.4f}")
            print(f"Final MSE loss: {self.losses[-1]:.4f}")
            print(f"Total epochs trained: {epoch + 1}\n{'='*80}\n")


    def generate_and_plot_checkpoint(self, epoch, num_samples=10, save_dir="generated_samples"):
        os.makedirs(save_dir, exist_ok=True)
        images = self.model.generate(num_samples=num_samples, htm_model=self.htm_model)
        save_path = os.path.join(save_dir, f"epoch_{epoch + 1}.png")
        self.model.plot_images(images, num_cols=num_samples, save_path=save_path)


    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            return checkpoint['epoch'], checkpoint['best_ssim']
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return None, None
