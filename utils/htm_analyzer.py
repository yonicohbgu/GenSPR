"""HTM Output Analysis and Visualization Utilities"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Union
from htm.bindings.sdr import SDR, Metrics
import seaborn as sns
import torch


def _to_sdr(t):
    """Convert tensor to SDR if needed."""
    if isinstance(t, torch.Tensor):
        arr = t.cpu().numpy()
        sdr = SDR(arr.shape)
        sdr.dense = arr
        return sdr
    return t


class HTMAnalyzer:
    """
    Analysis toolkit for HTM Spatial Pooler outputs and SDRs.

    Features:
    - Overlap metric for SDR similarity
    - Statistical tracking using HTM Metrics (sparsity, entropy, activation frequency, overlap)
    - SDR visualization (side-by-side comparisons, heatmaps, overlap matrices)
    - Temporal analysis across sequences of SDRs

    Example:
        # >>> analyzer = HTMAnalyzer()
        # >>> overlap = analyzer.overlap_score(sdr1, sdr2)
        # >>> analyzer.plot_sdrs_comparison([sdr1, sdr2], titles=["Original", "Masked"])
        # >>> metrics = analyzer.track_sdr_sequence(sdr_sequence, name="test")
        # >>> analyzer.print_metrics_report("test")
    """

    def __init__(self, history_window: int = 1000):
        """
        Initialize HTM analyzer.

        Args:
            history_window: Number of SDRs to track for metrics (default: 1000)
        """
        self.history_window = history_window
        self.metrics_trackers: Dict[str, Metrics] = {}

    # ==================== Similarity Metric ====================

    def overlap_score(self, sdr1: SDR, sdr2: SDR) -> int:
        """
        Calculate raw overlap score (number of matching active bits).

        Args:
            sdr1, sdr2: SDRs or tensors to compare

        Returns:
            Number of active bits that match between the two SDRs
        """
        sdr1 = _to_sdr(sdr1)
        sdr2 = _to_sdr(sdr2)
        return sdr1.getOverlap(sdr2)

    # ==================== HTM Metrics Tracking ====================

    def track_sdr_sequence(self, sdr_sequence: List[SDR], name: str = "default") -> Metrics:
        """
        Track a sequence of SDRs using HTM Metrics module.

        This automatically computes sparsity, activation frequency, entropy, and overlap
        statistics across the sequence.

        Args:
            sdr_sequence: List of SDRs or tensors to analyze
            name: Identifier for this sequence

        Returns:
            Metrics object with computed statistics
        """
        if not sdr_sequence:
            raise ValueError("SDR sequence cannot be empty")

        # Convert tensors to SDRs
        sdr_sequence = [_to_sdr(sdr) for sdr in sdr_sequence]

        # Create metrics tracker (initialize without SDR to use addData)
        first_sdr = sdr_sequence[0]
        metrics = Metrics(first_sdr.dimensions, self.history_window)

        # Feed all SDRs
        for sdr in sdr_sequence:
            metrics.addData(sdr)

        # Store for later reference
        self.metrics_trackers[name] = metrics

        return metrics

    def get_metrics_report(self, metrics: Union[Metrics, str]) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive metrics report from HTM Metrics.

        Args:
            metrics: Either a Metrics object or name of tracked sequence

        Returns:
            Dictionary with sparsity, activation frequency, entropy, and overlap statistics
        """
        if isinstance(metrics, str):
            if metrics not in self.metrics_trackers:
                raise ValueError(f"No tracked sequence named '{metrics}'")
            metrics = self.metrics_trackers[metrics]

        return {
            'sparsity': {
                'min': metrics.sparsity.min(),
                'mean': metrics.sparsity.mean(),
                'std': metrics.sparsity.std(),
                'max': metrics.sparsity.max(),
            },
            'activation_frequency': {
                'min': metrics.activationFrequency.min(),
                'mean': metrics.activationFrequency.mean(),
                'std': metrics.activationFrequency.std(),
                'max': metrics.activationFrequency.max(),
            },
            'entropy': {
                'value': metrics.activationFrequency.entropy(),
            },
            'overlap': {
                'min': metrics.overlap.min(),
                'mean': metrics.overlap.mean(),
                'std': metrics.overlap.std(),
                'max': metrics.overlap.max(),
            }
        }

    def print_metrics_report(self, metrics: Union[Metrics, str]):
        """
        Pretty print metrics report from HTM Metrics.

        Args:
            metrics: Either a Metrics object or name of tracked sequence
        """
        report = self.get_metrics_report(metrics)

        print("\n" + "="*60)
        print("HTM SDR METRICS REPORT")
        print("="*60)

        print("\n📊 SPARSITY (percentage of active bits)")
        print(f"   Min:  {report['sparsity']['min']:.4f}")
        print(f"   Mean: {report['sparsity']['mean']:.4f}")
        print(f"   Std:  {report['sparsity']['std']:.6f}")
        print(f"   Max:  {report['sparsity']['max']:.4f}")

        print("\n🔥 ACTIVATION FREQUENCY (how often bits activate)")
        print(f"   Min:  {report['activation_frequency']['min']:.4f}")
        print(f"   Mean: {report['activation_frequency']['mean']:.4f}")
        print(f"   Std:  {report['activation_frequency']['std']:.4f}")
        print(f"   Max:  {report['activation_frequency']['max']:.4f}")

        print("\n📈 ENTROPY (information content)")
        print(f"   Value: {report['entropy']['value']:.6f}")
        print(f"   (1.0 = maximum, 0.0 = minimum)")

        print("\n🔗 OVERLAP (between consecutive SDRs)")
        print(f"   Min:  {report['overlap']['min']:.4f}")
        print(f"   Mean: {report['overlap']['mean']:.4f}")
        print(f"   Std:  {report['overlap']['std']:.4f}")
        print(f"   Max:  {report['overlap']['max']:.4f}")

        print("\n" + "="*60 + "\n")

    # ==================== Visualization ====================

    def plot_sdr(self, sdr: SDR, ax=None, title: str = "SDR", cmap: str = 'viridis'):
        """
        Plot a single SDR as a 2D heatmap.

        Args:
            sdr: SDR or tensor to visualize
            ax: Matplotlib axis (creates new if None)
            title: Plot title
            cmap: Colormap name
        """
        sdr = _to_sdr(sdr)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # Get 2D representation
        dense = sdr.dense.reshape(sdr.dimensions)

        im = ax.imshow(dense, cmap=cmap, interpolation='nearest')
        ax.set_title(f"{title}\nSparsity: {sdr.getSparsity():.3f} ({sdr.getSum()} active)")
        ax.axis('off')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        return ax

    def plot_sdrs_comparison(self,
                            sdrs: List[SDR],
                            titles: Optional[List[str]] = None,
                            cmap: str = 'viridis',
                            figsize: Optional[Tuple[int, int]] = None):
        """
        Plot multiple SDRs side by side for comparison.

        Args:
            sdrs: List of SDRs or tensors to compare
            titles: Optional titles for each SDR
            cmap: Colormap name
            figsize: Figure size (auto-calculated if None)
        """
        # Convert tensors to SDRs
        sdrs = [_to_sdr(sdr) for sdr in sdrs]

        n_sdrs = len(sdrs)

        if titles is None:
            titles = [f"SDR {i+1}" for i in range(n_sdrs)]

        if figsize is None:
            figsize = (6 * n_sdrs, 6)

        fig, axes = plt.subplots(1, n_sdrs, figsize=figsize)

        if n_sdrs == 1:
            axes = [axes]

        for i, (sdr, title) in enumerate(zip(sdrs, titles)):
            self.plot_sdr(sdr, ax=axes[i], title=title, cmap=cmap)

        plt.tight_layout()
        plt.show()

    def plot_sdrs_grid(self,
                      sdrs: List[SDR],
                      titles: Optional[List[str]] = None,
                      cmap: str = 'viridis',
                      ncols: int = 5,
                      figsize: Optional[Tuple[int, int]] = None):
        """
        Plot multiple SDRs in a grid layout.

        Args:
            sdrs: List of SDRs or tensors to compare
            titles: Optional titles for each SDR
            cmap: Colormap name
            ncols: Number of columns in the grid (default: 5)
            figsize: Figure size (auto-calculated if None)
        """
        # Convert tensors to SDRs
        sdrs = [_to_sdr(sdr) for sdr in sdrs]

        n_sdrs = len(sdrs)

        if titles is None:
            titles = [f"SDR {i+1}" for i in range(n_sdrs)]

        # Calculate grid dimensions
        nrows = (n_sdrs + ncols - 1) // ncols

        if figsize is None:
            figsize = (5 * ncols, 5 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # Flatten axes for easy indexing
        if nrows == 1 and ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (sdr, title) in enumerate(zip(sdrs, titles)):
            self.plot_sdr(sdr, ax=axes[i], title=title, cmap=cmap)

        # Hide unused subplots
        for i in range(n_sdrs, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_sdr_difference(self, sdr1: SDR, sdr2: SDR,
                           titles: Tuple[str, str] = ("SDR 1", "SDR 2"),
                           figsize: Tuple[int, int] = (18, 5)):
        """
        Plot two SDRs and their difference side by side.

        Args:
            sdr1, sdr2: SDRs or tensors to compare
            titles: Titles for the two SDRs
            figsize: Figure size
        """
        sdr1 = _to_sdr(sdr1)
        sdr2 = _to_sdr(sdr2)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Convert to 2D
        dense1 = sdr1.dense.reshape(sdr1.dimensions)
        dense2 = sdr2.dense.reshape(sdr2.dimensions)

        # Plot first SDR
        im1 = axes[0].imshow(dense1, cmap='viridis', interpolation='nearest')
        axes[0].set_title(f"{titles[0]}\nSparsity: {sdr1.getSparsity():.3f}")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot second SDR
        im2 = axes[1].imshow(dense2, cmap='viridis', interpolation='nearest')
        axes[1].set_title(f"{titles[1]}\nSparsity: {sdr2.getSparsity():.3f}")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Plot difference
        diff = np.abs(dense1 - dense2)
        im3 = axes[2].imshow(diff, cmap='hot', interpolation='nearest')

        overlap = self.overlap_score(sdr1, sdr2)
        axes[2].set_title(f"Difference\nOverlap: {overlap} bits")
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    def plot_overlap_matrix(self, sdrs: List[SDR],
                           labels: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (8, 8)):
        """
        Create a heatmap showing pairwise overlap between multiple SDRs.

        Args:
            sdrs: List of SDRs or tensors to compare
            labels: Optional labels for each SDR
            figsize: Figure size
        """
        # Convert tensors to SDRs
        sdrs = [_to_sdr(sdr) for sdr in sdrs]

        n = len(sdrs)

        if labels is None:
            labels = [f"SDR {i+1}" for i in range(n)]

        # Compute pairwise overlaps
        overlap_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                overlap_matrix[i, j] = self.overlap_score(sdrs[i], sdrs[j])

        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(overlap_matrix,
                   annot=True,
                   fmt='.0f',
                   cmap='YlOrRd',
                   xticklabels=labels,
                   yticklabels=labels,
                   ax=ax)

        ax.set_title('SDR Overlap Matrix (# of matching active bits)')
        plt.tight_layout()
        plt.show()

        return overlap_matrix

    def plot_metrics_over_sequence(self, sdr_sequence: List[SDR],
                                  figsize: Tuple[int, int] = (15, 10)):
        """
        Plot how metrics evolve over a sequence of SDRs.

        Shows sparsity, overlap with previous SDR, and active bit count.

        Args:
            sdr_sequence: List of SDRs or tensors in temporal order
            figsize: Figure size
        """
        if len(sdr_sequence) < 2:
            raise ValueError("Need at least 2 SDRs for sequence analysis")

        # Convert tensors to SDRs
        sdr_sequence = [_to_sdr(sdr) for sdr in sdr_sequence]

        # Collect metrics
        sparsities = [sdr.getSparsity() for sdr in sdr_sequence]
        overlaps = [self.overlap_score(sdr_sequence[i], sdr_sequence[i+1])
                   for i in range(len(sdr_sequence)-1)]
        overlaps.insert(0, np.nan)  # No overlap for first SDR

        active_counts = [sdr.getSum() for sdr in sdr_sequence]

        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=figsize)

        # Sparsity over time
        axes[0].plot(sparsities, marker='o', linewidth=2, markersize=6)
        axes[0].axhline(y=np.mean(sparsities), color='r', linestyle='--',
                       label=f'Mean: {np.mean(sparsities):.3f}')
        axes[0].set_title('Sparsity Over Sequence')
        axes[0].set_ylabel('Sparsity')
        axes[0].set_xlabel('SDR Index')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Overlap with previous SDR
        axes[1].plot(overlaps[1:], marker='s', linewidth=2, markersize=6, color='orange')
        axes[1].axhline(y=np.nanmean(overlaps), color='r', linestyle='--',
                       label=f'Mean: {np.nanmean(overlaps):.1f}')
        axes[1].set_title('Overlap with Previous SDR')
        axes[1].set_ylabel('Overlap (# bits)')
        axes[1].set_xlabel('SDR Index')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Active bit count
        axes[2].plot(active_counts, marker='^', linewidth=2, markersize=6, color='green')
        axes[2].axhline(y=np.mean(active_counts), color='r', linestyle='--',
                       label=f'Mean: {np.mean(active_counts):.1f}')
        axes[2].set_title('Number of Active Bits')
        axes[2].set_ylabel('Active Bits')
        axes[2].set_xlabel('SDR Index')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        plt.show()


# Convenience functions for quick analysis

def quick_compare(sdr1: SDR, sdr2: SDR):
    """Quick comparison of two SDRs with overlap score."""
    analyzer = HTMAnalyzer()
    sdr1 = _to_sdr(sdr1)
    sdr2 = _to_sdr(sdr2)

    overlap = analyzer.overlap_score(sdr1, sdr2)

    print("\n" + "="*50)
    print("SDR COMPARISON")
    print("="*50)
    print(f"Overlap: {overlap} bits")
    print(f"Sparsity SDR1: {sdr1.getSparsity():.4f} ({sdr1.getSum()} active)")
    print(f"Sparsity SDR2: {sdr2.getSparsity():.4f} ({sdr2.getSum()} active)")
    print("="*50 + "\n")

    return overlap


def quick_visualize(sdrs: List[SDR], titles: Optional[List[str]] = None):
    """Quick visualization of multiple SDRs side by side."""
    analyzer = HTMAnalyzer()
    analyzer.plot_sdrs_comparison(sdrs, titles=titles)
