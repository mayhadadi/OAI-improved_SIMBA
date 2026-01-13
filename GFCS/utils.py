"""
GFCS Utilities
==============
Helper functions for visualization, evaluation, and data loading.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import json
import os


def compute_l2_norm(x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
    """Compute L2 norm of perturbation."""
    return torch.norm(x_adv - x_orig).item()


def compute_linf_norm(x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
    """Compute L-infinity norm of perturbation."""
    return torch.max(torch.abs(x_adv - x_orig)).item()


def plot_attack_results(
    query_counts: List[int],
    success_mask: List[bool],
    title: str = "GFCS Attack Results",
    save_path: Optional[str] = None
):
    """
    Plot histogram of query counts and CDF.
    
    Similar to Figure 3 in the GFCS paper.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax1 = axes[0]
    successful_queries = [q for q, s in zip(query_counts, success_mask) if s]
    ax1.hist(successful_queries, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(np.median(successful_queries), color='r', linestyle='--', label=f'Median: {np.median(successful_queries):.0f}')
    ax1.axvline(np.mean(successful_queries), color='g', linestyle=':', label=f'Mean: {np.mean(successful_queries):.0f}')
    ax1.set_xlabel('Number of Queries')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Query Distribution')
    ax1.legend()
    ax1.set_xscale('log')
    
    # CDF
    ax2 = axes[1]
    sorted_queries = np.sort(successful_queries)
    cdf = np.arange(1, len(sorted_queries) + 1) / len(sorted_queries)
    ax2.plot(sorted_queries, cdf * 100, 'b-', linewidth=2)
    ax2.set_xlabel('Number of Queries')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Cumulative Success Rate')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_gradient_vs_coimage(
    gradient_queries: List[int],
    coimage_queries: List[int],
    title: str = "Gradient vs Coimage Query Breakdown",
    save_path: Optional[str] = None
):
    """
    Plot scatter of gradient queries vs coimage queries.
    
    Similar to Figure 3 in the GFCS paper showing the breakdown.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Main scatter plot
    ax.scatter(gradient_queries, coimage_queries, alpha=0.5, s=10)
    ax.set_xlabel('Gradient Query Count')
    ax.set_ylabel('Coimage Query Count')
    ax.set_title(title)
    
    # Use log scale for both axes (handling zeros)
    gradient_queries_plot = [max(1, g) for g in gradient_queries]
    coimage_queries_plot = [max(1, c) for c in coimage_queries]
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([0.5, max(gradient_queries_plot) * 2])
    ax.set_ylim([0.5, max(coimage_queries_plot) * 2])
    
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_comparison(
    results_dict: dict,
    metric: str = 'queries',
    title: str = "Method Comparison",
    save_path: Optional[str] = None
):
    """
    Plot comparison between different methods.
    
    Args:
        results_dict: Dictionary mapping method names to lists of query counts
        metric: What to plot ('queries' or 'success_rate')
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for (method_name, queries), color in zip(results_dict.items(), colors):
        sorted_queries = np.sort(queries)
        cdf = np.arange(1, len(sorted_queries) + 1) / len(sorted_queries) * 100
        ax.plot(sorted_queries, cdf, label=method_name, color=color, linewidth=2)
    
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(title)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_perturbation(
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    true_class: int,
    adv_class: int,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize original image, adversarial image, and perturbation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Convert to numpy
    orig_np = x_orig.squeeze().cpu().permute(1, 2, 0).numpy()
    adv_np = x_adv.squeeze().cpu().permute(1, 2, 0).numpy()
    pert_np = (adv_np - orig_np)
    
    # Normalize perturbation for visualization
    pert_viz = (pert_np - pert_np.min()) / (pert_np.max() - pert_np.min() + 1e-8)
    
    axes[0].imshow(np.clip(orig_np, 0, 1))
    axes[0].set_title(f'Original\nClass: {true_class}')
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(adv_np, 0, 1))
    axes[1].set_title(f'Adversarial\nClass: {adv_class}')
    axes[1].axis('off')
    
    axes[2].imshow(pert_viz)
    l2_norm = compute_l2_norm(x_orig, x_adv)
    linf_norm = compute_linf_norm(x_orig, x_adv)
    axes[2].set_title(f'Perturbation\nL2: {l2_norm:.2f}, Linf: {linf_norm:.4f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def save_results(results: dict, filepath: str):
    """Save results to JSON file."""
    # Convert numpy types to python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj
    
    converted = {k: convert(v) if not isinstance(v, list) else [convert(x) for x in v] 
                 for k, v in results.items()}
    
    with open(filepath, 'w') as f:
        json.dump(converted, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_table(results: dict, method_names: List[str]):
    """
    Print results table similar to Table 1 in the GFCS paper.
    """
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'Median Queries':<15} {'Success Rate':<15} {'Avg L2':<10}")
    print("=" * 70)
    
    for name in method_names:
        if name in results:
            r = results[name]
            median_q = np.median(r['query_counts'])
            success = r['success_count'] / r['total_count'] * 100
            avg_l2 = np.mean(r['perturbation_norms'])
            print(f"{name:<20} {median_q:<15.0f} {success:<15.1f}% {avg_l2:<10.2f}")
    
    print("=" * 70)


class EarlyStopping:
    """Early stopping for attack optimization."""
    def __init__(self, patience: int = 100, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def __call__(self, loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


def get_imagenet_classes() -> List[str]:
    """Return ImageNet class names (first 10 for demo)."""
    # In practice, load from imagenet_classes.txt
    return [
        'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead',
        'electric_ray', 'stingray', 'cock', 'hen', 'ostrich'
    ]


if __name__ == "__main__":
    # Demo visualization functions
    print("Utilities module loaded successfully!")
    
    # Create sample data for testing visualization
    sample_queries = list(np.random.exponential(scale=100, size=100).astype(int))
    sample_success = [True] * 95 + [False] * 5
    
    print("\nSample statistics:")
    print(f"Median queries: {np.median([q for q, s in zip(sample_queries, sample_success) if s]):.0f}")
    print(f"Success rate: {sum(sample_success)/len(sample_success)*100:.1f}%")
