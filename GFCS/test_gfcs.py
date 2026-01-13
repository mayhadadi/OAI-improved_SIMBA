"""
GFCS Attack Demo
================
Demonstrates the GFCS attack on ImageNet models.

Usage:
    python test_gfcs.py [--num_images 10] [--max_queries 10000]
    
Requirements:
    pip install torch torchvision pillow numpy
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import numpy as np
import argparse
from typing import List, Tuple
import time
import os

from gfcs import GFCS, simba_ods


def load_imagenet_models(device: str = 'cuda') -> Tuple[nn.Module, List[nn.Module]]:
    """
    Load pretrained ImageNet models.
    
    Following the paper:
    - Victim: VGG-16, ResNet-50, or Inception-v3
    - Surrogates: ResNet-152 (single) or {VGG-19, ResNet-34, DenseNet-121, MobileNet-v2} (four)
    
    Returns:
        victim: The victim model
        surrogates: List of surrogate models
    """
    print("Loading pretrained models...")
    
    # Victim model
    victim = models.resnet50(pretrained=True).to(device).eval()
    
    # Surrogate models (using the 4-surrogate setup from the paper)
    surrogates = [
        models.vgg19(pretrained=True).to(device).eval(),
        models.resnet34(pretrained=True).to(device).eval(),
        models.densenet121(pretrained=True).to(device).eval(),
        models.mobilenet_v2(pretrained=True).to(device).eval(),
    ]
    
    print(f"Loaded victim: ResNet-50")
    print(f"Loaded {len(surrogates)} surrogates: VGG-19, ResNet-34, DenseNet-121, MobileNet-v2")
    
    return victim, surrogates


def load_single_surrogate(device: str = 'cuda') -> Tuple[nn.Module, List[nn.Module]]:
    """Load single surrogate setup (ResNet-152)."""
    print("Loading pretrained models (single surrogate setup)...")
    
    victim = models.vgg16(pretrained=True).to(device).eval()
    surrogates = [models.resnet152(pretrained=True).to(device).eval()]
    
    print(f"Loaded victim: VGG-16")
    print(f"Loaded surrogate: ResNet-152")
    
    return victim, surrogates


def get_imagenet_transform():
    """Standard ImageNet preprocessing."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Note: We don't normalize here to keep images in [0,1] range
        # The models expect normalized input, so we'll handle that separately
    ])


def normalize_for_model(x: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return (x - mean) / std


class NormalizedModel(nn.Module):
    """Wrapper that applies normalization before the model."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)


def create_sample_images(num_images: int = 10, device: str = 'cuda') -> List[Tuple[torch.Tensor, int]]:
    """
    Create sample images for testing.
    In practice, you'd load from ImageNet validation set.
    Here we create random images for demonstration.
    """
    print(f"Creating {num_images} sample images...")
    samples = []
    
    for i in range(num_images):
        # Random image in [0, 1] range
        x = torch.rand(1, 3, 224, 224, device=device)
        # Assign a random "true" class
        true_class = np.random.randint(0, 1000)
        samples.append((x, true_class))
    
    return samples


def run_gfcs_attack(
    victim: nn.Module,
    surrogates: List[nn.Module],
    images: List[Tuple[torch.Tensor, int]],
    epsilon: float = 2.0,
    max_queries: int = 10000,
    device: str = 'cuda'
) -> dict:
    """
    Run GFCS attack on a set of images.
    
    Returns:
        results: Dictionary with attack statistics
    """
    # Wrap models with normalization
    victim_wrapped = NormalizedModel(victim).to(device).eval()
    surrogates_wrapped = [NormalizedModel(s).to(device).eval() for s in surrogates]
    
    # Initialize attacker
    attacker = GFCS(
        victim_model=victim_wrapped,
        surrogate_models=surrogates_wrapped,
        epsilon=epsilon,
        max_queries=max_queries,
        targeted=False,
        device=device
    )
    
    results = {
        'success_count': 0,
        'total_count': 0,
        'query_counts': [],
        'gradient_query_counts': [],
        'coimage_query_counts': [],
        'perturbation_norms': []
    }
    
    print(f"\nRunning GFCS attack on {len(images)} images...")
    print(f"Parameters: epsilon={epsilon}, max_queries={max_queries}")
    print("-" * 60)
    
    for i, (x, true_class) in enumerate(images):
        # Get actual prediction from victim
        with torch.no_grad():
            pred_logits = victim_wrapped(x)
            pred_class = pred_logits.argmax(dim=1).item()
        
        # Skip if already misclassified
        if pred_class != true_class:
            print(f"Image {i+1}: Already misclassified, skipping")
            continue
        
        start_time = time.time()
        
        # Run attack
        x_adv, stats = attacker.attack(x, true_class)
        
        elapsed = time.time() - start_time
        
        # Compute perturbation norm
        perturbation_norm = torch.norm(x_adv - x).item()
        
        results['total_count'] += 1
        results['query_counts'].append(stats['total_queries'])
        results['gradient_query_counts'].append(stats['gradient_queries'])
        results['coimage_query_counts'].append(stats['coimage_queries'])
        results['perturbation_norms'].append(perturbation_norm)
        
        if stats['success']:
            results['success_count'] += 1
            print(f"Image {i+1}: SUCCESS - Queries: {stats['total_queries']}, "
                  f"Grad: {stats['gradient_queries']}, ODS: {stats['coimage_queries']}, "
                  f"L2: {perturbation_norm:.2f}, Time: {elapsed:.2f}s")
        else:
            print(f"Image {i+1}: FAILED - Queries: {stats['total_queries']}, "
                  f"L2: {perturbation_norm:.2f}, Time: {elapsed:.2f}s")
    
    # Compute aggregate statistics
    print("\n" + "=" * 60)
    print("GFCS ATTACK RESULTS")
    print("=" * 60)
    
    if results['total_count'] > 0:
        success_rate = results['success_count'] / results['total_count'] * 100
        median_queries = np.median(results['query_counts'])
        mean_queries = np.mean(results['query_counts'])
        median_grad = np.median(results['gradient_query_counts'])
        median_ods = np.median(results['coimage_query_counts'])
        mean_l2 = np.mean(results['perturbation_norms'])
        
        print(f"Success Rate: {success_rate:.1f}% ({results['success_count']}/{results['total_count']})")
        print(f"Median Queries: {median_queries:.0f}")
        print(f"Mean Queries: {mean_queries:.1f}")
        print(f"Median Gradient Queries: {median_grad:.0f}")
        print(f"Median ODS Queries: {median_ods:.0f}")
        print(f"Mean L2 Norm: {mean_l2:.2f}")
    else:
        print("No valid images to attack")
    
    return results


def compare_with_simba_ods(
    victim: nn.Module,
    surrogates: List[nn.Module],
    images: List[Tuple[torch.Tensor, int]],
    max_queries: int = 10000,
    device: str = 'cuda'
):
    """Compare GFCS with SimBA-ODS baseline."""
    print("\n" + "=" * 60)
    print("COMPARISON: GFCS vs SimBA-ODS")
    print("=" * 60)
    
    # Wrap models
    victim_wrapped = NormalizedModel(victim).to(device).eval()
    surrogate_wrapped = NormalizedModel(surrogates[0]).to(device).eval()
    
    gfcs_queries = []
    simba_queries = []
    
    for i, (x, true_class) in enumerate(images[:5]):  # Test on first 5 images
        # Get actual prediction
        with torch.no_grad():
            pred_class = victim_wrapped(x).argmax(dim=1).item()
        
        if pred_class != true_class:
            continue
            
        print(f"\nImage {i+1}:")
        
        # Run GFCS
        attacker = GFCS(
            victim_model=victim_wrapped,
            surrogate_models=[NormalizedModel(s).to(device).eval() for s in surrogates],
            epsilon=2.0,
            max_queries=max_queries,
            device=device
        )
        _, gfcs_stats = attacker.attack(x, true_class)
        print(f"  GFCS: {gfcs_stats['total_queries']} queries, success={gfcs_stats['success']}")
        
        # Run SimBA-ODS
        _, simba_count = simba_ods(
            victim_wrapped, surrogate_wrapped, x, true_class,
            epsilon=2.0, max_queries=max_queries, device=device
        )
        print(f"  SimBA-ODS: {simba_count} queries")
        
        if gfcs_stats['success']:
            gfcs_queries.append(gfcs_stats['total_queries'])
        if simba_count < max_queries:
            simba_queries.append(simba_count)
    
    if gfcs_queries and simba_queries:
        print(f"\nMedian GFCS queries: {np.median(gfcs_queries):.0f}")
        print(f"Median SimBA-ODS queries: {np.median(simba_queries):.0f}")


def main():
    parser = argparse.ArgumentParser(description='GFCS Attack Demo')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to attack')
    parser.add_argument('--max_queries', type=int, default=10000, help='Maximum queries per image')
    parser.add_argument('--epsilon', type=float, default=2.0, help='Step size')
    parser.add_argument('--single_surrogate', action='store_true', help='Use single surrogate (ResNet-152)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--compare', action='store_true', help='Compare with SimBA-ODS')
    args = parser.parse_args()
    
    print("=" * 60)
    print("GFCS: Gradient First, Coimage Second - Attack Demo")
    print("=" * 60)
    print(f"Device: {args.device}")
    
    # Load models
    if args.single_surrogate:
        victim, surrogates = load_single_surrogate(args.device)
    else:
        victim, surrogates = load_imagenet_models(args.device)
    
    # Create sample images (in practice, load from ImageNet)
    images = create_sample_images(args.num_images, args.device)
    
    # For actual testing, you need real images where the victim makes correct predictions
    # Here we'll fake it by using the victim's actual predictions
    print("\nAdjusting true classes to match victim predictions...")
    adjusted_images = []
    for x, _ in images:
        with torch.no_grad():
            true_class = NormalizedModel(victim).to(args.device).eval()(x).argmax(dim=1).item()
        adjusted_images.append((x, true_class))
    
    # Run attack
    results = run_gfcs_attack(
        victim=victim,
        surrogates=surrogates,
        images=adjusted_images,
        epsilon=args.epsilon,
        max_queries=args.max_queries,
        device=args.device
    )
    
    # Optional comparison
    if args.compare:
        compare_with_simba_ods(
            victim=victim,
            surrogates=surrogates,
            images=adjusted_images,
            max_queries=args.max_queries,
            device=args.device
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
