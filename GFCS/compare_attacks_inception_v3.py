"""
Compare GFCS vs GFCS-WeightedAlignment on Inception-v3
========================================================
This script:
1. Loads Inception-v3 as victim
2. Loads surrogate models (ResNet-50, VGG-16, DenseNet-121)
3. Attacks a sample image with both:
   - Original GFCS (from gfcs.py)
   - GFCS-WeightedAlignment (from gfcs_weighted_alignment.py)
4. Visualizes original image and both adversarial examples side by side
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the GFCS directory to path (adjust as needed)
# Assuming this script is in the GFCS directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your existing implementations
from gfcs import GFCS
from gfcs_weighted_alignment import GFCSWeightedAlignment


class NormalizedModel(nn.Module):
    """Wrapper that applies normalization before the model."""
    def __init__(self, model: nn.Module, mean: list, std: list):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is in [0, 1] range
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)


def load_sample_image(image_path=None, size=299):
    """
    Load a sample image.
    If image_path is None, downloads a sample from internet or creates random image.
    """
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
        print(f"✓ Loaded image from {image_path}")
    else:
        # Try to download a sample image
        try:
            import requests
            from io import BytesIO
            url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg"
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            print("✓ Downloaded sample cat image")
        except:
            # Create a random image if download fails
            print("⚠ Creating random image (download failed)")
            img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
    
    # Transform to tensor
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization."""
    img = tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    return img


def get_prediction(model, img_tensor, device):
    """Get model prediction and confidence."""
    with torch.no_grad():
        output = model(img_tensor.to(device))
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("="*70)
    print("GFCS vs GFCS-WeightedAlignment Comparison on Inception-v3")
    print("="*70)
    print(f"Device: {device}\n")
    
    # ========================================
    # Load Models
    # ========================================
    print("Loading Models...")
    print("-"*70)
    
    # ImageNet normalization parameters
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Victim: Inception-v3
    print("Loading Inception-v3 (victim)...", end=" ")
    victim_base = models.inception_v3(pretrained=True)
    victim = NormalizedModel(victim_base, imagenet_mean, imagenet_std).to(device).eval()
    print("✓")
    
    # Surrogates
    print("Loading surrogate models...")
    surrogate_models = [
        ("ResNet-50", models.resnet50(pretrained=True)),
        ("VGG-16", models.vgg16(pretrained=True)),
        ("DenseNet-121", models.densenet121(pretrained=True)),
    ]
    
    surrogates = []
    for name, model in surrogate_models:
        surr = NormalizedModel(model, imagenet_mean, imagenet_std).to(device).eval()
        surrogates.append(surr)
        print(f"  ✓ {name}")
    
    print(f"\nTotal: 1 victim + {len(surrogates)} surrogates\n")
    
    # ========================================
    # Load Sample Image
    # ========================================
    print("Loading Sample Image...")
    print("-"*70)
    
    # You can specify your own image path here
    img_tensor = load_sample_image(image_path=None, size=299)
    img_tensor = img_tensor.to(device)
    
    # Get original prediction
    true_class, confidence = get_prediction(victim, img_tensor, device)
    print(f"Original classification: Class {true_class} (confidence: {confidence:.2%})\n")
    
    # ========================================
    # Attack Parameters
    # ========================================
    epsilon = 2.0
    max_queries = 10000
    
    print("Attack Parameters:")
    print("-"*70)
    print(f"Epsilon (step size): {epsilon}")
    print(f"Max queries: {max_queries}")
    print(f"Attack type: Untargeted\n")
    
    # ========================================
    # Attack 1: Original GFCS
    # ========================================
    print("="*70)
    print("ATTACK 1: Original GFCS")
    print("="*70)
    
    attacker_gfcs = GFCS(
        victim_model=victim,
        surrogate_models=surrogates,
        epsilon=epsilon,
        max_queries=max_queries,
        targeted=False,
        device=device
    )
    
    x_adv_gfcs, stats_gfcs = attacker_gfcs.attack(img_tensor.clone(), true_class)
    
    print(f"\nResults:")
    print(f"  Success: {stats_gfcs['success']}")
    print(f"  Total queries: {stats_gfcs['total_queries']}")
    print(f"  Gradient queries: {stats_gfcs['gradient_queries']}")
    print(f"  Coimage queries: {stats_gfcs['coimage_queries']}")
    
    if stats_gfcs['success']:
        adv_class_gfcs, adv_conf_gfcs = get_prediction(victim, x_adv_gfcs, device)
        l2_norm_gfcs = torch.norm(x_adv_gfcs - img_tensor).item()
        print(f"  Adversarial class: {adv_class_gfcs} (confidence: {adv_conf_gfcs:.2%})")
        print(f"  L2 perturbation: {l2_norm_gfcs:.4f}")
    else:
        print(f"  Attack failed - no adversarial example found")
        adv_class_gfcs, adv_conf_gfcs, l2_norm_gfcs = None, None, None
    
    # ========================================
    # Attack 2: GFCS-WeightedAlignment
    # ========================================
    print("\n" + "="*70)
    print("ATTACK 2: GFCS-WeightedAlignment")
    print("="*70)
    
    attacker_weighted = GFCSWeightedAlignment(
        victim_model=victim,
        surrogate_models=surrogates,
        epsilon=epsilon,
        max_queries=max_queries,
        targeted=False,
        device=device,
        use_adaptive_weighting=True,
        use_smart_ods=True
    )
    
    x_adv_weighted, stats_weighted = attacker_weighted.attack(img_tensor.clone(), true_class)
    
    print(f"\nResults:")
    print(f"  Success: {stats_weighted['success']}")
    print(f"  Total queries: {stats_weighted['total_queries']}")
    print(f"  Gradient queries: {stats_weighted['gradient_queries']}")
    print(f"  Coimage queries: {stats_weighted['coimage_queries']}")
    print(f"  Final loss: {stats_weighted['final_loss']:.4f}")
    print(f"  Trust scores: {[f'{t:.2f}' for t in stats_weighted['trust_scores']]}")
    print(f"  Avg alignment: {[f'{a:.2f}' for a in stats_weighted['surrogate_avg_alignment']]}")
    
    if stats_weighted['success']:
        adv_class_weighted, adv_conf_weighted = get_prediction(victim, x_adv_weighted, device)
        l2_norm_weighted = torch.norm(x_adv_weighted - img_tensor).item()
        print(f"  Adversarial class: {adv_class_weighted} (confidence: {adv_conf_weighted:.2%})")
        print(f"  L2 perturbation: {l2_norm_weighted:.4f}")
    else:
        print(f"  Attack failed - no adversarial example found")
        adv_class_weighted, adv_conf_weighted, l2_norm_weighted = None, None, None
    
    # ========================================
    # Comparison Summary
    # ========================================
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"{'Metric':<30} {'GFCS':<20} {'WeightedAlignment':<20}")
    print("-"*70)
    
    success_gfcs = '✓' if stats_gfcs['success'] else '✗'
    success_weighted = '✓' if stats_weighted['success'] else '✗'
    print(f"{'Success':<30} {success_gfcs:<20} {success_weighted:<20}")
    print(f"{'Total Queries':<30} {stats_gfcs['total_queries']:<20} {stats_weighted['total_queries']:<20}")
    print(f"{'Gradient Queries':<30} {stats_gfcs['gradient_queries']:<20} {stats_weighted['gradient_queries']:<20}")
    print(f"{'Coimage Queries':<30} {stats_gfcs['coimage_queries']:<20} {stats_weighted['coimage_queries']:<20}")
    
    if stats_gfcs['success'] and stats_weighted['success']:
        print(f"{'L2 Perturbation':<30} {l2_norm_gfcs:<20.4f} {l2_norm_weighted:<20.4f}")
        print(f"{'Adversarial Class':<30} {adv_class_gfcs:<20} {adv_class_weighted:<20}")
        
        # Calculate query reduction
        query_reduction = ((stats_gfcs['total_queries'] - stats_weighted['total_queries']) / 
                          stats_gfcs['total_queries'] * 100)
        print(f"\n⚡ Query reduction: {query_reduction:+.1f}%")
        
        if l2_norm_weighted < l2_norm_gfcs:
            print(f"⚡ WeightedAlignment achieved {((l2_norm_gfcs - l2_norm_weighted) / l2_norm_gfcs * 100):.1f}% smaller perturbation")
    elif stats_gfcs['success']:
        print(f"{'L2 Perturbation':<30} {l2_norm_gfcs:<20.4f} {'N/A':<20}")
        print(f"{'Adversarial Class':<30} {adv_class_gfcs:<20} {'N/A':<20}")
        print("\n⚠ Only GFCS succeeded")
    elif stats_weighted['success']:
        print(f"{'L2 Perturbation':<30} {'N/A':<20} {l2_norm_weighted:<20.4f}")
        print(f"{'Adversarial Class':<30} {'N/A':<20} {adv_class_weighted:<20}")
        print("\n⚠ Only WeightedAlignment succeeded")
    else:
        print("\n⚠ Both attacks failed")
    
    # ========================================
    # Visualization
    # ========================================
    print("\n" + "="*70)
    print("Generating Visualizations...")
    print("="*70)
    
    # Main comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(tensor_to_numpy(img_tensor))
    axes[0].set_title(f'Original Image\nClass: {true_class}\nConfidence: {confidence:.2%}', 
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # GFCS result
    if stats_gfcs['success']:
        axes[1].imshow(tensor_to_numpy(x_adv_gfcs))
        axes[1].set_title(
            f'GFCS Attack\n'
            f'Class: {adv_class_gfcs} (Conf: {adv_conf_gfcs:.2%})\n'
            f'Queries: {stats_gfcs["total_queries"]} | L2: {l2_norm_gfcs:.3f}',
            fontsize=12, fontweight='bold'
        )
    else:
        axes[1].text(0.5, 0.5, 'Attack Failed', ha='center', va='center', fontsize=16)
        axes[1].set_title('GFCS Attack', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # WeightedAlignment result
    if stats_weighted['success']:
        axes[2].imshow(tensor_to_numpy(x_adv_weighted))
        axes[2].set_title(
            f'GFCS-WeightedAlignment\n'
            f'Class: {adv_class_weighted} (Conf: {adv_conf_weighted:.2%})\n'
            f'Queries: {stats_weighted["total_queries"]} | L2: {l2_norm_weighted:.3f}',
            fontsize=12, fontweight='bold'
        )
    else:
        axes[2].text(0.5, 0.5, 'Attack Failed', ha='center', va='center', fontsize=16)
        axes[2].set_title('GFCS-WeightedAlignment', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/gfcs_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: gfcs_comparison.png")
    
    # Perturbation visualization
    if stats_gfcs['success'] and stats_weighted['success']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image (for reference)
        axes[0].imshow(tensor_to_numpy(img_tensor))
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # GFCS perturbation
        perturbation_gfcs = (x_adv_gfcs - img_tensor).abs()
        pert_vis_gfcs = tensor_to_numpy(perturbation_gfcs)
        # Amplify for visibility
        pert_vis_gfcs = pert_vis_gfcs / (pert_vis_gfcs.max() + 1e-8)
        
        axes[1].imshow(pert_vis_gfcs)
        axes[1].set_title(f'GFCS Perturbation\nL2 norm: {l2_norm_gfcs:.4f}', 
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # WeightedAlignment perturbation
        perturbation_weighted = (x_adv_weighted - img_tensor).abs()
        pert_vis_weighted = tensor_to_numpy(perturbation_weighted)
        # Amplify for visibility
        pert_vis_weighted = pert_vis_weighted / (pert_vis_weighted.max() + 1e-8)
        
        axes[2].imshow(pert_vis_weighted)
        axes[2].set_title(f'WeightedAlignment Perturbation\nL2 norm: {l2_norm_weighted:.4f}', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/perturbations_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: perturbations_comparison.png")
    
    print("\n" + "="*70)
    print("✓ All visualizations saved to /mnt/user-data/outputs/")
    print("="*70)


if __name__ == "__main__":
    main()