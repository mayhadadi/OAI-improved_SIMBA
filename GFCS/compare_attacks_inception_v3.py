"""
Compare GFCS vs GFCS-WeightedAlignment on Inception-v3
========================================================
Shows: Original Image | Perturbation | Adversarial Image
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add GFCS directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)


def load_sample_image(image_path=None, size=299):
    """Load a real ImageNet image."""
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
        print(f"✓ Loaded image from {image_path}")
    else:
        # Try multiple image sources
        image_urls = [
            "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02123045_Persian_cat.JPEG",
            "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02099601_golden_retriever.JPEG",
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        ]
        
        img = None
        for url in image_urls:
            try:
                import requests
                from io import BytesIO
                print(f"Trying: {url.split('/')[-1]}")
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    print(f"✓ Downloaded successfully")
                    break
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        if img is None:
            print("⚠ Creating random image (all downloads failed)")
            img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    return transform(img).unsqueeze(0)


def tensor_to_numpy(tensor):
    """Convert tensor to numpy for visualization."""
    img = tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0, 1)


def get_prediction(model, img_tensor, device):
    """Get model prediction."""
    with torch.no_grad():
        output = model(img_tensor.to(device))
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("="*70)
    print("GFCS vs GFCS-WeightedAlignment Comparison")
    print("="*70)
    print(f"Device: {device}\n")
    
    # Load models
    print("Loading models...")
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    victim = NormalizedModel(models.inception_v3(pretrained=True), imagenet_mean, imagenet_std).to(device).eval()
    surrogates = [
        NormalizedModel(models.resnet50(pretrained=True), imagenet_mean, imagenet_std).to(device).eval(),
        NormalizedModel(models.vgg16(pretrained=True), imagenet_mean, imagenet_std).to(device).eval(),
        NormalizedModel(models.densenet121(pretrained=True), imagenet_mean, imagenet_std).to(device).eval(),
    ]
    print("✓ Models loaded\n")
    
    # Load image
    print("Loading image...")
    img_tensor = load_sample_image().to(device)
    true_class, confidence = get_prediction(victim, img_tensor, device)
    print(f"Original: Class {true_class} ({confidence:.2%})\n")
    
    # GFCS Attack
    print("="*70)
    print("Running GFCS Attack...")
    print("="*70)
    attacker_gfcs = GFCS(victim, surrogates, epsilon=2.0, max_queries=10000, device=device)
    x_adv_gfcs, stats_gfcs = attacker_gfcs.attack(img_tensor.clone(), true_class)
    
    if stats_gfcs['success']:
        adv_class_gfcs, adv_conf_gfcs = get_prediction(victim, x_adv_gfcs, device)
        l2_gfcs = torch.norm(x_adv_gfcs - img_tensor).item()
        print(f"✓ Success: Class {adv_class_gfcs} ({adv_conf_gfcs:.2%})")
        print(f"  Queries: {stats_gfcs['total_queries']}, L2: {l2_gfcs:.4f}\n")
    
    # WeightedAlignment Attack
    print("="*70)
    print("Running GFCS-WeightedAlignment Attack...")
    print("="*70)
    attacker_weighted = GFCSWeightedAlignment(victim, surrogates, epsilon=2.0, max_queries=10000, device=device)
    x_adv_weighted, stats_weighted = attacker_weighted.attack(img_tensor.clone(), true_class)
    
    if stats_weighted['success']:
        adv_class_weighted, adv_conf_weighted = get_prediction(victim, x_adv_weighted, device)
        l2_weighted = torch.norm(x_adv_weighted - img_tensor).item()
        print(f"✓ Success: Class {adv_class_weighted} ({adv_conf_weighted:.2%})")
        print(f"  Queries: {stats_weighted['total_queries']}, L2: {l2_weighted:.4f}\n")
    
    # Visualizations
    print("="*70)
    print("Creating Visualizations...")
    print("="*70)
    
    if stats_gfcs['success'] and stats_weighted['success']:
        # Create 2-row comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Attack Comparison', fontsize=16, fontweight='bold')
        
        # Row 1: GFCS
        axes[0, 0].imshow(tensor_to_numpy(img_tensor))
        axes[0, 0].set_title(f'Original\nClass: {true_class} ({confidence:.1%})', fontweight='bold')
        axes[0, 0].axis('off')
        axes[0, 0].text(-0.15, 0.5, 'GFCS', transform=axes[0, 0].transAxes, 
                       fontsize=14, fontweight='bold', va='center', rotation=90)
        
        pert_gfcs = tensor_to_numpy((x_adv_gfcs - img_tensor).abs())
        axes[0, 1].imshow(np.clip(pert_gfcs * 10, 0, 1))
        axes[0, 1].set_title(f'Perturbation (10x)\nL2: {l2_gfcs:.3f}', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(tensor_to_numpy(x_adv_gfcs))
        axes[0, 2].set_title(f'Adversarial\nClass: {adv_class_gfcs} ({adv_conf_gfcs:.1%})\nQueries: {stats_gfcs["total_queries"]}', 
                            fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: WeightedAlignment
        axes[1, 0].imshow(tensor_to_numpy(img_tensor))
        axes[1, 0].set_title(f'Original\nClass: {true_class} ({confidence:.1%})', fontweight='bold')
        axes[1, 0].axis('off')
        axes[1, 0].text(-0.15, 0.5, 'Weighted\nAlignment', transform=axes[1, 0].transAxes, 
                       fontsize=13, fontweight='bold', va='center', rotation=90)
        
        pert_weighted = tensor_to_numpy((x_adv_weighted - img_tensor).abs())
        axes[1, 1].imshow(np.clip(pert_weighted * 10, 0, 1))
        axes[1, 1].set_title(f'Perturbation (10x)\nL2: {l2_weighted:.3f}', fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(tensor_to_numpy(x_adv_weighted))
        axes[1, 2].set_title(f'Adversarial\nClass: {adv_class_weighted} ({adv_conf_weighted:.1%})\nQueries: {stats_weighted["total_queries"]}', 
                            fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('attack_comparison.png', dpi=150, bbox_inches='tight')
        
        print("✓ Saved: attack_comparison.png")
        print("\n" + "="*70)
        plt.show()  # This will display in Colab
        plt.close()
        
        # Print summary
        query_reduction = (stats_gfcs['total_queries'] - stats_weighted['total_queries']) / stats_gfcs['total_queries'] * 100
        print(f"\n{'='*70}")
        print(f"SUMMARY:")
        print(f"  Query Reduction: {query_reduction:+.1f}%")
        print(f"  GFCS L2: {l2_gfcs:.4f} | Weighted L2: {l2_weighted:.4f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()