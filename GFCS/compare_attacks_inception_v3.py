"""
Compare GFCS vs GFCS-WeightedAlignment on Inception-v3
========================================================
Tests 20 random images and saves the best example where
WeightedAlignment uses FEWER queries than GFCS
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
import random

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


def load_tiny_imagenet_samples(n_samples=20, size=299):
    """Load random samples from Tiny ImageNet using the project's loader."""
    try:
        # Import the project's loader
        from tiny_imagenet_loader import load_tiny_imagenet_dataset
        
        # Load dataset (returns unnormalized images - only ToTensor applied)
        dataset, label_names = load_tiny_imagenet_dataset(
            dataset_path="./data/tiny_imagenet",
            download=True
        )
        
        # Get random samples
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        samples = []
        
        # Resize if needed
        if size != 224:
            resize_transform = transforms.Resize((size, size))
            for idx in indices:
                img, label = dataset[idx]
                img = resize_transform(transforms.ToPILImage()(img))
                img = transforms.ToTensor()(img).unsqueeze(0)
                samples.append(img)
        else:
            for idx in indices:
                img, label = dataset[idx]
                samples.append(img.unsqueeze(0))
        
        print(f"✓ Loaded {len(samples)} samples from Tiny ImageNet")
        return samples
        
    except Exception as e:
        print(f"✗ Could not load Tiny ImageNet: {e}")
        print("  Make sure tiny_imagenet_loader.py is in the same directory")
        return None


def download_imagenet_samples(n_samples=20, size=299):
    """Download sample images from ImageNet repository."""
    image_urls = [
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02123045_Persian_cat.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02099601_golden_retriever.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02110185_Siberian_husky.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02106662_German_shepherd.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02129604_tiger.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01440764_tench.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02086240_Shih-Tzu.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02504458_African_elephant.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01484850_great_white_shark.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02119789_kit_fox.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02123159_tiger_cat.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02127052_lynx.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02134084_ice_bear.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02504013_Indian_elephant.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n07753592_banana.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01882714_koala.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02088364_beagle.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02108089_boxer.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02108915_French_bulldog.JPEG",
        "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02112018_Pomeranian.JPEG",
    ]
    
    samples = []
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    for url in image_urls[:n_samples]:
        try:
            import requests
            from io import BytesIO
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                samples.append(img_tensor)
                print(f"✓ {url.split('/')[-1]}")
        except Exception as e:
            print(f"✗ Failed: {url.split('/')[-1]}")
            continue
    
    print(f"✓ Downloaded {len(samples)} samples")
    return samples


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
    print("Finding Best WeightedAlignment Example from 20 Images")
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
    
    # Load images
    print("Loading images...")
    print("-"*70)
    images = load_tiny_imagenet_samples(n_samples=20)
    
    if images is None or len(images) == 0:
        print("Downloading ImageNet samples instead...")
        images = download_imagenet_samples(n_samples=20)
    
    if len(images) == 0:
        print("✗ Could not load any images. Exiting.")
        return
    
    print(f"\n✓ Loaded {len(images)} images\n")
    
    # Test each image
    print("="*70)
    print("Testing attacks on each image...")
    print("="*70)
    
    best_result = None
    best_reduction = -float('inf')
    results_summary = []
    
    for idx, img_tensor in enumerate(images):
        img_tensor = img_tensor.to(device)
        
        # Check if correctly classified
        true_class, confidence = get_prediction(victim, img_tensor, device)
        
        print(f"\n[{idx+1}/{len(images)}] Image {idx+1}")
        print(f"  Original: Class {true_class} ({confidence:.1%})")
        
        if confidence < 0.5:
            print(f"  ⚠ Low confidence, skipping...")
            continue
        
        # Run GFCS
        try:
            attacker_gfcs = GFCS(victim, surrogates, epsilon=2.0, max_queries=10000, device=device)
            x_adv_gfcs, stats_gfcs = attacker_gfcs.attack(img_tensor.clone(), true_class)
            
            if not stats_gfcs['success']:
                print(f"  ✗ GFCS failed")
                continue
            
            adv_class_gfcs, adv_conf_gfcs = get_prediction(victim, x_adv_gfcs, device)
            l2_gfcs = torch.norm(x_adv_gfcs - img_tensor).item()
            print(f"  GFCS: {stats_gfcs['total_queries']} queries, L2: {l2_gfcs:.3f}")
        except Exception as e:
            print(f"  ✗ GFCS error: {e}")
            continue
        
        # Run WeightedAlignment
        try:
            attacker_weighted = GFCSWeightedAlignment(victim, surrogates, epsilon=2.0, max_queries=10000, device=device)
            x_adv_weighted, stats_weighted = attacker_weighted.attack(img_tensor.clone(), true_class)
            
            if not stats_weighted['success']:
                print(f"  ✗ WeightedAlignment failed")
                continue
            
            adv_class_weighted, adv_conf_weighted = get_prediction(victim, x_adv_weighted, device)
            l2_weighted = torch.norm(x_adv_weighted - img_tensor).item()
            print(f"  Weighted: {stats_weighted['total_queries']} queries, L2: {l2_weighted:.3f}")
        except Exception as e:
            print(f"  ✗ WeightedAlignment error: {e}")
            continue
        
        # Calculate query reduction
        query_reduction = (stats_gfcs['total_queries'] - stats_weighted['total_queries']) / stats_gfcs['total_queries'] * 100
        
        result_summary = {
            'image_idx': idx + 1,
            'gfcs_queries': stats_gfcs['total_queries'],
            'weighted_queries': stats_weighted['total_queries'],
            'reduction': query_reduction
        }
        results_summary.append(result_summary)
        
        if stats_weighted['total_queries'] < stats_gfcs['total_queries']:
            print(f"  ✓ WeightedAlignment wins! Reduction: {query_reduction:.1f}%")
            
            if query_reduction > best_reduction:
                best_reduction = query_reduction
                best_result = {
                    'img_tensor': img_tensor.clone(),
                    'image_idx': idx + 1,
                    'x_adv_gfcs': x_adv_gfcs.clone(),
                    'x_adv_weighted': x_adv_weighted.clone(),
                    'true_class': true_class,
                    'confidence': confidence,
                    'stats_gfcs': stats_gfcs.copy(),
                    'stats_weighted': stats_weighted.copy(),
                    'adv_class_gfcs': adv_class_gfcs,
                    'adv_conf_gfcs': adv_conf_gfcs,
                    'adv_class_weighted': adv_class_weighted,
                    'adv_conf_weighted': adv_conf_weighted,
                    'l2_gfcs': l2_gfcs,
                    'l2_weighted': l2_weighted,
                    'query_reduction': query_reduction
                }
        else:
            print(f"  GFCS was better (reduction: {query_reduction:.1f}%)")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    if results_summary:
        weighted_wins = sum(1 for r in results_summary if r['reduction'] > 0)
        print(f"WeightedAlignment won: {weighted_wins}/{len(results_summary)} times")
        print(f"Average reduction: {np.mean([r['reduction'] for r in results_summary]):.1f}%")
    
    # Create visualization for best result
    if best_result:
        print(f"\n{'='*70}")
        print(f"BEST RESULT FOUND!")
        print(f"  Image: #{best_result['image_idx']}")
        print(f"  GFCS queries: {best_result['stats_gfcs']['total_queries']}")
        print(f"  WeightedAlignment queries: {best_result['stats_weighted']['total_queries']}")
        print(f"  Query reduction: {best_result['query_reduction']:.1f}%")
        print(f"{'='*70}\n")
        
        print("Creating visualization...")
        
        # Create 2-row comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Attack Comparison - WeightedAlignment More Efficient! ({best_result["query_reduction"]:.1f}% fewer queries)', 
                     fontsize=16, fontweight='bold', color='green')
        
        # Row 1: GFCS
        axes[0, 0].imshow(tensor_to_numpy(best_result['img_tensor']))
        axes[0, 0].set_title(f'Original\nClass: {best_result["true_class"]} ({best_result["confidence"]:.1%})', fontweight='bold')
        axes[0, 0].axis('off')
        axes[0, 0].text(-0.15, 0.5, 'GFCS', transform=axes[0, 0].transAxes, 
                       fontsize=14, fontweight='bold', va='center', rotation=90)
        
        pert_gfcs = tensor_to_numpy((best_result['x_adv_gfcs'] - best_result['img_tensor']).abs())
        axes[0, 1].imshow(np.clip(pert_gfcs * 10, 0, 1))
        axes[0, 1].set_title(f'Perturbation (10x)\nL2: {best_result["l2_gfcs"]:.3f}', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(tensor_to_numpy(best_result['x_adv_gfcs']))
        axes[0, 2].set_title(f'Adversarial\nClass: {best_result["adv_class_gfcs"]} ({best_result["adv_conf_gfcs"]:.1%})\nQueries: {best_result["stats_gfcs"]["total_queries"]}', 
                            fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: WeightedAlignment
        axes[1, 0].imshow(tensor_to_numpy(best_result['img_tensor']))
        axes[1, 0].set_title(f'Original\nClass: {best_result["true_class"]} ({best_result["confidence"]:.1%})', fontweight='bold')
        axes[1, 0].axis('off')
        axes[1, 0].text(-0.15, 0.5, 'Weighted\nAlignment', transform=axes[1, 0].transAxes, 
                       fontsize=13, fontweight='bold', va='center', rotation=90, color='green')
        
        pert_weighted = tensor_to_numpy((best_result['x_adv_weighted'] - best_result['img_tensor']).abs())
        axes[1, 1].imshow(np.clip(pert_weighted * 10, 0, 1))
        axes[1, 1].set_title(f'Perturbation (10x)\nL2: {best_result["l2_weighted"]:.3f}', fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(tensor_to_numpy(best_result['x_adv_weighted']))
        axes[1, 2].set_title(f'Adversarial\nClass: {best_result["adv_class_weighted"]} ({best_result["adv_conf_weighted"]:.1%})\nQueries: {best_result["stats_weighted"]["total_queries"]} ✓', 
                            fontweight='bold', color='green')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('attack_comparison.png', dpi=150, bbox_inches='tight')
        
        print("✓ Saved: attack_comparison.png")
        print("\n" + "="*70)
        plt.show()  # This will display in Colab
        plt.close()
        
        # Print detailed summary
        print(f"\n{'='*70}")
        print(f"SUMMARY:")
        print(f"  WeightedAlignment used {best_result['query_reduction']:.1f}% fewer queries!")
        print(f"  GFCS: {best_result['stats_gfcs']['total_queries']} queries, L2: {best_result['l2_gfcs']:.4f}")
        print(f"  Weighted: {best_result['stats_weighted']['total_queries']} queries, L2: {best_result['l2_weighted']:.4f}")
        print(f"  Trust scores: {best_result['stats_weighted']['trust_scores']}")
        print(f"  Avg alignment: {best_result['stats_weighted']['surrogate_avg_alignment']}")
        print(f"{'='*70}")
    else:
        print("\n⚠ No example found where WeightedAlignment uses fewer queries.")
        print("  This can happen due to randomness in the attacks.")
        print("  Try running again or with more images.")
        print("="*70)


if __name__ == "__main__":
    main()