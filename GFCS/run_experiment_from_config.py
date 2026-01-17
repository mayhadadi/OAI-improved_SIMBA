"""
GFCS Experiment Runner (with SimBA and Minimal Victim GFCS support)
======================
Runs experiments based on JSON configuration files.
Supports GFCS, SimBA, and GFCS Minimal Victim attacks.

Usage:
    python run_experiment_from_config.py exp_001
    python run_experiment_from_config.py exp_001 exp_002 exp_003
    python run_experiment_from_config.py --config_dir ./my_configs exp_001
    python run_experiment_from_config.py --list  # List all available experiments
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder
import numpy as np
import json
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import time
from datetime import datetime

from tiny_imagenet_loader import download_tiny_imagenet, load_tiny_imagenet_dataset
from gfcs import GFCS
from SimBA import SimBA
from cifar10_models import load_cifar10_model

# Import minimal victim GFCS
try:
    from gfcs_minimal_victim_queries import GFCSMinimalVictimQueries
    MINIMAL_VICTIM_AVAILABLE = True
except ImportError:
    MINIMAL_VICTIM_AVAILABLE = False
    print("WARNING: gfcs_minimal_victim_queries.py not found. Minimal victim method will not be available.")


class NormalizedModel(nn.Module):
    """Wrapper that applies normalization before the model."""
    def __init__(self, model: nn.Module, mean: List[float], std: List[float]):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate experiment configuration.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check required top-level fields
    required_fields = ['experiment_id', 'victim', 'surrogates', 'dataset', 'attack']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return errors
    
    # Validate victim
    if 'model_name' not in config['victim']:
        errors.append("Missing victim.model_name")
    
    # Validate surrogates (can be empty for SimBA)
    if not isinstance(config['surrogates'], list):
        errors.append("surrogates must be a list (can be empty for SimBA)")
    
    # Validate dataset
    dataset_name = config['dataset'].get('name')
    if dataset_name not in ['cifar10', 'imagenet', 'imagenet_r', 'tiny_imagenet', 'custom']:
        errors.append(f"Invalid dataset name: {dataset_name}")
    
    # Validate attack method
    attack_method = config['attack'].get('method')
    if attack_method not in ['gfcs', 'gfcs_minimal_victim', 'simba']:
        errors.append(f"Invalid attack method: {attack_method}. Must be 'gfcs', 'gfcs_minimal_victim', or 'simba'")
    
    # Check that minimal victim GFCS is available if requested
    if attack_method == 'gfcs_minimal_victim' and not MINIMAL_VICTIM_AVAILABLE:
        errors.append("gfcs_minimal_victim method requested but gfcs_minimal_victim_queries.py not found")
    
    # Validate max_iterations for minimal victim GFCS
    if attack_method == 'gfcs_minimal_victim':
        if 'max_iterations' not in config['attack']:
            errors.append("'max_iterations' required for gfcs_minimal_victim method")
    
    # Check dataset path for non-auto datasets
    if dataset_name in ['imagenet', 'imagenet_r', 'custom']:
        dataset_path = config['dataset'].get('path')
        if not dataset_path:
            errors.append(f"Dataset path required for {dataset_name}")
    
    return errors


def load_model(model_config: Dict[str, Any], device: str) -> nn.Module:
    """
    Load a model based on configuration.
    
    Args:
        model_config: Model configuration dict
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model_name = model_config['model_name']
    num_classes = model_config.get('num_classes', 1000)
    
    print(f"Loading {model_name} (num_classes={num_classes})...")
    
    # Load CIFAR-10 pretrained models if num_classes=10
    if num_classes == 10:
        model = load_cifar10_model(model_name, device)
    # Load ImageNet pretrained models
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        if num_classes != 1000:
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        if num_classes != 1000:
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        if num_classes != 1000:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        if num_classes != 1000:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device).eval()
    
    # Wrap with normalization
    mean = model_config['normalization']['mean']
    std = model_config['normalization']['std']
    model_wrapped = NormalizedModel(model, mean, std).to(device)
    
    return model_wrapped


def load_dataset(dataset_config: Dict[str, Any], device: str) -> List[Tuple[torch.Tensor, int]]:
    """
    Load dataset based on configuration.
    
    Supports: cifar10, tiny_imagenet, imagenet (auto-fallback to tiny_imagenet), custom
    """
    dataset_name = dataset_config['name']
    num_images = dataset_config['num_images']
    seed = dataset_config['seed']
    image_size = dataset_config.get('image_size', 224)
    
    print(f"Loading {dataset_name} dataset (num_images={num_images}, seed={seed})...")
    
    # Set seed
    np.random.seed(seed)
    
    if dataset_name == 'cifar10':
        # CIFAR-10: Auto-download, no normalization (handled by NormalizedModel)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        
    elif dataset_name == 'tiny_imagenet':
        # Tiny ImageNet: Auto-download, pre-normalized
        print(f"Using Tiny ImageNet...")
        
        tiny_dataset, label_names = load_tiny_imagenet_dataset(
            dataset_path="./data/tiny_imagenet",
            download=True
        )
        
        # Sample images (Tiny ImageNet is already normalized in the loader)
        indices = np.random.choice(
            len(tiny_dataset),
            size=min(num_images, len(tiny_dataset)),
            replace=False
        )
        
        samples = []
        for idx in indices:
            img, label = tiny_dataset[int(idx)]
            img = img.unsqueeze(0).to(device)
            samples.append((img, label))
        
        print(f"✓ Loaded {len(samples)} samples from Tiny ImageNet")
        return samples
        
    elif dataset_name in ['imagenet', 'imagenet_r']:
        # ImageNet / ImageNet-R: Try full dataset, fallback to Tiny ImageNet
        dataset_path = dataset_config['path']
        
        if not os.path.exists(dataset_path):
            # AUTO-FALLBACK to Tiny ImageNet
            print(f"⚠️  ImageNet path not found: {dataset_path}")
            print(f"📥 Automatically using Tiny ImageNet instead...")
            
            tiny_dataset, label_names = load_tiny_imagenet_dataset(
                dataset_path="./data/tiny_imagenet",
                download=True
            )
            
            # Sample images (already normalized)
            indices = np.random.choice(
                len(tiny_dataset),
                size=min(num_images, len(tiny_dataset)),
                replace=False
            )
            
            samples = []
            for idx in indices:
                img, label = tiny_dataset[int(idx)]
                img = img.unsqueeze(0).to(device)
                samples.append((img, label))
            
            print(f"✓ Loaded {len(samples)} samples from Tiny ImageNet")
            return samples
        
        # Full ImageNet exists - use it
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        dataset = ImageFolder(root=dataset_path, transform=transform)
        
    elif dataset_name == 'custom':
        # Custom dataset
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        dataset_path = dataset_config['path']
        dataset = ImageFolder(root=dataset_path, transform=transform)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Sample indices (for non-tiny_imagenet datasets)
    indices = np.random.choice(len(dataset), size=min(num_images, len(dataset)), replace=False)
    
    # Load samples
    samples = []
    for idx in indices:
        img, label = dataset[int(idx)]
        img = img.unsqueeze(0).to(device)
        samples.append((img, label))
    
    print(f"Loaded {len(samples)} samples")
    return samples


def filter_correctly_classified(
    samples: List[Tuple[torch.Tensor, int]],
    victim_model: nn.Module
) -> List[Tuple[torch.Tensor, int]]:
    """Filter samples to only keep correctly classified ones."""
    print("Filtering correctly classified samples...")
    
    filtered = []
    for img, true_label in samples:
        with torch.no_grad():
            logits = victim_model(img)
            pred_label = logits.argmax(dim=1).item()
        if pred_label == true_label:
            filtered.append((img, true_label))
    
    accuracy = (len(filtered) / len(samples)) * 100 if len(samples) > 0 else 0
    print(f"Victim accuracy: {accuracy:.2f}% ({len(filtered)}/{len(samples)})")
    
    return filtered


def run_minimal_victim_attack(
    samples: List[Tuple[torch.Tensor, int]],
    victim_model: nn.Module,
    surrogate_models: List[nn.Module],
    attack_config: Dict[str, Any],
    device: str
) -> Dict[str, Any]:
    """
    Run GFCS attack with minimal victim queries.
    
    Args:
        samples: List of (image, label) tuples
        victim_model: Victim model
        surrogate_models: List of surrogate models
        attack_config: Attack configuration
        device: Device to use
        
    Returns:
        Dictionary with attack results
    """
    if not MINIMAL_VICTIM_AVAILABLE:
        raise RuntimeError("gfcs_minimal_victim_queries module not available")
    
    # Extract attack parameters
    epsilon = attack_config.get('epsilon', 2.0)
    max_iterations = attack_config.get('max_iterations', 1000)
    targeted = attack_config.get('targeted', False)
    norm_bound_config = attack_config.get('norm_bound', {'type': 'auto', 'value': None})
    
    # Determine norm bound
    if norm_bound_config['type'] == 'auto':
        norm_bound = None  # Will be auto-computed
    else:
        norm_bound = norm_bound_config.get('value')
    
    print(f"\nRunning gfcs_minimal_victim attack...")
    print(f"Parameters: epsilon={epsilon}, max_iterations={max_iterations}, norm_bound={norm_bound}")
    print(f"Number of samples: {len(samples)}")
    print(f"Number of surrogates: {len(surrogate_models)}")
    
    # Create attacker
    attacker = GFCSMinimalVictimQueries(
        victim_model=victim_model,
        surrogate_models=surrogate_models,
        epsilon=epsilon,
        norm_bound=norm_bound,
        max_iterations=max_iterations,
        targeted=targeted,
        device=device
    )
    
    # Track results
    results = {
        'success_count': 0,
        'total_samples': len(samples),
        'victim_queries': [],
        'surrogate_queries': [],
        'iterations': [],
        'perturbation_norms': [],
        'times': [],
        'failed_indices': []
    }
    
    # Attack each sample
    for idx, (img, true_class) in enumerate(samples):
        print(f"[{idx+1}/{len(samples)}] Attacking image...", end=' ')
        
        start_time = time.time()
        
        try:
            x_adv, stats = attacker.attack(img, true_class)
            elapsed_time = time.time() - start_time
            
            # Record results
            if stats['success']:
                results['success_count'] += 1
            
            results['victim_queries'].append(stats['victim_queries'])
            results['surrogate_queries'].append(stats['surrogate_queries'])
            results['iterations'].append(stats['iterations'])
            results['times'].append(elapsed_time)
            
            # Compute perturbation norm
            delta = (x_adv - img).view(1, -1)
            pert_norm = torch.norm(delta, p=2).item()
            results['perturbation_norms'].append(pert_norm)
            
            if stats['success']:
                print(f"✓ SUCCESS - VictimQ:{stats['victim_queries']}, "
                      f"SurrQ:{stats['surrogate_queries']}, "
                      f"Iters:{stats['iterations']}, "
                      f"L2:{pert_norm:.2f}, Time:{elapsed_time:.2f}s")
            else:
                results['failed_indices'].append(idx)
                print(f"✗ FAILED - VictimQ:{stats['victim_queries']}, "
                      f"SurrQ:{stats['surrogate_queries']}, "
                      f"Iters:{stats['iterations']}, "
                      f"L2:{pert_norm:.2f}, Time:{elapsed_time:.2f}s")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"✗ ERROR: {str(e)}")
            results['failed_indices'].append(idx)
            results['victim_queries'].append(0)
            results['surrogate_queries'].append(0)
            results['iterations'].append(0)
            results['perturbation_norms'].append(0)
            results['times'].append(elapsed_time)
    
    # Compute summary statistics
    results['success_rate'] = (results['success_count'] / results['total_samples']) * 100
    results['median_victim_queries'] = float(np.median(results['victim_queries']))
    results['mean_victim_queries'] = float(np.mean(results['victim_queries']))
    results['median_surrogate_queries'] = float(np.median(results['surrogate_queries']))
    results['mean_surrogate_queries'] = float(np.mean(results['surrogate_queries']))
    results['median_iterations'] = float(np.median(results['iterations']))
    results['mean_iterations'] = float(np.mean(results['iterations']))
    results['mean_perturbation_norm'] = float(np.mean(results['perturbation_norms']))
    results['mean_time'] = float(np.mean(results['times']))
    
    # For compatibility with existing code, add query_counts
    results['query_counts'] = results['victim_queries']
    results['median_queries'] = results['median_victim_queries']
    results['mean_queries'] = results['mean_victim_queries']
    
    return results


def run_attack(
    samples: List[Tuple[torch.Tensor, int]],
    victim_model: nn.Module,
    surrogate_models: List[nn.Module],
    attack_config: Dict[str, Any],
    device: str
) -> Dict[str, Any]:
    """
    Run attack based on configuration.
    
    Supports GFCS, GFCS Minimal Victim, and SimBA attacks.
    
    Args:
        samples: List of (image, label) tuples
        victim_model: Victim model
        surrogate_models: List of surrogate models (empty for SimBA)
        attack_config: Attack configuration dict
        device: Device
        
    Returns:
        Results dictionary
    """
    method = attack_config.get('method', 'gfcs')
    
    # Route to minimal victim GFCS if requested
    if method == 'gfcs_minimal_victim':
        return run_minimal_victim_attack(samples, victim_model, surrogate_models, attack_config, device)
    
    # Original GFCS and SimBA code
    epsilon = attack_config.get('epsilon', 2.0)
    max_queries = attack_config.get('max_queries', 10000)
    targeted = attack_config.get('targeted', False)
    
    print(f"\nRunning {method} attack...")
    print(f"Parameters: epsilon={epsilon}, max_queries={max_queries}")
    print(f"Number of samples: {len(samples)}")
    
    if method == 'gfcs':
        # Get norm bound
        norm_bound_config = attack_config.get('norm_bound', {'type': 'auto'})
        if norm_bound_config['type'] == 'auto':
            norm_bound = None  # Will be computed by GFCS
        elif norm_bound_config['type'] == 'fixed':
            norm_bound = norm_bound_config['value']
        else:
            norm_bound = None
        
        print(f"Number of surrogates: {len(surrogate_models)}")
        print(f"Norm bound: {norm_bound}")
        
        attacker = GFCS(
            victim_model=victim_model,
            surrogate_models=surrogate_models,
            epsilon=epsilon,
            norm_bound=norm_bound,
            max_queries=max_queries,
            targeted=targeted,
            device=device
        )
        
    elif method == 'simba':
        # SimBA-specific parameters
        pixel_attack = attack_config.get('pixel_attack', True)
        freq_dims = attack_config.get('freq_dims', None)
        order = attack_config.get('order', 'random')
        
        variant = "SimBA-pixel" if pixel_attack else "SimBA-DCT"
        print(f"Variant: {variant}")
        print(f"Order: {order}")
        if not pixel_attack and freq_dims:
            print(f"Frequency dimensions: {freq_dims}")
        
        attacker = SimBA(
            model=victim_model,
            epsilon=epsilon,
            max_queries=max_queries,
            freq_dims=freq_dims,
            order=order,
            targeted=targeted,
            pixel_attack=pixel_attack,
            device=device
        )
    else:
        raise ValueError(f"Unknown attack method: {method}")
    
    results = {
        'success_count': 0,
        'total_samples': len(samples),
        'query_counts': [],
        'gradient_query_counts': [],
        'coimage_query_counts': [],
        'perturbation_norms': [],
        'times': [],
        'failed_indices': []
    }
    
    for i, (img, true_label) in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Attacking image...", end=' ')
        
        start_time = time.time()
        x_adv, stats = attacker.attack(img, true_label)
        elapsed_time = time.time() - start_time
        
        perturbation_norm = torch.norm(x_adv - img).item()
        
        results['query_counts'].append(stats['total_queries'])
        results['perturbation_norms'].append(perturbation_norm)
        results['times'].append(elapsed_time)
        
        # Handle different stat formats
        if method == 'gfcs':
            results['gradient_query_counts'].append(stats['gradient_queries'])
            results['coimage_query_counts'].append(stats['coimage_queries'])
        else:
            # SimBA doesn't have these, use 0
            results['gradient_query_counts'].append(0)
            results['coimage_query_counts'].append(0)
        
        if stats['success']:
            results['success_count'] += 1
            if method == 'gfcs':
                print(f"✓ SUCCESS - Q:{stats['total_queries']}, "
                      f"Grad:{stats['gradient_queries']}, ODS:{stats['coimage_queries']}, "
                      f"L2:{perturbation_norm:.2f}, Time:{elapsed_time:.2f}s")
            else:
                print(f"✓ SUCCESS - Q:{stats['total_queries']}, "
                      f"L2:{perturbation_norm:.2f}, Time:{elapsed_time:.2f}s")
        else:
            results['failed_indices'].append(i)
            print(f"✗ FAILED - Q:{stats['total_queries']}, "
                  f"L2:{perturbation_norm:.2f}, Time:{elapsed_time:.2f}s")
    
    # Compute aggregate statistics
    results['success_rate'] = (results['success_count'] / results['total_samples']) * 100
    results['median_queries'] = float(np.median(results['query_counts']))
    results['mean_queries'] = float(np.mean(results['query_counts']))
    
    if method == 'gfcs':
        results['median_gradient_queries'] = float(np.median(results['gradient_query_counts']))
        results['median_coimage_queries'] = float(np.median(results['coimage_query_counts']))
    
    results['mean_perturbation_norm'] = float(np.mean(results['perturbation_norms']))
    results['mean_time'] = float(np.mean(results['times']))
    
    return results


def print_results(results: Dict[str, Any], experiment_id: str, description: str):
    """Print experiment results."""
    print(f"\n{'='*80}")
    print(f"RESULTS: {experiment_id}")
    print(f"Description: {description}")
    print(f"{'='*80}")
    print(f"Success Rate: {results['success_rate']:.2f}% ({results['success_count']}/{results['total_samples']})")
    
    # Check if this is minimal victim GFCS (has separate victim/surrogate queries)
    if 'victim_queries' in results and 'surrogate_queries' in results:
        print(f"\n--- Minimal Victim GFCS Statistics ---")
        print(f"Median Victim Queries: {results['median_victim_queries']:.1f}")
        print(f"Mean Victim Queries: {results['mean_victim_queries']:.2f}")
        print(f"Median Surrogate Queries: {results['median_surrogate_queries']:.1f}")
        print(f"Mean Surrogate Queries: {results['mean_surrogate_queries']:.2f}")
        print(f"Median Iterations: {results['median_iterations']:.1f}")
        print(f"Mean Iterations: {results['mean_iterations']:.2f}")
    else:
        # Standard GFCS or SimBA
        print(f"Median Queries: {results['median_queries']:.0f}")
        print(f"Mean Queries: {results['mean_queries']:.1f}")
        
        # Only print gradient/coimage stats if they exist (GFCS only)
        if 'median_gradient_queries' in results:
            print(f"Median Gradient Queries: {results['median_gradient_queries']:.0f}")
            print(f"Median Coimage Queries: {results['median_coimage_queries']:.0f}")
    
    print(f"\nMean L2 Norm: {results['mean_perturbation_norm']:.2f}")
    print(f"Mean Time per Image: {results['mean_time']:.2f}s")
    print(f"Total Time: {sum(results['times']):.2f}s ({sum(results['times'])/60:.1f} minutes)")
    if len(results['failed_indices']) > 0:
        print(f"Failed Image Indices: {results['failed_indices'][:10]}{'...' if len(results['failed_indices']) > 10 else ''}")
    print(f"{'='*80}\n")


def save_results(results: Dict[str, Any], config: Dict[str, Any], output_dir: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    experiment_id = config['experiment_id']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_data = {
        'config': config,
        'results': results,
        'timestamp': timestamp
    }
    
    filename = f"{experiment_id}_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {filepath}")
    return filepath


def list_experiments(config_dir: str):
    """List all available experiment configurations."""
    config_files = sorted(Path(config_dir).glob("exp_*.json"))
    
    if not config_files:
        print(f"No experiment configurations found in {config_dir}")
        return
    
    print(f"\nAvailable Experiments in {config_dir}:")
    print(f"{'='*80}")
    
    for config_file in config_files:
        try:
            config = load_config(str(config_file))
            exp_id = config.get('experiment_id', 'unknown')
            desc = config.get('description', 'No description')
            victim = config.get('victim', {}).get('model_name', 'unknown')
            dataset = config.get('dataset', {}).get('name', 'unknown')
            method = config.get('attack', {}).get('method', 'unknown')
            
            print(f"\n{exp_id}: {config_file.name}")
            print(f"  Description: {desc}")
            print(f"  Method: {method}, Victim: {victim}, Dataset: {dataset}")
        except Exception as e:
            print(f"\n{config_file.name}: ERROR - {str(e)}")
    
    print(f"\n{'='*80}\n")


def run_experiment_from_config(config_path: str, device: str, output_dir: str):
    """
    Run a single experiment from configuration file.
    
    Args:
        config_path: Path to configuration JSON file
        device: Device to use
        output_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"Loading experiment configuration: {config_path}")
    print(f"{'='*80}")
    
    # Load and validate config
    config = load_config(config_path)
    errors = validate_config(config)
    
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return None
    
    experiment_id = config['experiment_id']
    description = config.get('description', 'No description')
    method = config['attack']['method']
    
    print(f"Experiment ID: {experiment_id}")
    print(f"Description: {description}")
    print(f"Attack Method: {method}")
    
    # Set seed
    seed = config['dataset']['seed']
    set_seed(seed)
    print(f"Random seed set to: {seed}")
    
    # Load victim model
    print(f"\n{'-'*80}")
    print("LOADING VICTIM MODEL")
    print(f"{'-'*80}")
    victim_model = load_model(config['victim'], device)
    
    # Load surrogate models (for GFCS and GFCS Minimal Victim)
    surrogate_models = []
    if method in ['gfcs', 'gfcs_minimal_victim']:
        print(f"\n{'-'*80}")
        print("LOADING SURROGATE MODELS")
        print(f"{'-'*80}")
        for i, surrogate_config in enumerate(config['surrogates']):
            print(f"Surrogate {i+1}/{len(config['surrogates'])}: ", end='')
            surrogate_model = load_model(surrogate_config, device)
            surrogate_models.append(surrogate_model)
    else:
        print(f"\n{'-'*80}")
        print("SIMBA ATTACK (No surrogates needed)")
        print(f"{'-'*80}")
    
    # Load dataset
    print(f"\n{'-'*80}")
    print("LOADING DATASET")
    print(f"{'-'*80}")
    samples = load_dataset(config['dataset'], device)
    
    # Filter correctly classified
    print(f"\n{'-'*80}")
    print("FILTERING CORRECTLY CLASSIFIED SAMPLES")
    print(f"{'-'*80}")
    samples_filtered = filter_correctly_classified(samples, victim_model)
    
    if len(samples_filtered) == 0:
        print("ERROR: No correctly classified samples found!")
        return None
    
    # Run attack
    print(f"\n{'-'*80}")
    print("RUNNING ATTACK")
    print(f"{'-'*80}")
    results = run_attack(
        samples_filtered,
        victim_model,
        surrogate_models,
        config['attack'],
        device
    )
    
    # Print results
    print_results(results, experiment_id, description)
    
    # Save results
    if config.get('output', {}).get('save_detailed_logs', True):
        save_results(results, config, output_dir)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run GFCS/SimBA/GFCS-Minimal-Victim experiments from configuration files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  python run_experiment_from_config.py exp_001
  
  # Run multiple experiments
  python run_experiment_from_config.py exp_001 exp_002 exp_003
  
  # Run comparison (original vs minimal victim)
  python run_experiment_from_config.py exp_001 exp_011
  
  # Use custom config directory
  python run_experiment_from_config.py --config_dir ./my_configs exp_001
  
  # List all available experiments
  python run_experiment_from_config.py --list
        """
    )
    
    parser.add_argument('experiments', nargs='*', help='Experiment IDs to run (e.g., exp_001)')
    parser.add_argument('--config_dir', type=str, default='./configs',
                        help='Directory containing experiment config files')
    parser.add_argument('--output_dir', type=str, default='./experiment_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--list', action='store_true',
                        help='List all available experiments and exit')
    
    args = parser.parse_args()
    
    # List experiments if requested
    if args.list:
        list_experiments(args.config_dir)
        return
    
    # Check if any experiments specified
    if not args.experiments:
        print("Error: No experiments specified. Use --list to see available experiments.")
        parser.print_help()
        return
    
    print("="*80)
    print("GFCS/SIMBA/MINIMAL-VICTIM EXPERIMENT RUNNER")
    print("="*80)
    print(f"Config directory: {args.config_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Experiments to run: {', '.join(args.experiments)}")
    if MINIMAL_VICTIM_AVAILABLE:
        print(f"Minimal Victim GFCS: ✓ Available")
    else:
        print(f"Minimal Victim GFCS: ✗ Not available (gfcs_minimal_victim_queries.py not found)")
    print("="*80)
    
    # Run each experiment
    all_results = []
    for exp_id in args.experiments:
        # Find config file
        config_file = None
        for ext in ['json']:
            potential_path = os.path.join(args.config_dir, f"{exp_id}.{ext}")
            if os.path.exists(potential_path):
                config_file = potential_path
                break
        
        if not config_file:
            # Try with full filename
            potential_path = os.path.join(args.config_dir, exp_id)
            if os.path.exists(potential_path):
                config_file = potential_path
        
        if not config_file:
            print(f"\nERROR: Configuration file not found for experiment: {exp_id}")
            print(f"Looked in: {args.config_dir}")
            continue
        
        # Run experiment
        try:
            result = run_experiment_from_config(config_file, args.device, args.output_dir)
            if result:
                all_results.append((exp_id, result))
        except Exception as e:
            print(f"\nERROR running experiment {exp_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print summary of all experiments
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("SUMMARY OF ALL EXPERIMENTS")
        print("="*80)
        
        # Check if we have minimal victim experiments
        has_minimal = any('victim_queries' in result for _, result in all_results)
        
        if has_minimal:
            print(f"{'Experiment':<20} {'Success Rate':<15} {'Victim Q':<12} {'Total Q':<12} {'Mean L2':<10}")
            print("-"*80)
            for exp_id, result in all_results:
                if 'victim_queries' in result:
                    # Minimal victim GFCS
                    victim_q = f"{result['median_victim_queries']:.0f}"
                    total_q = f"{result['median_surrogate_queries']:.0f}"
                else:
                    # Original GFCS or SimBA
                    victim_q = f"{result['median_queries']:.0f}"
                    total_q = "N/A"
                
                print(f"{exp_id:<20} {result['success_rate']:>6.2f}%        "
                      f"{victim_q:>6}      {total_q:>6}      "
                      f"{result['mean_perturbation_norm']:>6.2f}")
        else:
            # Standard table
            print(f"{'Experiment':<20} {'Success Rate':<15} {'Median Queries':<15} {'Mean L2':<10}")
            print("-"*80)
            for exp_id, result in all_results:
                print(f"{exp_id:<20} {result['success_rate']:>6.2f}%        "
                      f"{result['median_queries']:>6.0f}          "
                      f"{result['mean_perturbation_norm']:>6.2f}")
        
        print("="*80)
    
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()