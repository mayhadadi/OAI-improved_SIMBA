"""
Config Generator Utility
========================
Helper script to generate experiment configuration files programmatically.

Usage:
    python generate_configs.py --help
    python generate_configs.py --epsilon_sweep 0.5 1.0 1.5 2.0 2.5
    python generate_configs.py --victim_sweep resnet50 vgg16 inception_v3
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


# Default configuration templates
IMAGENET_NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}

DEFAULT_SURROGATES = [
    {"model_name": "vgg19", "num_classes": 1000, "normalization": IMAGENET_NORMALIZATION},
    {"model_name": "resnet34", "num_classes": 1000, "normalization": IMAGENET_NORMALIZATION},
    {"model_name": "densenet121", "num_classes": 1000, "normalization": IMAGENET_NORMALIZATION},
    {"model_name": "mobilenet_v2", "num_classes": 1000, "normalization": IMAGENET_NORMALIZATION}
]


def create_base_config(
    experiment_id: str,
    description: str,
    victim_model: str,
    dataset_name: str,
    dataset_path: str = None,
    num_images: int = 2000,
    seed: int = 42,
    epsilon: float = 2.0,
    max_queries: int = 10000,
    surrogates: List[Dict] = None
) -> Dict[str, Any]:
    """Create a base configuration."""
    
    if surrogates is None:
        surrogates = DEFAULT_SURROGATES
    
    if dataset_path is None:
        if dataset_name == "cifar10":
            dataset_path = None
        elif dataset_name == "imagenet":
            dataset_path = "./data/imagenet/val"
        elif dataset_name == "imagenet_r":
            dataset_path = "./data/imagenet-r"
    
    config = {
        "experiment_id": experiment_id,
        "description": description,
        
        "victim": {
            "model_name": victim_model,
            "num_classes": 1000,
            "normalization": IMAGENET_NORMALIZATION
        },
        
        "surrogates": surrogates,
        
        "dataset": {
            "name": dataset_name,
            "path": dataset_path,
            "num_images": num_images,
            "seed": seed,
            "image_size": 224
        },
        
        "attack": {
            "method": "gfcs",
            "epsilon": epsilon,
            "max_queries": max_queries,
            "targeted": False,
            "norm_bound": {
                "type": "auto",
                "value": None
            },
            "direction_selection": {
                "method": "standard",
                "gradient_normalization": "l2",
                "ods_sampling": "uniform"
            }
        },
        
        "output": {
            "save_adversarial_examples": False,
            "save_perturbations": False,
            "save_detailed_logs": True
        }
    }
    
    return config


def save_config(config: Dict[str, Any], output_dir: str):
    """Save configuration to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"{config['experiment_id']}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created: {filepath}")
    return str(filepath)


def generate_epsilon_sweep(
    output_dir: str,
    epsilon_values: List[float],
    victim_model: str = "resnet50",
    dataset_name: str = "imagenet",
    start_id: int = 100
):
    """Generate configs for epsilon sweep."""
    print(f"\nGenerating epsilon sweep experiments...")
    print(f"Epsilon values: {epsilon_values}")
    print(f"Victim: {victim_model}, Dataset: {dataset_name}")
    
    for i, epsilon in enumerate(epsilon_values):
        exp_id = f"exp_{start_id + i:03d}"
        description = f"Epsilon sweep - ε={epsilon} - {victim_model} on {dataset_name}"
        
        config = create_base_config(
            experiment_id=exp_id,
            description=description,
            victim_model=victim_model,
            dataset_name=dataset_name,
            epsilon=epsilon
        )
        
        save_config(config, output_dir)


def generate_victim_sweep(
    output_dir: str,
    victim_models: List[str],
    dataset_name: str = "imagenet",
    start_id: int = 200
):
    """Generate configs for victim model sweep."""
    print(f"\nGenerating victim model sweep experiments...")
    print(f"Victims: {victim_models}")
    print(f"Dataset: {dataset_name}")
    
    for i, victim in enumerate(victim_models):
        exp_id = f"exp_{start_id + i:03d}"
        description = f"Victim sweep - {victim} on {dataset_name}"
        
        config = create_base_config(
            experiment_id=exp_id,
            description=description,
            victim_model=victim,
            dataset_name=dataset_name
        )
        
        save_config(config, output_dir)


def generate_dataset_sweep(
    output_dir: str,
    datasets: List[str],
    victim_model: str = "resnet50",
    start_id: int = 300
):
    """Generate configs for dataset sweep."""
    print(f"\nGenerating dataset sweep experiments...")
    print(f"Datasets: {datasets}")
    print(f"Victim: {victim_model}")
    
    for i, dataset_name in enumerate(datasets):
        exp_id = f"exp_{start_id + i:03d}"
        description = f"Dataset sweep - {victim_model} on {dataset_name}"
        
        config = create_base_config(
            experiment_id=exp_id,
            description=description,
            victim_model=victim_model,
            dataset_name=dataset_name
        )
        
        save_config(config, output_dir)


def generate_ablation_study(
    output_dir: str,
    victim_model: str = "resnet50",
    dataset_name: str = "imagenet",
    start_id: int = 400
):
    """Generate configs for ablation study."""
    print(f"\nGenerating ablation study experiments...")
    
    # Standard GFCS (baseline)
    exp_id = f"exp_{start_id:03d}"
    config = create_base_config(
        experiment_id=exp_id,
        description=f"Ablation - Standard GFCS (baseline) - {victim_model} on {dataset_name}",
        victim_model=victim_model,
        dataset_name=dataset_name
    )
    save_config(config, output_dir)
    
    # TODO: Add more ablation variants when implemented
    # For now, this shows the structure for future ablations:
    
    # Example: Only gradient (no ODS fallback) - would need code changes
    # exp_id = f"exp_{start_id + 1:03d}"
    # config = create_base_config(...)
    # config['attack']['direction_selection']['disable_ods'] = True
    # save_config(config, output_dir)
    
    # Example: Only ODS (no gradient) - would need code changes
    # exp_id = f"exp_{start_id + 2:03d}"
    # config = create_base_config(...)
    # config['attack']['direction_selection']['disable_gradient'] = True
    # save_config(config, output_dir)


def generate_norm_bound_sweep(
    output_dir: str,
    norm_bounds: List[float],
    victim_model: str = "resnet50",
    dataset_name: str = "imagenet",
    start_id: int = 500
):
    """Generate configs for norm bound sweep."""
    print(f"\nGenerating norm bound sweep experiments...")
    print(f"Norm bounds: {norm_bounds}")
    
    for i, norm_bound in enumerate(norm_bounds):
        exp_id = f"exp_{start_id + i:03d}"
        description = f"Norm bound sweep - ν={norm_bound} - {victim_model} on {dataset_name}"
        
        config = create_base_config(
            experiment_id=exp_id,
            description=description,
            victim_model=victim_model,
            dataset_name=dataset_name
        )
        
        # Override norm bound
        config['attack']['norm_bound'] = {
            "type": "fixed",
            "value": norm_bound
        }
        
        save_config(config, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Generate GFCS experiment configuration files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate epsilon sweep (0.5, 1.0, 1.5, 2.0, 2.5)
  python generate_configs.py --epsilon_sweep 0.5 1.0 1.5 2.0 2.5
  
  # Generate victim model sweep
  python generate_configs.py --victim_sweep resnet50 vgg16 inception_v3
  
  # Generate dataset sweep
  python generate_configs.py --dataset_sweep cifar10 imagenet imagenet_r
  
  # Generate norm bound sweep
  python generate_configs.py --norm_sweep 1.0 2.0 5.0 10.0
  
  # Generate ablation study
  python generate_configs.py --ablation
  
  # Combine multiple sweeps
  python generate_configs.py --epsilon_sweep 1.0 2.0 --victim_sweep resnet50 vgg16
        """
    )
    
    parser.add_argument('--output_dir', type=str, default='./experiment_configs',
                        help='Directory to save config files')
    parser.add_argument('--epsilon_sweep', nargs='+', type=float,
                        help='Generate epsilon sweep with specified values')
    parser.add_argument('--victim_sweep', nargs='+', type=str,
                        help='Generate victim model sweep with specified models')
    parser.add_argument('--dataset_sweep', nargs='+', type=str,
                        help='Generate dataset sweep with specified datasets')
    parser.add_argument('--norm_sweep', nargs='+', type=float,
                        help='Generate norm bound sweep with specified values')
    parser.add_argument('--ablation', action='store_true',
                        help='Generate ablation study configs')
    parser.add_argument('--start_id', type=int, default=100,
                        help='Starting experiment ID number')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GFCS CONFIG GENERATOR")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    generated_any = False
    
    if args.epsilon_sweep:
        generate_epsilon_sweep(args.output_dir, args.epsilon_sweep, start_id=args.start_id)
        generated_any = True
    
    if args.victim_sweep:
        generate_victim_sweep(args.output_dir, args.victim_sweep, start_id=args.start_id + 100)
        generated_any = True
    
    if args.dataset_sweep:
        generate_dataset_sweep(args.output_dir, args.dataset_sweep, start_id=args.start_id + 200)
        generated_any = True
    
    if args.norm_sweep:
        generate_norm_bound_sweep(args.output_dir, args.norm_sweep, start_id=args.start_id + 300)
        generated_any = True
    
    if args.ablation:
        generate_ablation_study(args.output_dir, start_id=args.start_id + 400)
        generated_any = True
    
    if not generated_any:
        print("\nNo generation options specified. Use --help to see available options.")
        parser.print_help()
    else:
        print("\n" + "="*80)
        print("CONFIG GENERATION COMPLETED")
        print("="*80)
        print(f"\nConfigs saved to: {args.output_dir}")
        print("\nTo run experiments:")
        print(f"  python run_experiment_from_config.py --config_dir {args.output_dir} exp_XXX")


if __name__ == "__main__":
    main()
