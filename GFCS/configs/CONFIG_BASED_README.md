# GFCS Configuration-Based Experiment System

A flexible, scalable experiment system using JSON configuration files for GFCS adversarial attacks.

## Overview

This system allows you to:
- Define experiments as JSON configuration files
- Run experiments by specifying config IDs
- Easily add new experiments without changing code
- Support future extensions (averaged gradients, custom norm bounds, etc.)
- Maintain full reproducibility with seeds
- Track all experimental parameters

## Quick Start

### 1. List Available Experiments

```bash
python run_experiment_from_config.py --list
```

This shows all experiments in the `experiment_configs/` directory.

### 2. Run Single Experiment

```bash
python run_experiment_from_config.py exp_001
```

### 3. Run Multiple Experiments

```bash
python run_experiment_from_config.py exp_001 exp_002 exp_003
```

### 4. Run All Experiments (Baseline Suite)

```bash
# ImageNet experiments
python run_experiment_from_config.py exp_001 exp_002 exp_003

# CIFAR-10 experiments
python run_experiment_from_config.py exp_004 exp_005 exp_006

# ImageNet-R experiments
python run_experiment_from_config.py exp_007 exp_008 exp_009
```

## Experiment Configuration Files

All experiments are defined as JSON files in `experiment_configs/`:

```
experiment_configs/
├── CONFIG_SCHEMA.md                      # Full schema documentation
├── exp_001_baseline_resnet50_imagenet.json
├── exp_002_baseline_vgg16_imagenet.json
├── exp_003_baseline_inception_imagenet.json
├── exp_004_cifar10_resnet50.json
├── exp_005_cifar10_vgg16.json
├── exp_006_cifar10_inception.json
├── exp_007_imagenet_r_resnet50.json
├── exp_008_imagenet_r_vgg16.json
└── exp_009_imagenet_r_inception.json
```

## Pre-defined Experiments

| Experiment | Dataset | Victim | Surrogates | Description |
|------------|---------|--------|------------|-------------|
| exp_001 | ImageNet | ResNet-50 | 4 (VGG-19, ResNet-34, DenseNet-121, MobileNet-v2) | Baseline |
| exp_002 | ImageNet | VGG-16 | 4 | Baseline |
| exp_003 | ImageNet | Inception-v3 | 4 | Baseline |
| exp_004 | CIFAR-10 | ResNet-50 | 4 | Transfer learning test |
| exp_005 | CIFAR-10 | VGG-16 | 4 | Transfer learning test |
| exp_006 | CIFAR-10 | Inception-v3 | 4 | Transfer learning test |
| exp_007 | ImageNet-R | ResNet-50 | 4 | Robustness test |
| exp_008 | ImageNet-R | VGG-16 | 4 | Robustness test |
| exp_009 | ImageNet-R | Inception-v3 | 4 | Robustness test |

All experiments use:
- **2000 images** (seed 42)
- **ε = 2.0** (step size)
- **Max queries = 10,000**
- **Norm bound = auto** (√(0.001×D))

## Configuration File Structure

### Minimal Example

```json
{
  "experiment_id": "exp_010",
  "description": "My custom experiment",
  
  "victim": {
    "model_name": "resnet50",
    "num_classes": 1000,
    "normalization": {
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  },
  
  "surrogates": [
    {
      "model_name": "vgg19",
      "num_classes": 1000,
      "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
      }
    }
  ],
  
  "dataset": {
    "name": "imagenet",
    "path": "./data/imagenet/val",
    "num_images": 2000,
    "seed": 42,
    "image_size": 224
  },
  
  "attack": {
    "method": "gfcs",
    "epsilon": 2.0,
    "max_queries": 10000,
    "targeted": false,
    "norm_bound": {
      "type": "auto",
      "value": null
    },
    "direction_selection": {
      "method": "standard",
      "gradient_normalization": "l2",
      "ods_sampling": "uniform"
    }
  },
  
  "output": {
    "save_adversarial_examples": false,
    "save_perturbations": false,
    "save_detailed_logs": true
  }
}
```

See `CONFIG_SCHEMA.md` for complete documentation.

## Creating New Experiments

### Example: Custom Epsilon Test

Create `experiment_configs/exp_010_custom_epsilon.json`:

```json
{
  "experiment_id": "exp_010",
  "description": "Test with smaller epsilon (1.0)",
  
  "victim": {
    "model_name": "resnet50",
    "num_classes": 1000,
    "normalization": {
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  },
  
  "surrogates": [
    {"model_name": "vgg19", "num_classes": 1000, 
     "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
    {"model_name": "resnet34", "num_classes": 1000,
     "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
    {"model_name": "densenet121", "num_classes": 1000,
     "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
    {"model_name": "mobilenet_v2", "num_classes": 1000,
     "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
  ],
  
  "dataset": {
    "name": "imagenet",
    "path": "./data/imagenet/val",
    "num_images": 500,
    "seed": 42,
    "image_size": 224
  },
  
  "attack": {
    "method": "gfcs",
    "epsilon": 1.0,
    "max_queries": 10000,
    "targeted": false,
    "norm_bound": {"type": "auto", "value": null},
    "direction_selection": {
      "method": "standard",
      "gradient_normalization": "l2",
      "ods_sampling": "uniform"
    }
  },
  
  "output": {
    "save_adversarial_examples": false,
    "save_perturbations": false,
    "save_detailed_logs": true
  }
}
```

Then run:
```bash
python run_experiment_from_config.py exp_010
```

## Command Line Options

```
python run_experiment_from_config.py [OPTIONS] experiment_id [experiment_id ...]

Positional arguments:
  experiments          Experiment IDs to run (e.g., exp_001 exp_002)

Options:
  --config_dir DIR     Directory with config files (default: ./experiment_configs)
  --output_dir DIR     Directory to save results (default: ./experiment_results)
  --device DEVICE      Device to use: cuda or cpu (default: cuda)
  --list               List all available experiments and exit
  -h, --help           Show help message
```

### Examples

**List all experiments:**
```bash
python run_experiment_from_config.py --list
```

**Run on CPU:**
```bash
python run_experiment_from_config.py --device cpu exp_001
```

**Custom config directory:**
```bash
python run_experiment_from_config.py --config_dir ./my_experiments exp_001
```

**Custom output directory:**
```bash
python run_experiment_from_config.py --output_dir ./results_2025 exp_001
```

## Output Structure

After running experiments, results are saved to `experiment_results/`:

```
experiment_results/
├── exp_001_results_20250115_143022.json
├── exp_002_results_20250115_150433.json
└── ...
```

Each result file contains:
```json
{
  "config": { ... },  // Complete experiment configuration
  "results": {
    "success_rate": 99.85,
    "median_queries": 4,
    "mean_queries": 6.2,
    "median_gradient_queries": 3,
    "median_coimage_queries": 1,
    "query_counts": [4, 5, 3, ...],
    "gradient_query_counts": [3, 4, 2, ...],
    "coimage_query_counts": [1, 1, 1, ...],
    ...
  },
  "timestamp": "20250115_143022"
}
```

## Future Extensions

The configuration system is designed to support future research:

### 1. Averaged Gradient Direction

Create a config with:
```json
"attack": {
  "method": "gfcs",
  "direction_selection": {
    "method": "averaged",
    "gradient_normalization": "l2"
  }
}
```

Then implement in `gfcs.py`:
```python
if config['direction_selection']['method'] == 'averaged':
    # Average gradients from all surrogates
    gradients = [self.get_surrogate_gradient(x, s, ...) for s in self.surrogates]
    q = sum(gradients) / len(gradients)
    q = q / torch.norm(q)  # Normalize
```

### 2. Weighted Surrogate Combination

```json
"attack": {
  "direction_selection": {
    "method": "weighted",
    "weighting_strategy": "confidence"
  }
}
```

### 3. Custom Norm Bounds

```json
"attack": {
  "norm_bound": {
    "type": "fixed",
    "value": 5.0
  }
}
```

### 4. Different Gradient Normalizations

```json
"attack": {
  "direction_selection": {
    "gradient_normalization": "l1"  // or "linf", "none"
  }
}
```

## Reproducibility

All experiments use:
- **Fixed seeds** for data sampling
- **Deterministic PyTorch operations**
- **Saved configurations** with results

To reproduce experiment `exp_001`:
```bash
python run_experiment_from_config.py exp_001
```

The same seed (42) ensures the same 2000 images are selected every time.

## Best Practices

### Naming Conventions

Use descriptive experiment IDs:
```
exp_NNN_<variant>_<victim>_<dataset>.json

Examples:
exp_010_epsilon1.0_resnet50_imagenet.json
exp_011_averaged_gradients_vgg16_imagenet.json
exp_012_fixed_norm_inception_cifar10.json
```

### Organizing Experiments

Group related experiments:
```
experiment_configs/
├── baseline/
│   ├── exp_001_baseline_resnet50_imagenet.json
│   ├── exp_002_baseline_vgg16_imagenet.json
│   └── ...
├── ablations/
│   ├── exp_020_no_gradient_resnet50_imagenet.json
│   ├── exp_021_no_ods_resnet50_imagenet.json
│   └── ...
└── epsilon_sweep/
    ├── exp_030_epsilon0.5_resnet50_imagenet.json
    ├── exp_031_epsilon1.0_resnet50_imagenet.json
    └── ...
```

Then run:
```bash
python run_experiment_from_config.py --config_dir experiment_configs/baseline exp_001 exp_002
```

## Troubleshooting

### Config validation failed
Check the error message and compare your config to `CONFIG_SCHEMA.md`.

### Dataset path does not exist
Update the `dataset.path` field in your config:
```json
"dataset": {
  "path": "/correct/path/to/dataset"
}
```

### Model not found
Check supported models in `CONFIG_SCHEMA.md`:
- resnet50, resnet34, resnet152
- vgg16, vgg19
- inception_v3
- densenet121
- mobilenet_v2

### Out of memory
Reduce batch size or number of images:
```json
"dataset": {
  "num_images": 100
}
```

## Advanced Usage

### Running Batch Experiments

Create a shell script `run_all_baseline.sh`:
```bash
#!/bin/bash
python run_experiment_from_config.py exp_001 &
python run_experiment_from_config.py exp_002 &
python run_experiment_from_config.py exp_003 &
wait
echo "All experiments completed"
```

### Comparing Results

```python
import json
import pandas as pd

# Load results
with open('experiment_results/exp_001_results_*.json') as f:
    exp001 = json.load(f)

with open('experiment_results/exp_002_results_*.json') as f:
    exp002 = json.load(f)

# Compare
df = pd.DataFrame([
    {
        'Experiment': exp001['config']['experiment_id'],
        'Success Rate': exp001['results']['success_rate'],
        'Median Queries': exp001['results']['median_queries']
    },
    {
        'Experiment': exp002['config']['experiment_id'],
        'Success Rate': exp002['results']['success_rate'],
        'Median Queries': exp002['results']['median_queries']
    }
])

print(df)
```

## Summary

✅ **Flexible**: Easy to add new experiments  
✅ **Reproducible**: Fixed seeds + saved configs  
✅ **Scalable**: Run multiple experiments in batch  
✅ **Extensible**: Supports future modifications (averaged gradients, etc.)  
✅ **Organized**: All parameters in one place  
✅ **Version-controlled**: Config files can be tracked in git  

For questions or issues, see `CONFIG_SCHEMA.md` for detailed documentation.
