# Experiment Configuration Schema

This document describes the JSON configuration format for GFCS experiments.

## Configuration Fields

### Required Fields

```json
{
  "experiment_id": "exp_001",
  "description": "Human-readable description of the experiment",
  
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

## Field Descriptions

### `experiment_id`
- Type: `string`
- Description: Unique identifier for the experiment
- Example: `"exp_001"`, `"resnet50_imagenet_baseline"`

### `description`
- Type: `string`
- Description: Human-readable description of what this experiment tests
- Example: `"Baseline GFCS attack on ImageNet with ResNet-50 victim"`

### `victim`
Configuration for the victim (target) model.

#### `victim.model_name`
- Type: `string`
- Options: `"resnet50"`, `"vgg16"`, `"inception_v3"`, `"densenet121"`, `"mobilenet_v2"`, `"resnet34"`, `"vgg19"`, `"resnet152"`
- Description: Name of the victim model architecture

#### `victim.num_classes`
- Type: `integer`
- Description: Number of output classes
- Common values: `10` (CIFAR-10), `1000` (ImageNet)

#### `victim.normalization`
- Type: `object`
- Description: Input normalization parameters
- Fields:
  - `mean`: Array of 3 floats (RGB channel means)
  - `std`: Array of 3 floats (RGB channel standard deviations)

### `surrogates`
Array of surrogate model configurations. Each surrogate has the same structure as `victim`.

### `dataset`
Configuration for the dataset.

#### `dataset.name`
- Type: `string`
- Options: `"cifar10"`, `"imagenet"`, `"imagenet_r"`, `"cifar100"`, `"custom"`
- Description: Name of the dataset

#### `dataset.path`
- Type: `string` or `null`
- Description: Path to dataset directory (null for auto-download datasets like CIFAR-10)
- Example: `"./data/imagenet/val"`

#### `dataset.num_images`
- Type: `integer`
- Description: Number of images to use in the experiment
- Example: `2000`

#### `dataset.seed`
- Type: `integer`
- Description: Random seed for reproducible image selection
- Example: `42`

#### `dataset.image_size`
- Type: `integer`
- Description: Size to resize images to (square)
- Example: `224`

### `attack`
Configuration for the attack method.

#### `attack.method`
- Type: `string`
- Options: `"gfcs"`, `"gfcs_modified"`, `"simba_ods"`
- Description: Attack algorithm to use
- Note: For future extensions with modified GFCS variants

#### `attack.epsilon`
- Type: `float`
- Description: Step size for perturbations
- Default: `2.0` (as per paper)

#### `attack.max_queries`
- Type: `integer`
- Description: Maximum number of queries allowed per image
- Default: `10000`

#### `attack.targeted`
- Type: `boolean`
- Description: Whether to perform targeted attack
- Default: `false`

#### `attack.norm_bound`
Configuration for perturbation norm constraint.

##### `attack.norm_bound.type`
- Type: `string`
- Options: `"auto"`, `"fixed"`, `"none"`
- Description: How to compute norm bound
  - `"auto"`: Use `sqrt(0.001 * D)` where D is image dimension (paper default)
  - `"fixed"`: Use fixed value specified in `value`
  - `"none"`: No norm constraint

##### `attack.norm_bound.value`
- Type: `float` or `null`
- Description: Fixed norm bound value (used when `type="fixed"`)
- Example: `5.0`

#### `attack.direction_selection`
Configuration for how to select attack directions.

##### `attack.direction_selection.method`
- Type: `string`
- Options: `"standard"`, `"averaged"`, `"weighted"`, `"consensus"`
- Description: Method for selecting/combining directions
  - `"standard"`: Use single surrogate gradient (paper default)
  - `"averaged"`: Average gradients from multiple surrogates
  - `"weighted"`: Weighted combination based on surrogate confidence
  - `"consensus"`: Only use directions where surrogates agree
- Note: Only `"standard"` is currently implemented

##### `attack.direction_selection.gradient_normalization`
- Type: `string`
- Options: `"l2"`, `"l1"`, `"linf"`, `"none"`
- Description: How to normalize gradients
- Default: `"l2"` (paper default)

##### `attack.direction_selection.ods_sampling`
- Type: `string`
- Options: `"uniform"`, `"gaussian"`, `"adaptive"`
- Description: Distribution for ODS weight sampling
- Default: `"uniform"` (paper default: U(-1, 1))

### `output`
Configuration for what to save.

#### `output.save_adversarial_examples`
- Type: `boolean`
- Description: Whether to save adversarial images
- Default: `false` (saves disk space)

#### `output.save_perturbations`
- Type: `boolean`
- Description: Whether to save perturbations separately
- Default: `false`

#### `output.save_detailed_logs`
- Type: `boolean`
- Description: Whether to save detailed per-query logs
- Default: `true`

## Example Configurations

See the `experiment_configs/` directory for example configurations:

1. `exp_001_baseline_resnet50_imagenet.json` - Baseline GFCS on ImageNet
2. `exp_002_baseline_vgg16_imagenet.json` - VGG-16 variant
3. `exp_003_baseline_inception_imagenet.json` - Inception-v3 variant
4. `exp_004_cifar10_resnet50.json` - CIFAR-10 experiments
5. `exp_005_imagenet_r_resnet50.json` - ImageNet-R robustness test

## Future Extensions

The configuration system is designed to support future research directions:

### Direction Selection Methods
- **Averaged gradients**: `"direction_selection.method": "averaged"`
  - Average gradients from all surrogates
  - Potential for more robust directions

- **Weighted combination**: `"direction_selection.method": "weighted"`
  - Weight surrogates by their confidence/agreement
  - Requires additional fields for weighting strategy

- **Consensus-based**: `"direction_selection.method": "consensus"`
  - Only use directions where surrogates agree (e.g., cosine similarity > threshold)
  - Requires additional fields for consensus threshold

### Custom Norm Bounds
- Test different norm constraints
- Dataset-specific norm bounds
- Adaptive norm bounds

### Additional Attack Variants
- Different ODS sampling distributions
- Gradient normalization variants
- Adaptive epsilon scheduling

## Validation

The experiment runner validates all configuration files and will report errors if:
- Required fields are missing
- Invalid model names are specified
- Invalid enum values are used
- Dataset paths don't exist (when required)
