# Experiment Configuration Schema

This document describes the JSON configuration format for GFCS experiments.

## Configuration Fields

### Required Fields

```json
{
  "experiment_id": "exp_001",
  "description": "Human-readable description of the experiment",
  "surr_desc": "Description of surrogate models used",
  "data_desc": "Description of dataset configuration",
  "addition_desc": "Additional experimental parameters and notes",

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

### SimBA Attack Example

```json
{
  "experiment_id": "exp_007",
  "description": "SimBA pixel-based attack",
  "surr_desc": "No surrogates (black-box only)",
  "data_desc": "2000 Tiny ImageNet validation images",
  "addition_desc": "SimBA-pixel baseline",

  "victim": {
    "model_name": "resnet50",
    "num_classes": 200,
    "normalization": {
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  },

  "surrogates": [],

  "dataset": {
    "name": "tiny_imagenet",
    "path": "./data/tiny_imagenet",
    "num_images": 2000,
    "seed": 42,
    "image_size": 224
  },

  "attack": {
    "method": "simba",
    "epsilon": 0.2,
    "max_queries": 20000,
    "targeted": false,
    "pixel_attack": true,
    "freq_dims": null,
    "order": "random"
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

### `surr_desc`
- Type: `string`
- Description: Description of the surrogate models used in the experiment
- Example: `"4 surrogates: VGG19, ResNet34, DenseNet121, MobileNetV2"`

### `data_desc`
- Type: `string`
- Description: Description of the dataset configuration
- Example: `"2000 Tiny ImageNet validation images"`

### `addition_desc`
- Type: `string`
- Description: Additional experimental parameters, notes, or special configurations
- Example: `"Testing gradient transfer efficiency with multiple surrogates"`

### `victim`
Configuration for the victim (target) model.

#### `victim.model_name`
- Type: `string`
- Options:
  - **ImageNet models:** `"resnet50"`, `"resnet34"`, `"resnet152"`, `"vgg16"`, `"vgg19"`, `"inception_v3"`, `"densenet121"`, `"mobilenet_v2"`, `"alexnet"`, `"squeezenet"`, `"shufflenet"`
  - **CIFAR-10 models:** `"resnet20"`, `"resnet32"`, `"resnet44"`, `"resnet56"`, `"vgg11"`, `"vgg13"`, `"vgg16"`, `"vgg19"`
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
- Can be empty array `[]` for black-box only attacks that don't use surrogates (e.g., SimBA)
- For GFCS variants, typically contains 1-4 surrogate models

### `dataset`
Configuration for the dataset.

#### `dataset.name`
- Type: `string`
- Options: `"cifar10"`, `"imagenet"`, `"tiny_imagenet"`, `"imagenet_r"`, `"cifar100"`, `"custom"`
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
- Options:
  - `"gfcs"` - Standard GFCS (Gradient First, Coimage Second)
  - `"simba"` - SimBA (Simple Black-box Adversarial attacks)
  - `"gfcs_minimal_victim"` - GFCS variant minimizing victim queries
  - `"gfcs_averaged_gradients"` - GFCS with averaged gradients from all surrogates
  - `"gfcs_adaptive"` - Adaptive GFCS variant
  - `"gfcs_weighted_alignment"` - GFCS with trust-based weighting and smart ODS
- Description: Attack algorithm to use

#### `attack.epsilon`
- Type: `float`
- Description: Step size for perturbations
- Default: `2.0` (as per paper)

#### `attack.max_queries` or `attack.max_iterations`
- Type: `integer`
- Description: Maximum budget allowed per image
  - Use `max_queries` for standard GFCS and SimBA (counts victim model queries)
  - Use `max_iterations` for GFCS variants like `gfcs_minimal_victim`, `gfcs_averaged_gradients` (counts iterations)
- Default: `10000` for GFCS, `20000` for SimBA, `1000` for iteration-based variants

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

#### SimBA-Specific Attack Parameters

These parameters are only used when `attack.method = "simba"`:

##### `attack.pixel_attack`
- Type: `boolean`
- Description: Whether to use pixel-based attack (true) or DCT-based attack (false)
- Default: `true`

##### `attack.freq_dims`
- Type: `integer` or `null`
- Description: Number of frequency dimensions to use for DCT-based attack
- Only used when `pixel_attack = false`
- Default: `null` (auto-computed based on freq_fraction)

##### `attack.freq_fraction`
- Type: `float`
- Description: Fraction of total frequency dimensions to use for DCT
- Common value: `0.125` (12.5% of dimensions)
- Only used when `pixel_attack = false`

##### `attack.order`
- Type: `string`
- Options: `"random"`, `"sequential"`
- Description: Order of dimension perturbation
- Default: `"random"`

#### `attack.direction_selection`
Configuration for how to select attack directions.

##### `attack.direction_selection.method`
- Type: `string`
- Options: `"standard"`, `"averaged_gradients"`
- Description: Method for selecting/combining directions
  - `"standard"`: Use single surrogate gradient (paper default)
  - `"averaged_gradients"`: Average gradients from multiple surrogates
- Note: Most configs use `"standard"`

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

See the `configs/` directory for all experiment configurations. Key examples:

### Standard GFCS (exp_001-006, exp_010-012, exp_019-022)
- `exp_001.json` - ResNet-50 victim, 4 surrogates, Tiny ImageNet
- `exp_004.json` - ResNet-20 victim, 4 surrogates, CIFAR-10
- `exp_010.json` - ResNet-32 victim, full CIFAR-10 test set (10000 images)

### SimBA Attacks (exp_007-015)
- `exp_007.json` - SimBA-pixel on ResNet-50, Tiny ImageNet
- `exp_008.json` - SimBA-DCT on ResNet-50, Tiny ImageNet
- `exp_013.json` - SimBA-pixel on ResNet-32, CIFAR-10

### GFCS Variants
- `exp_016.json` - GFCS Minimal Victim (reduces victim queries)
- `exp_025.json` - GFCS Averaged Gradients
- `exp_028.json` - GFCS Adaptive
- `exp_029.json` - GFCS Weighted Alignment (trust-based weighting)

## Common Configuration Patterns

### Normalization Values
- **ImageNet/Tiny ImageNet:** `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- **CIFAR-10:** `mean=[0.4914, 0.4822, 0.4465]`, `std=[0.2470, 0.2435, 0.2616]`

### Epsilon Values
- **GFCS attacks:** `epsilon = 2.0`
- **SimBA attacks:** `epsilon = 0.2`

### Image Sizes
- **CIFAR-10:** `image_size = 32`
- **ImageNet/Tiny ImageNet:** `image_size = 224`

### Seeds
- All experiments use `seed = 42` for reproducibility

## Attack Method Comparison

| Method | Victim Queries | Surrogate Use | Typical Epsilon | Typical Budget |
|--------|---------------|---------------|-----------------|----------------|
| GFCS | High | Gradient + ODS | 2.0 | 10000 queries |
| SimBA | High | None | 0.2 | 20000 queries |
| GFCS Minimal Victim | Low | Gradient validation | 2.0 | 1000 iterations |
| GFCS Averaged Gradients | Medium | Averaged gradients | 2.0 | 1000 iterations |
| GFCS Weighted Alignment | Medium | Trust-based weighting | 2.0 | 10000 queries |

## Validation

The experiment runner validates all configuration files and will report errors if:
- Required fields are missing
- Invalid model names are specified
- Invalid enum values are used
- Dataset paths don't exist (when required)
