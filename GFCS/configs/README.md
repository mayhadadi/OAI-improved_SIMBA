# GFCS Experiment Configurations

This directory contains all experiment configurations for testing GFCS (Gradient First, Coimage Second) and related black-box adversarial attack methods.

## Overview

All experiments are defined in JSON format following the schema in [CONFIG_SCHEMA.md](CONFIG_SCHEMA.md). Each configuration specifies:
- Victim model and surrogate models
- Dataset and sampling parameters
- Attack method and hyperparameters
- Output settings

## Running Experiments

To run an experiment:

```bash
# Run single experiment
python run_experiment_from_config.py exp_001

# Run multiple experiments
python run_experiment_from_config.py exp_001 exp_002 exp_003

# List all available experiments
python run_experiment_from_config.py --list
```

## Experiment Categories

### 1. Standard GFCS Baseline (exp_001-003)
**Purpose:** Establish baseline performance of GFCS on Tiny ImageNet with different victim models

- **exp_001**: ResNet-50 victim, 4 surrogates (VGG-19, ResNet-34, DenseNet-121, MobileNet-v2)
- **exp_002**: VGG-16 victim, 4 surrogates (VGG-19, ResNet-34, DenseNet-121, MobileNet-v2)
- **exp_003**: Inception-v3 victim, 4 surrogates (VGG-19, ResNet-34, DenseNet-121, MobileNet-v2)

**Configuration:**
- Dataset: Tiny ImageNet, 2000 images
- Attack: GFCS, epsilon=2.0, max_queries=10000
- Norm bound: auto (sqrt(0.001*D))

---

### 2. CIFAR-10 Baseline (exp_004-006)
**Purpose:** Test GFCS on CIFAR-10 with CIFAR-10 pretrained models

- **exp_004**: ResNet-56 victim, 4 surrogates (ResNet-20, ResNet-32, ResNet-44, VGG-16)
- **exp_005**: VGG-19 victim, 4 surrogates (VGG-16, VGG-13, VGG-11, ResNet-56)
- **exp_006**: MobileNet-v2 victim, 4 surrogates (ResNet-56, ResNet-44, ResNet-32, VGG-19)

**Configuration:**
- Dataset: CIFAR-10, 2000 images
- Attack: GFCS, epsilon=2.0, max_queries=10000
- Image size: 32x32
- CIFAR-10 normalization: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]

---

### 3. SimBA Pixel-Based Attacks (exp_007-009)
**Purpose:** Compare GFCS with SimBA pixel-based attacks (surrogate-free black-box) on Tiny ImageNet

- **exp_007**: SimBA-pixel on ResNet-50
- **exp_008**: SimBA-pixel on VGG-16
- **exp_009**: SimBA-pixel on Inception-v3

**Configuration:**
- Dataset: Tiny ImageNet, 2000 images
- Attack: SimBA, epsilon=0.2, max_queries=20000
- Pixel-based perturbations
- No surrogates (black-box only)

---

### 4. CIFAR-10 Full Test Set (exp_010-012)
**Purpose:** Comprehensive evaluation on full CIFAR-10 test set (10,000 images)

- **exp_010**: ResNet-56 victim, 4 surrogates (ResNet-20, ResNet-32, ResNet-44, VGG-16)
- **exp_011**: VGG-19 victim, 4 surrogates (VGG-16, VGG-13, VGG-11, ResNet-56)
- **exp_012**: MobileNet-v2 victim, 4 surrogates (ResNet-56, ResNet-44, ResNet-32, VGG-19)

**Configuration:**
- Dataset: CIFAR-10, **10000 images** (full test set)
- Attack: GFCS, epsilon=2.0, max_queries=10000

---

### 5. SimBA DCT-Based Attacks (exp_013-015)
**Purpose:** SimBA DCT (frequency-domain) attacks on Tiny ImageNet

- **exp_013**: SimBA-DCT on ResNet-50 (frequency-domain perturbations, 12.5% of dimensions)
- **exp_014**: SimBA-DCT on VGG-16 (frequency-domain perturbations, 12.5% of dimensions)
- **exp_015**: SimBA-DCT on Inception-v3 (frequency-domain perturbations, 12.5% of dimensions)

**Configuration:**
- Dataset: Tiny ImageNet, 2000 images
- Attack: SimBA, epsilon=0.2, max_queries=20000
- DCT-based perturbations (freq_fraction=0.125)

---

### 6. GFCS Minimal Victim Queries (exp_016-018)
**Purpose:** Reduce victim model queries by using averaged gradients from surrogates and minimal validation

- **exp_016**: ResNet-50 victim, 4 surrogates, Tiny ImageNet
- **exp_017**: VGG-16 victim, 4 surrogates, Tiny ImageNet
- **exp_018**: Inception-v3 victim, 4 surrogates, Tiny ImageNet

**Configuration:**
- Attack: GFCS Minimal Victim
- max_iterations=1000 (iteration budget instead of query budget)
- Minimal victim queries (only 2 per iteration for validation)
- Uses averaged gradients from all surrogates

**Key Difference:** Traditional GFCS queries victim for every step; this variant minimizes victim queries by averaging surrogate gradients and only querying victim for validation.

---

### 7. GFCS with Varying Surrogate Count (exp_019-022)
**Purpose:** Study the effect of number of surrogates on attack success

- **exp_019**: Inception-v3 victim, **1 surrogate** (ResNet-34)
- **exp_020**: Inception-v3 victim, **2 surrogates** (ResNet-34, VGG-19)
- **exp_021**: Inception-v3 victim, **7 surrogates** (VGG-19, VGG-16, ResNet-152, ResNet-50, ResNet-34, DenseNet-121, MobileNet-v2)
- **exp_022**: Inception-v3 victim, **10 surrogates** (all available pretrained ImageNet models)

**Configuration:**
- Dataset: Tiny ImageNet, 2000 images
- Attack: GFCS, epsilon=2.0, max_queries=10000

**Research Question:** Does having more surrogates improve attack efficiency?

---

### 8. GFCS Minimal Victim - Extended Surrogates (exp_023-024)
**Purpose:** Test minimal victim variant with more surrogates on smaller dataset

- **exp_023**: Inception-v3 victim, **7 surrogates**, 50 images
- **exp_024**: Inception-v3 victim, **10 surrogates**, 50 images

**Configuration:**
- Dataset: Tiny ImageNet, **50 images** (quick test)
- Attack: GFCS Minimal Victim, max_iterations=1000
- Minimal victim queries (only 2), averaged gradients

---

### 9. GFCS Averaged Gradients (exp_025-027)
**Purpose:** Use averaged gradients from all surrogates instead of random selection

- **exp_025**: Inception-v3 victim, 7 surrogates
- **exp_026**: Inception-v3 victim, 10 surrogates
- **exp_027**: Inception-v3 victim, 4 surrogates

**Configuration:**
- Dataset: Tiny ImageNet, 2000 images
- Attack: GFCS Averaged Gradients
- max_iterations=1000
- Direction selection: averaged gradients from all surrogates

**Key Difference:** Standard GFCS randomly picks one surrogate per iteration; this variant averages all surrogate gradients for potentially more robust directions.

---

### 10. GFCS Adaptive (exp_028)
**Purpose:** Test adaptive variant of GFCS

- **exp_028**: Inception-v3 victim, 4 surrogates, 2000 images

**Configuration:**
- Attack: GFCS Adaptive
- epsilon=2.0, max_queries=10000

**Note:** Adaptive variant adjusts strategy based on attack progress.

---

### 11. GFCS Weighted Alignment (exp_029-034)
**Purpose:** Trust-based weighting of surrogates and smart ODS fallback

This variant adaptively weights surrogate gradients based on their historical alignment with successful directions and uses momentum-based ODS when gradients fail.

- **exp_029**: Inception-v3 victim, 4 surrogates, **500 images** (initial test)
- **exp_030**: Inception-v3 victim, 4 surrogates, 2000 images
- **exp_031**: Inception-v3 victim, 2 surrogates (ResNet-34, VGG-19), 2000 images
- **exp_032**: Inception-v3 victim, 7 surrogates, 2000 images
- **exp_033**: Inception-v3 victim, **10 surrogates**, 2000 images
- **exp_034**: Inception-v3 victim, **1 surrogate** (ResNet-34), 2000 images

**Configuration:**
- Attack: GFCS Weighted Alignment
- epsilon=2.0, max_queries=10000
- Trust-based weighting: surrogates with better historical performance get higher weight
- Smart ODS: uses momentum from previous successful directions

**Research Question:** Can adaptive trust-based weighting improve query efficiency over random surrogate selection?

---

## Key Parameters Summary

### Attack Methods

| Method | Query Type | Typical Budget | Surrogate Use |
|--------|-----------|----------------|---------------|
| GFCS | Victim queries | 10000 | Gradient + ODS fallback |
| SimBA | Victim queries | 20000 | None (black-box only) |
| GFCS Minimal Victim | Iterations | 1000 | Averaged gradients, minimal victim validation |
| GFCS Averaged Gradients | Iterations | 1000 | Averaged from all surrogates |
| GFCS Adaptive | Victim queries | 10000 | Adaptive strategy |
| GFCS Weighted Alignment | Victim queries | 10000 | Trust-based weighting + smart ODS |

### Common Settings

- **Random Seed:** 42 (all experiments for reproducibility)
- **Untargeted attacks:** All experiments use untargeted attacks
- **Norm bound:** auto (sqrt(0.001 * D)) for GFCS variants
- **Gradient normalization:** L2 (all GFCS variants)
- **ODS sampling:** Uniform(-1, 1) for standard GFCS

### Datasets

| Dataset | Images | Image Size | Normalization |
|---------|--------|------------|---------------|
| Tiny ImageNet | 50-2000 | 224x224 | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| CIFAR-10 | 2000-10000 | 32x32 | mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616] |

### Epsilon Values

- **GFCS variants:** epsilon = 2.0
- **SimBA:** epsilon = 0.2

---

## Experiment Naming Convention

Experiments are numbered sequentially (exp_001 through exp_034):

- **001-003:** Tiny ImageNet baseline (different victims)
- **004-006:** CIFAR-10 baseline
- **007-009:** SimBA pixel-based on Tiny ImageNet
- **010-012:** CIFAR-10 full test set
- **013-015:** SimBA DCT-based on Tiny ImageNet
- **016-018:** Minimal victim queries
- **019-022:** Varying surrogate count
- **023-024:** Minimal victim extended
- **025-027:** Averaged gradients
- **028:** Adaptive variant
- **029-034:** Weighted alignment with varying surrogate counts

---

## Research Questions Addressed

1. **Baseline Performance:** How does GFCS perform on different architectures? (exp_001-003)
2. **Dataset Transfer:** Does GFCS work well on CIFAR-10? (exp_004-006, exp_010-012)
3. **SimBA Comparison:** How does GFCS compare to surrogate-free SimBA? (exp_007-009, exp_013-015)
4. **Query Efficiency:** Can we reduce victim queries using surrogate averaging? (exp_016-018, exp_023-024)
5. **Surrogate Count:** Does increasing surrogates improve performance? (exp_019-022)
6. **Gradient Averaging:** Is averaging all surrogates better than random selection? (exp_025-027)
7. **Trust-Based Weighting:** Can adaptive weighting improve efficiency? (exp_029-034)

---

## Output Files

Results are saved to `./experiment_results/` with filenames:
```
{experiment_id}_results_{timestamp}.json
```

Each results file contains:
- Full configuration used
- Per-image statistics (queries, success, perturbation norm)
- Aggregate statistics (success rate, median/mean queries)
- Timestamp and metadata

---

## Common Surrogate Combinations

### 4 Surrogates (Standard)
- VGG-19, ResNet-34, DenseNet-121, MobileNet-v2

### 7 Surrogates
- VGG-19, VGG-16, ResNet-152, ResNet-50, ResNet-34, DenseNet-121, MobileNet-v2

### 10 Surrogates (All Available)
- VGG-19, VGG-16, ResNet-152, ResNet-50, ResNet-34, DenseNet-121, MobileNet-v2, AlexNet, SqueezeNet, ShuffleNet

### CIFAR-10 Surrogates
- VGG-13, VGG-16, ResNet-20, ResNet-32, ResNet-44, ResNet-56

---

## Notes

- **Tiny ImageNet** is used as a drop-in replacement for full ImageNet (200 classes, 64x64 images upscaled to 224x224)
- **Auto-download:** CIFAR-10 and Tiny ImageNet are automatically downloaded when needed
- **Reproducibility:** All experiments use seed=42 for deterministic sampling
- **Output:** By default, only detailed logs are saved (not adversarial images or perturbations to save disk space)

---

## Configuration Schema

For detailed information about the configuration format, see [CONFIG_SCHEMA.md](CONFIG_SCHEMA.md).

To validate a config:
```bash
python run_experiment_from_config.py --list
```

This will list all experiments and validate their configurations.
