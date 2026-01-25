# GFCS: Gradient First, Coimage Second

Implementation of the GFCS black-box adversarial attack from:

> **"Attacking Deep Networks with Surrogate-Based Adversarial Black-Box Methods is Easy"**  
> Lord, Mueller & Bertinetto, ICLR 2022  
> https://github.com/fiveai/GFCS

## Overview

GFCS is a highly query-efficient black-box adversarial attack that leverages surrogate models. The key insight is that gradient transfer from surrogates typically succeeds - it just needs an occasional fallback mechanism.

### Algorithm Summary

```
1. Input: image x, victim v, surrogates S, step size ε, norm bound ν
2. Initialize: x_adv = x, S_rem = S
3. While not adversarial:
   a. If S_rem not empty:
      - Pick random surrogate s from S_rem (without replacement)
      - q = normalized loss gradient from s  [GRADIENT FIRST]
   b. Else:
      - Pick random surrogate s from S
      - q = random ODS direction from s      [COIMAGE SECOND]
   c. For α in {ε, -ε}:
      - If L_v(project(x_adv + α*q)) > L_v(x_adv):
        - Accept step, reset S_rem = S
        - break
4. Return x_adv
```

## Algorithm Variants

This repository implements several variants of GFCS to improve query efficiency and attack success:

### 1. Standard GFCS ([gfcs.py](gfcs.py))
The original algorithm from the paper:
- Randomly samples ONE surrogate at a time for gradient transfer
- Falls back to ODS when all surrogates exhausted
- Resets surrogate pool S_rem on successful steps
- **Query budget**: Measured in victim queries
- **Use case**: Baseline comparison

### 2. GFCS Averaged Gradients ([gfcs_averaged_gradients.py](gfcs_averaged_gradients.py))
**Key modification**: Averages gradients from ALL remaining surrogates instead of sampling one

```python
# Original GFCS
q = gradient_from_random_surrogate(S_rem)

# Averaged Gradients
q = average([gradient_from_surrogate(s) for s in S_rem])
```

- Uses ALL surrogates in S_rem at once, then empties S_rem
- More robust direction by combining multiple gradient signals
- Same query budget and structure as original GFCS
- **Hypothesis**: Averaged gradients transfer better than individual gradients

### 3. GFCS Minimal Victim Queries ([gfcs_minimal_victim_queries.py](gfcs_minimal_victim_queries.py))
**Key modification**: Minimizes victim queries to only 2 (start and end)

```python
# Query 1: Verify initial classification
victim_logits = victim(x)

# Attack loop: Only query surrogates
while not all_surrogates_fooled(x_adv):
    q = average_all_surrogate_gradients(x_adv)
    x_adv = update_with_surrogate_validation(x_adv, q)

# Query 2: Verify final success
final_logits = victim(x_adv)
```

- **Success criterion**: Fools ALL surrogates (not the victim during loop)
- Uses averaged gradients from all surrogates
- Switches to ODS after 100 failed gradient attempts
- **Budget**: Measured in iterations (not queries)
- **Assumption**: If all surrogates fooled → victim likely fooled
- **Trade-off**: Fewer victim queries but potentially lower success rate

### 4. GFCS Adaptive ([gfcs_adaptive.py](gfcs_adaptive.py))
**Key modifications**: Adaptive surrogate weighting + smart ODS fallback

**Adaptive Surrogate Weighting:**
- Maintains trust score for each surrogate
- Trust increases when surrogate's gradient leads to successful step
- Trust decreases when surrogate's gradient fails
- Gradients weighted by trust scores (higher trust = higher weight)

```python
# Update trust based on success
if step_successful:
    surrogate_trust[used_surrogate] += learning_rate
else:
    surrogate_trust[used_surrogate] *= decay_factor

# Weight gradients by trust
weights = softmax(surrogate_trust / temperature)
q = sum(weights[i] * gradients[i])
```

**Smart ODS Fallback:**
- Margin-aware class weight sampling (boost top-k classes, penalize true class)
- Momentum from previously successful ODS directions
- Tracks which weight patterns worked in the past

**Parameters:**
- `trust_learning_rate`: 0.1 (how fast trust adapts)
- `trust_decay`: 0.95 (decay for failed attempts)
- `ods_momentum`: 0.5 (momentum for ODS directions)

### 5. GFCS Weighted Alignment ([gfcs_weighted_alignment.py](gfcs_weighted_alignment.py))
**Key modifications**: Alignment-based trust + smart ODS (similar to Adaptive but different trust update)

**Alignment-Based Trust:**
- Computes all surrogate gradients
- Uses weighted combination based on trust
- **Updates trust by gradient alignment**: Surrogates whose gradients align with successful direction get increased trust

```python
# After successful step
for i, grad_i in enumerate(surrogate_gradients):
    alignment = cosine_similarity(grad_i, successful_direction)
    surrogate_trust[i] += learning_rate * alignment
```

- More nuanced than binary success/failure (Adaptive variant)
- Rewards surrogates proportional to their contribution
- Combines with smart ODS fallback (same as Adaptive)

**When to use each variant:**
- **Standard GFCS**: Baseline, paper reproduction
- **Averaged Gradients**: When you believe multiple surrogates provide complementary information
- **Minimal Victim**: When victim queries are expensive (rate-limited APIs, etc.)
- **Adaptive**: When you want to learn which surrogates work best
- **Weighted Alignment**: When you want fine-grained credit assignment

---

## Key Components (Standard GFCS)

### 1. Margin Loss
```python
L(x) = f(c_t) - f(c_s)  # second_highest - true_class
```
Maximizing this loss pushes the prediction away from the true class.

### 2. Direct Transfer (Gradient First)
Use the surrogate's loss gradient directly:
```python
q = ∇_x L_surrogate(x) / ||∇_x L_surrogate(x)||
```

### 3. ODS - Output Diversified Sampling (Coimage Second)
Sample from the row space of the Jacobian:
```python
w ~ Uniform(-1, 1)^C  # random class weights
q = ∇_x(w^T f(x)) / ||∇_x(w^T f(x))||
```
This explores directions the surrogate is sensitive to.

### 4. Projection (PGA)
Project onto the L2 ball around the original image:
```python
Π(x_adv) = x + ν * (x_adv - x) / ||x_adv - x||  if ||x_adv - x|| > ν
```

## Files

### Core Implementations
- [gfcs.py](gfcs.py) - Main GFCS implementation (Algorithm 1 from the paper)
- [SimBA.py](SimBA.py) - SimBA baseline (pixel and DCT variants)

### Algorithm Variants
- [gfcs_averaged_gradients.py](gfcs_averaged_gradients.py) - GFCS with averaged surrogate gradients
- [gfcs_minimal_victim_queries.py](gfcs_minimal_victim_queries.py) - GFCS with minimal victim queries (only 2)
- [gfcs_adaptive.py](gfcs_adaptive.py) - GFCS with adaptive surrogate weighting
- [gfcs_weighted_alignment.py](gfcs_weighted_alignment.py) - GFCS with alignment-based trust weighting

### Utilities and Experiments
- [run_experiment_from_config.py](run_experiment_from_config.py) - Run experiments from JSON configs
- [test_gfcs.py](test_gfcs.py) - Demo script with pretrained ImageNet models
- [utils.py](utils.py) - Visualization and evaluation utilities
- [cifar10_models.py](cifar10_models.py) - CIFAR-10 model definitions
- [tiny_imagenet_loader.py](tiny_imagenet_loader.py) - Tiny ImageNet dataset loader
- [compare_attacks_inception_v3.py](compare_attacks_inception_v3.py) - Comparison visualization script
- [generate_configs.py](generate_configs.py) - Config file generation utility

## Usage

### Basic Usage

```python
from gfcs import GFCS
import torchvision.models as models

# Load models
victim = models.resnet50(pretrained=True).eval()
surrogates = [
    models.vgg19(pretrained=True).eval(),
    models.resnet34(pretrained=True).eval(),
    models.densenet121(pretrained=True).eval(),
    models.mobilenet_v2(pretrained=True).eval(),
]

# Create attacker
attacker = GFCS(
    victim_model=victim,
    surrogate_models=surrogates,
    epsilon=2.0,
    max_queries=10000,
    targeted=False
)

# Run attack
x_adv, stats = attacker.attack(x, true_class)

print(f"Success: {stats['success']}")
print(f"Queries: {stats['total_queries']}")
print(f"Gradient queries: {stats['gradient_queries']}")
print(f"ODS queries: {stats['coimage_queries']}")
```

### Running the Demo

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib scipy

# Run demo
python test_gfcs.py --num_images 10 --max_queries 10000

# Use single surrogate (ResNet-152)
python test_gfcs.py --single_surrogate

# Compare with SimBA-ODS
python test_gfcs.py --compare
```

### Using Algorithm Variants

```python
# 1. GFCS with Averaged Gradients
from gfcs_averaged_gradients import GFCSAveragedGradients

attacker = GFCSAveragedGradients(
    victim_model=victim,
    surrogate_models=surrogates,
    epsilon=2.0,
    max_queries=10000
)
x_adv, stats = attacker.attack(x, true_class)

# 2. GFCS with Minimal Victim Queries
from gfcs_minimal_victim_queries import GFCSMinimalVictimQueries

attacker = GFCSMinimalVictimQueries(
    victim_model=victim,
    surrogate_models=surrogates,
    epsilon=2.0,
    max_iterations=1000  # Note: iterations, not queries
)
x_adv, stats = attacker.attack(x, true_class)
print(f"Victim queries: {stats['victim_queries']}")  # Should be 2

# 3. GFCS Adaptive
from gfcs_adaptive import GFCSAdaptive

attacker = GFCSAdaptive(
    victim_model=victim,
    surrogate_models=surrogates,
    epsilon=2.0,
    max_queries=10000,
    trust_learning_rate=0.1,
    trust_decay=0.95,
    use_adaptive_weighting=True,
    use_smart_ods=True
)
x_adv, stats = attacker.attack(x, true_class)
print(f"Trust scores: {stats['trust_scores']}")

# 4. GFCS Weighted Alignment
from gfcs_weighted_alignment import GFCSWeightedAlignment

attacker = GFCSWeightedAlignment(
    victim_model=victim,
    surrogate_models=surrogates,
    epsilon=2.0,
    max_queries=10000,
    trust_learning_rate=0.2,
    use_adaptive_weighting=True,
    use_smart_ods=True
)
x_adv, stats = attacker.attack(x, true_class)

# 5. SimBA (Pixel or DCT)
from SimBA import SimBA

# SimBA-DCT (more efficient)
attacker = SimBA(
    model=victim,
    epsilon=0.2,
    max_queries=20000,
    pixel_attack=False,
    freq_fraction=0.125
)
x_adv, stats = attacker.attack(x, true_class)

# SimBA-Pixel
attacker = SimBA(
    model=victim,
    epsilon=0.2,
    max_queries=20000,
    pixel_attack=True
)
x_adv, stats = attacker.attack(x, true_class)
```

### Running Experiments from Configs

```bash
# Run single experiment
python run_experiment_from_config.py exp_001

# Run multiple experiments
python run_experiment_from_config.py exp_001 exp_002 exp_003

# List all available experiments
python run_experiment_from_config.py --list

# See detailed experiment descriptions
cat configs/README.md
```

## Expected Results (from paper)

| Victim | Surrogates | Median Queries | Success Rate |
|--------|------------|----------------|--------------|
| VGG-16 | 1 (ResNet-152) | 6 | 99.90% |
| ResNet-50 | 1 (ResNet-152) | 4 | 99.85% |
| Inception-v3 | 1 (ResNet-152) | 18 | 98.60% |
| VGG-16 | 4 | 4 | 100% |
| ResNet-50 | 4 | 4 | 99.95% |
| Inception-v3 | 4 | 9 | 99.40% |

## Key Insights from the Paper

1. **Transfer typically works**: Most examples are solved with just surrogate gradients
2. **Coimage is the fallback**: ODS is only needed for a small fraction of images
3. **Multiple surrogates help**: Using 4 surrogates reduces failures significantly
4. **Simple is effective**: No complex priors or heuristics needed

## SimBA Implementation ([SimBA.py](SimBA.py))

This repository also includes an exact implementation of SimBA for comparison:

### SimBA: Simple Black-box Adversarial Attack

**Key idea**: Search along orthonormal basis directions without using surrogates

**Two modes:**

1. **SimBA-Pixel**: Search along pixel basis
   - Perturb one pixel at a time
   - Random or deterministic order
   - Simple but high-dimensional

2. **SimBA-DCT**: Search along DCT (frequency) basis
   - Perturb low-frequency DCT coefficients
   - More efficient (focuses on perceptually important frequencies)
   - Default: Use lowest 12.5% of frequencies (`freq_fraction=0.125`)

**Algorithm:**
```
For each basis direction e_i (random order):
    For sign in {+ε, -ε}:
        x_candidate = x + sign * ε * e_i
        if p(true_class | x_candidate) < p(true_class | x):
            x = x_candidate  # Accept step
```

**Parameters:**
- `epsilon`: Step size (default: 0.2)
- `max_queries`: Query budget (default: 20000)
- `pixel_attack`: True for pixel basis, False for DCT basis
- `freq_fraction`: Fraction of DCT frequencies to use (default: 0.125)
- `order`: 'random' (paper default) or 'diag' (deterministic low→high frequency)

**Usage:**
```python
from SimBA import SimBA

attacker = SimBA(
    model=victim,
    epsilon=0.2,
    max_queries=20000,
    pixel_attack=False,  # Use DCT
    freq_fraction=0.125
)

x_adv, stats = attacker.attack(x, true_class)
```

## Comparison: SimBA vs GFCS Variants

| Aspect | SimBA | GFCS (Standard) | GFCS Variants |
|--------|-------|-----------------|---------------|
| Uses surrogates | No | Yes | Yes |
| Search directions | Orthonormal basis (pixel/DCT) | Surrogate gradients + ODS | Weighted/averaged gradients + smart ODS |
| Median queries | ~500-1000 | ~4-18 | ~3-15 (varies by variant) |
| Complexity | Very simple | Simple | Moderate |
| Query efficiency | Low | High | Very high (some variants) |
| Victim query budget | Full budget | Full budget | Minimal (2 queries) to full |

## Implemented Extensions

This repository implements several extensions to the original GFCS algorithm:

### 1. Adaptive Surrogate Ranking (Implemented in gfcs_adaptive.py and gfcs_weighted_alignment.py)

Both adaptive variants implement dynamic surrogate ranking:

**GFCS Adaptive:**
- Trust-based ranking: Surrogates sorted by trust scores
- Tries surrogates in trust order (highest first)
- Updates trust based on success/failure

**GFCS Weighted Alignment:**
- Alignment-based ranking: Credit proportional to gradient alignment
- More nuanced than binary success/failure
- Rewards surrogates based on how much they contributed

### 2. Gradient Averaging (Implemented in gfcs_averaged_gradients.py and gfcs_minimal_victim_queries.py)

Both variants use averaged gradients from multiple surrogates:

**GFCS Averaged Gradients:**
- Averages ALL surrogates in S_rem at once
- Same query budget as original GFCS
- Tests if averaging improves transfer

**GFCS Minimal Victim Queries:**
- Averages ALL surrogates for every step
- Minimizes victim queries to only 2
- Tests transfer assumption: if all surrogates fooled → victim fooled

### 3. Smart ODS Fallback (Implemented in gfcs_adaptive.py and gfcs_weighted_alignment.py)

When gradient transfer fails, enhanced ODS:

- **Margin-aware sampling**: Bias towards high-probability classes
- **Class-specific weighting**: Penalize true class, boost target class
- **Momentum**: Remember successful ODS directions
- **Adaptive weights**: Track which weight patterns worked

### 4. Query Budget Variants

Different query budget models for different use cases:

- **Standard GFCS**: Full victim query budget (10000 queries)
- **Averaged Gradients**: Full victim query budget
- **Minimal Victim**: Only 2 victim queries, iteration-based budget
- **Adaptive/Weighted**: Full victim query budget with adaptive strategy

### 5. Experiment Configuration Framework (configs/ directory)

JSON-based experiment definitions for systematic evaluation:

```bash
# All experiments defined in configs/exp_*.json
# Run with: python run_experiment_from_config.py exp_001

# See configs/README.md for:
# - 34 pre-defined experiments
# - Baseline comparisons (GFCS vs SimBA)
# - Surrogate count ablations (1, 2, 4, 7, 10 surrogates)
# - Algorithm variant comparisons
# - Different datasets (Tiny ImageNet, CIFAR-10)
```

See [configs/README.md](configs/README.md) for detailed experiment descriptions.

## Visualization and Analysis

### Comparing Attack Methods

Use [compare_attacks_inception_v3.py](compare_attacks_inception_v3.py) to visualize and compare different attack methods:

```bash
python compare_attacks_inception_v3.py
```

This script:
- Compares multiple GFCS variants and SimBA on the same images
- Generates side-by-side visualizations of adversarial examples
- Shows query counts and success rates for each method
- Helps identify which variants work best for different scenarios

### Utilities ([utils.py](utils.py))

Helper functions for:
- Visualizing adversarial perturbations
- Computing perturbation statistics (L2 norm, SSIM, etc.)
- Evaluating attack success rates
- Saving and loading attack results

## Dataset Support

### Tiny ImageNet ([tiny_imagenet_loader.py](tiny_imagenet_loader.py))
- Automatic download and extraction
- 200 classes, 64×64 images (upscaled to 224×224)
- Drop-in replacement for full ImageNet
- Default path: `./data/tiny_imagenet`

### CIFAR-10 ([cifar10_models.py](cifar10_models.py))
- Pretrained CIFAR-10 models (ResNet, VGG, MobileNet families)
- 32×32 images, 10 classes
- Automatic download via torchvision
- Custom normalization: `mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]`

## References

- [GFCS Paper](https://openreview.net/forum?id=Zf4ZdI4OQPV) - ICLR 2022
- [SimBA Paper](https://arxiv.org/abs/1905.07121) - ICML 2019
- [ODS Paper](https://arxiv.org/abs/2010.06838) - NeurIPS 2020
