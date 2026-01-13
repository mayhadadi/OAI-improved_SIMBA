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

## Key Components

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

- `gfcs.py` - Main GFCS implementation
- `test_gfcs.py` - Demo script with pretrained ImageNet models
- `utils.py` - Visualization and evaluation utilities

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
pip install torch torchvision numpy matplotlib

# Run demo
python test_gfcs.py --num_images 10 --max_queries 10000

# Use single surrogate (ResNet-152)
python test_gfcs.py --single_surrogate

# Compare with SimBA-ODS
python test_gfcs.py --compare
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

## Comparison with SimBA

| Aspect | SimBA | GFCS |
|--------|-------|------|
| Uses surrogates | No | Yes |
| Search directions | Orthonormal basis (pixel/DCT) | Surrogate gradients + ODS |
| Median queries | ~500-1000 | ~4-18 |
| Complexity | Very simple | Simple |

## Extensions for Your Project

### Ranking-Based Direction Selection

Instead of random surrogate selection, implement ranking:

```python
def rank_surrogates(self, x, true_class):
    """Rank surrogates by estimated gradient quality."""
    gradients = []
    for surrogate in self.surrogates:
        g = self.get_surrogate_gradient(x, surrogate, true_class)
        gradients.append(g)
    
    # Option 1: Agreement-based ranking
    # Compute pairwise cosine similarities
    # Rank by average agreement with others
    
    # Option 2: Magnitude-based ranking
    # Rank by gradient magnitude (larger = stronger signal)
    
    # Option 3: Loss-based ranking (requires queries)
    # Test each direction, rank by loss improvement
    
    return ranked_order
```

## References

- [GFCS Paper](https://openreview.net/forum?id=xx) - ICLR 2022
- [SimBA Paper](https://arxiv.org/abs/1905.07121) - ICML 2019
- [ODS Paper](https://arxiv.org/abs/2010.06838) - NeurIPS 2020
