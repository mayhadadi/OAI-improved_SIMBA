"""
GFCS-Adaptive: Gradient First, Coimage Second with Adaptive Improvements
=========================================================================
Enhanced implementation of GFCS with two key improvements:

1. ADAPTIVE SURROGATE WEIGHTING (Improvement #3):
   - Track success rate of each surrogate's gradient transfer
   - Weight gradients by historical success when combining
   - Prioritize surrogates that better predict victim behavior

2. SMARTER ODS FALLBACK (Improvement #4):
   - Track which ODS directions worked/failed
   - Bias ODS sampling towards successful class-weight patterns
   - Use momentum from successful ODS steps

Based on: "Attacking Deep Networks with Surrogate-Based Adversarial Black-Box Methods is Easy"
(Lord et al., ICLR 2022)

v2 Changes:
- Better surrogate differentiation: only credit individual surrogates, not groups
- Adjusted trust dynamics: lower learning rate, higher decay sensitivity
- Track actual success/attempt ratios per surrogate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import defaultdict


class GFCSAdaptive:
    """
    GFCS with Adaptive Surrogate Weighting and Smarter ODS Fallback.
    
    v2: Improved surrogate differentiation
    
    Args:
        victim_model: The black-box victim model (only used for forward passes/queries)
        surrogate_models: List of surrogate models with accessible gradients
        epsilon: Step size for perturbations (default: 2.0 as per paper)
        norm_bound: L2 norm bound for total perturbation
        max_queries: Maximum number of queries to victim model
        targeted: Whether this is a targeted attack
        device: torch device
        
        # Adaptive parameters
        trust_learning_rate: How fast trust scores adapt (default: 0.1, reduced from 0.3)
        trust_decay: Decay factor for failed attempts (default: 0.95, increased from 0.8)
        ods_momentum: Momentum for ODS direction memory (default: 0.5)
        use_adaptive_weighting: Enable adaptive surrogate weighting
        use_smart_ods: Enable smarter ODS fallback
    """
    
    def __init__(
        self,
        victim_model: nn.Module,
        surrogate_models: List[nn.Module],
        epsilon: float = 2.0,
        norm_bound: float = None,
        max_queries: int = 10000,
        targeted: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        # Adaptive parameters - adjusted defaults for better differentiation
        trust_learning_rate: float = 0.1,  # Reduced from 0.3
        trust_decay: float = 0.95,  # Increased from 0.8 (less aggressive decay)
        ods_momentum: float = 0.5,
        use_adaptive_weighting: bool = True,
        use_smart_ods: bool = True
    ):
        self.victim = victim_model.to(device).eval()
        self.surrogates = [s.to(device).eval() for s in surrogate_models]
        self.num_surrogates = len(surrogate_models)
        self.epsilon = epsilon
        self.norm_bound = norm_bound
        self.max_queries = max_queries
        self.targeted = targeted
        self.device = device
        
        # Adaptive parameters
        self.trust_lr = trust_learning_rate
        self.trust_decay = trust_decay
        self.ods_momentum = ods_momentum
        self.use_adaptive_weighting = use_adaptive_weighting
        self.use_smart_ods = use_smart_ods
        
        # Initialize trust scores for each surrogate (start at 1.0)
        self.surrogate_trust = torch.ones(self.num_surrogates, device=device)
        
        # Track actual success/attempt counts per surrogate (for analysis)
        self.surrogate_successes = torch.zeros(self.num_surrogates, device=device)
        self.surrogate_attempts = torch.zeros(self.num_surrogates, device=device)
        
        # ODS memory for smarter fallback
        self.ods_success_weights = None
        self.ods_momentum_direction = None
        
        # Statistics tracking
        self.query_count = 0
        self.gradient_queries = 0
        self.coimage_queries = 0
        
    def reset_statistics(self):
        """Reset per-attack statistics but keep learned trust scores."""
        self.query_count = 0
        self.gradient_queries = 0
        self.coimage_queries = 0
        self.ods_success_weights = None
        self.ods_momentum_direction = None
        
    def reset_trust_scores(self):
        """Reset trust scores to initial values (for new attack campaigns)."""
        self.surrogate_trust = torch.ones(self.num_surrogates, device=self.device)
        self.surrogate_successes = torch.zeros(self.num_surrogates, device=self.device)
        self.surrogate_attempts = torch.zeros(self.num_surrogates, device=self.device)
        
    def margin_loss(
        self, 
        logits: torch.Tensor, 
        true_class: int, 
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute the margin loss: L = f(c_t) - f(c_s)
        For untargeted: c_t is second highest, c_s is true class
        For targeted: c_t is target, c_s is true class
        """
        if self.targeted and target_class is not None:
            return logits[0, target_class] - logits[0, true_class]
        else:
            # Get second highest class (excluding true class)
            logits_copy = logits.clone()
            logits_copy[0, true_class] = float('-inf')
            second_highest_class = logits_copy.argmax(dim=1).item()
            return logits[0, second_highest_class] - logits[0, true_class]
    
    def get_surrogate_gradient(
        self,
        x: torch.Tensor,
        surrogate: nn.Module,
        true_class: int,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """Get normalized gradient from a single surrogate."""
        x_input = x.clone().detach().requires_grad_(True)
        logits = surrogate(x_input)
        loss = self.margin_loss(logits, true_class, target_class)
        loss.backward()
        
        grad = x_input.grad.detach()
        grad_norm = torch.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm
        return grad
    
    def get_weighted_gradient(
        self,
        x: torch.Tensor,
        true_class: int,
        target_class: Optional[int] = None,
        surrogate_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute weighted combination of surrogate gradients based on trust scores.
        """
        if surrogate_indices is None:
            surrogate_indices = list(range(self.num_surrogates))
            
        if not self.use_adaptive_weighting or len(surrogate_indices) == 1:
            # Simple averaging or single surrogate
            grads = []
            for idx in surrogate_indices:
                grad = self.get_surrogate_gradient(x, self.surrogates[idx], true_class, target_class)
                grads.append(grad)
            combined = torch.stack(grads).mean(dim=0)
            combined_norm = torch.norm(combined)
            if combined_norm > 0:
                combined = combined / combined_norm
            return combined
        
        # Weighted combination based on trust scores
        grads = []
        weights = []
        for idx in surrogate_indices:
            grad = self.get_surrogate_gradient(x, self.surrogates[idx], true_class, target_class)
            grads.append(grad)
            weights.append(self.surrogate_trust[idx].item())
        
        # Softmax weights with temperature for sharper differentiation
        weights = torch.tensor(weights, device=self.device)
        temperature = 0.5  # Lower = sharper differentiation
        weights = F.softmax(weights / temperature, dim=0)
        
        # Weighted combination
        combined = torch.zeros_like(grads[0])
        for i, grad in enumerate(grads):
            combined += weights[i] * grad
            
        # Normalize
        combined_norm = torch.norm(combined)
        if combined_norm > 0:
            combined = combined / combined_norm
            
        return combined
    
    def update_trust_single(self, surrogate_idx: int, success: bool):
        """
        Update trust score for a SINGLE surrogate based on its individual performance.
        This is the key change for better differentiation.
        """
        if not self.use_adaptive_weighting:
            return
        
        self.surrogate_attempts[surrogate_idx] += 1
        
        if success:
            # Increase trust for this specific surrogate
            self.surrogate_successes[surrogate_idx] += 1
            self.surrogate_trust[surrogate_idx] += self.trust_lr
        else:
            # Decrease trust for this specific surrogate
            self.surrogate_trust[surrogate_idx] *= self.trust_decay
            
        # Clamp to reasonable range [0.1, 3.0] - reduced upper bound
        self.surrogate_trust = torch.clamp(self.surrogate_trust, min=0.1, max=3.0)
    
    def get_smart_ods_direction(
        self,
        x: torch.Tensor,
        surrogate: nn.Module,
        num_classes: int,
        true_class: int,
        target_class: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Smarter ODS Fallback with margin-aware sampling and momentum.
        """
        x_input = x.clone().detach().requires_grad_(True)
        
        # Get current logits for smart sampling
        with torch.no_grad():
            logits = surrogate(x_input)
            probs = F.softmax(logits, dim=1)
        
        if self.use_smart_ods:
            # Margin-aware weighting
            w = torch.empty(num_classes, device=self.device).uniform_(-1, 1)
            
            # Boost weights for top-k classes
            top_k = min(10, num_classes)
            top_indices = probs[0].topk(top_k).indices
            w[top_indices] *= 2.0
            
            # Reduce weight on true class
            w[true_class] = -abs(w[true_class]) - 0.5
            
            # Boost target class for targeted attacks
            if self.targeted and target_class is not None:
                w[target_class] = abs(w[target_class]) + 0.5
            
            # Apply momentum from successful ODS directions
            if self.ods_success_weights is not None and self.ods_momentum > 0:
                w = (1 - self.ods_momentum) * w + self.ods_momentum * self.ods_success_weights
        else:
            # Original random ODS
            w = torch.empty(num_classes, device=self.device).uniform_(-1, 1)
        
        # Compute ODS direction
        x_input = x.clone().detach().requires_grad_(True)
        logits = surrogate(x_input)
        weighted_sum = (w * logits).sum()
        weighted_sum.backward()
        
        grad = x_input.grad.detach()
        grad_norm = torch.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm
            
        # Apply direction momentum
        if self.use_smart_ods and self.ods_momentum_direction is not None:
            grad = (1 - self.ods_momentum) * grad + self.ods_momentum * self.ods_momentum_direction
            grad = grad / torch.norm(grad)
            
        return grad, w
    
    def update_ods_memory(self, successful_weights: torch.Tensor, successful_direction: torch.Tensor):
        """Update ODS memory with successful weight vectors and directions."""
        if not self.use_smart_ods:
            return
            
        if self.ods_success_weights is None:
            self.ods_success_weights = successful_weights.clone()
            self.ods_momentum_direction = successful_direction.clone()
        else:
            # Exponential moving average
            self.ods_success_weights = 0.7 * self.ods_success_weights + 0.3 * successful_weights
            self.ods_momentum_direction = 0.7 * self.ods_momentum_direction + 0.3 * successful_direction
    
    def project_onto_ball(
        self,
        x_adv: torch.Tensor,
        x_orig: torch.Tensor,
        norm_bound: float
    ) -> torch.Tensor:
        """Project onto L2 ball and clamp to valid image range."""
        delta = x_adv - x_orig
        delta_norm = torch.norm(delta)
        
        if delta_norm > norm_bound:
            delta = delta * (norm_bound / delta_norm)
            
        return torch.clamp(x_orig + delta, 0, 1)
    
    def is_adversarial(
        self,
        logits: torch.Tensor,
        true_class: int,
        target_class: Optional[int] = None
    ) -> bool:
        """Check if prediction is adversarial."""
        pred_class = logits.argmax(dim=1).item()
        if self.targeted:
            return pred_class == target_class
        else:
            return pred_class != true_class
    
    def query_victim(self, x: torch.Tensor) -> torch.Tensor:
        """Query the victim model and increment counter."""
        self.query_count += 1
        with torch.no_grad():
            return self.victim(x)
    
    def attack(
        self,
        x: torch.Tensor,
        true_class: int,
        target_class: Optional[int] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Run the GFCS-Adaptive attack.
        
        v2 Algorithm Changes:
        - Try individual surrogates in trust order (not weighted combination first)
        - Only credit the specific surrogate that worked
        - Better differentiation between surrogate quality
        """
        self.reset_statistics()
        
        x = x.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        x_orig = x.clone()
        x_adv = x.clone()
        
        # Set norm bound if not specified
        norm_bound = self.norm_bound
        if norm_bound is None:
            D = x.numel()
            norm_bound = np.sqrt(0.001 * D)
        
        # Get number of classes from first surrogate
        with torch.no_grad():
            num_classes = self.surrogates[0](x_adv).shape[1]
        
        # Initial victim query
        logits = self.query_victim(x_adv)
        current_loss = self.margin_loss(logits, true_class, target_class).item()
        
        if self.is_adversarial(logits, true_class, target_class):
            return x_adv, {
                'success': True,
                'total_queries': self.query_count,
                'gradient_queries': 0,
                'coimage_queries': 0,
                'final_loss': current_loss,
                'trust_scores': self.surrogate_trust.cpu().numpy().tolist(),
                'surrogate_success_rates': self._get_success_rates()
            }
        
        while self.query_count < self.max_queries:
            step_successful = False
            
            # Get surrogate order by trust (highest first)
            surrogate_order = torch.argsort(self.surrogate_trust, descending=True).tolist()
            
            # PHASE 1: Try individual surrogate gradients in trust order
            for surr_idx in surrogate_order:
                q = self.get_surrogate_gradient(x_adv, self.surrogates[surr_idx], true_class, target_class)
                
                # Track that we're attempting with this surrogate
                for alpha in [self.epsilon, -self.epsilon]:
                    x_candidate = self.project_onto_ball(x_adv + alpha * q, x_orig, norm_bound)
                    logits = self.query_victim(x_candidate)
                    self.gradient_queries += 1
                    new_loss = self.margin_loss(logits, true_class, target_class).item()
                    
                    if new_loss > current_loss:
                        x_adv = x_candidate
                        current_loss = new_loss
                        step_successful = True
                        
                        # Credit only THIS surrogate
                        self.update_trust_single(surr_idx, success=True)
                        break
                        
                    if self.is_adversarial(logits, true_class, target_class):
                        self.update_trust_single(surr_idx, success=True)
                        return x_adv, {
                            'success': True,
                            'total_queries': self.query_count,
                            'gradient_queries': self.gradient_queries,
                            'coimage_queries': self.coimage_queries,
                            'final_loss': current_loss,
                            'trust_scores': self.surrogate_trust.cpu().numpy().tolist(),
                            'surrogate_success_rates': self._get_success_rates()
                        }
                
                if step_successful:
                    break
                else:
                    # This surrogate's gradient didn't help
                    self.update_trust_single(surr_idx, success=False)
            
            # PHASE 2: If no individual gradient worked, try weighted combination
            if not step_successful:
                q = self.get_weighted_gradient(x_adv, true_class, target_class)
                
                for alpha in [self.epsilon, -self.epsilon]:
                    x_candidate = self.project_onto_ball(x_adv + alpha * q, x_orig, norm_bound)
                    logits = self.query_victim(x_candidate)
                    self.gradient_queries += 1
                    new_loss = self.margin_loss(logits, true_class, target_class).item()
                    
                    if new_loss > current_loss:
                        x_adv = x_candidate
                        current_loss = new_loss
                        step_successful = True
                        # Don't update individual trust here - it's a group effort
                        break
                        
                    if self.is_adversarial(logits, true_class, target_class):
                        return x_adv, {
                            'success': True,
                            'total_queries': self.query_count,
                            'gradient_queries': self.gradient_queries,
                            'coimage_queries': self.coimage_queries,
                            'final_loss': current_loss,
                            'trust_scores': self.surrogate_trust.cpu().numpy().tolist(),
                            'surrogate_success_rates': self._get_success_rates()
                        }
            
            # PHASE 3: Smart ODS Fallback
            if not step_successful:
                # Pick surrogate weighted by trust
                surr_probs = F.softmax(self.surrogate_trust, dim=0)
                surr_idx = torch.multinomial(surr_probs, 1).item()
                
                q, w = self.get_smart_ods_direction(
                    x_adv, self.surrogates[surr_idx], num_classes, 
                    true_class, target_class
                )
                
                for alpha in [self.epsilon, -self.epsilon]:
                    x_candidate = self.project_onto_ball(x_adv + alpha * q, x_orig, norm_bound)
                    logits = self.query_victim(x_candidate)
                    self.coimage_queries += 1
                    new_loss = self.margin_loss(logits, true_class, target_class).item()
                    
                    if new_loss > current_loss:
                        x_adv = x_candidate
                        current_loss = new_loss
                        step_successful = True
                        
                        # Update ODS memory
                        self.update_ods_memory(w, q)
                        break
                        
                    if self.is_adversarial(logits, true_class, target_class):
                        self.update_ods_memory(w, q)
                        return x_adv, {
                            'success': True,
                            'total_queries': self.query_count,
                            'gradient_queries': self.gradient_queries,
                            'coimage_queries': self.coimage_queries,
                            'final_loss': current_loss,
                            'trust_scores': self.surrogate_trust.cpu().numpy().tolist(),
                            'surrogate_success_rates': self._get_success_rates()
                        }
        
        # Attack failed
        return x_adv, {
            'success': False,
            'total_queries': self.query_count,
            'gradient_queries': self.gradient_queries,
            'coimage_queries': self.coimage_queries,
            'final_loss': current_loss,
            'trust_scores': self.surrogate_trust.cpu().numpy().tolist(),
            'surrogate_success_rates': self._get_success_rates()
        }
    
    def _get_success_rates(self) -> List[float]:
        """Get per-surrogate success rates."""
        rates = []
        for i in range(self.num_surrogates):
            if self.surrogate_attempts[i] > 0:
                rate = (self.surrogate_successes[i] / self.surrogate_attempts[i]).item()
            else:
                rate = 0.0
            rates.append(rate)
        return rates


class GFCSAdaptiveAblation(GFCSAdaptive):
    """
    Ablation variants for testing individual improvements.
    """
    
    @classmethod
    def only_adaptive_weighting(cls, *args, **kwargs):
        """Only use adaptive surrogate weighting, not smart ODS."""
        kwargs['use_adaptive_weighting'] = True
        kwargs['use_smart_ods'] = False
        return cls(*args, **kwargs)
    
    @classmethod
    def only_smart_ods(cls, *args, **kwargs):
        """Only use smart ODS, not adaptive weighting."""
        kwargs['use_adaptive_weighting'] = False
        kwargs['use_smart_ods'] = True
        return cls(*args, **kwargs)
    
    @classmethod
    def baseline(cls, *args, **kwargs):
        """Baseline without any improvements (should match original GFCS)."""
        kwargs['use_adaptive_weighting'] = False
        kwargs['use_smart_ods'] = False
        return cls(*args, **kwargs)