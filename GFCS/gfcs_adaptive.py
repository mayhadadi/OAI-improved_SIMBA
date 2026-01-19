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
    
    Key Improvements:
    1. Adaptive Surrogate Weighting:
       - Each surrogate has a "trust score" initialized to 1.0
       - When a surrogate's gradient leads to successful victim update, increase trust
       - When it fails, decrease trust
       - Use trust scores to weight gradient contributions
       
    2. Smarter ODS Fallback:
       - Track successful ODS weight vectors
       - Use momentum: bias new ODS samples towards successful directions
       - Adaptive class weighting based on margin structure
    
    Args:
        victim_model: The black-box victim model (only used for forward passes/queries)
        surrogate_models: List of surrogate models with accessible gradients
        epsilon: Step size for perturbations (default: 2.0 as per paper)
        norm_bound: L2 norm bound for total perturbation
        max_queries: Maximum number of queries to victim model
        targeted: Whether this is a targeted attack
        device: torch device
        
        # New adaptive parameters
        trust_learning_rate: How fast trust scores adapt (default: 0.3)
        trust_decay: Decay factor for failed attempts (default: 0.8)
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
        # Adaptive parameters
        trust_learning_rate: float = 0.3,
        trust_decay: float = 0.8,
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
        
        # Initialize trust scores for each surrogate (Improvement #3)
        self.surrogate_trust = torch.ones(self.num_surrogates, device=device)
        
        # ODS memory for smarter fallback (Improvement #4)
        self.ods_success_weights = None  # Will store successful ODS weight vectors
        self.ods_momentum_direction = None  # Momentum from successful ODS steps
        
        # Statistics tracking
        self.query_count = 0
        self.gradient_queries = 0
        self.coimage_queries = 0
        
        # Detailed statistics for analysis
        self.surrogate_success_counts = torch.zeros(self.num_surrogates, device=device)
        self.surrogate_attempt_counts = torch.zeros(self.num_surrogates, device=device)
        
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
        self.surrogate_success_counts = torch.zeros(self.num_surrogates, device=self.device)
        self.surrogate_attempt_counts = torch.zeros(self.num_surrogates, device=self.device)
        
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
        IMPROVEMENT #3: Adaptive Surrogate Weighting
        
        Compute weighted combination of surrogate gradients based on trust scores.
        Surrogates that have historically predicted victim behavior better get higher weights.
        """
        if surrogate_indices is None:
            surrogate_indices = list(range(self.num_surrogates))
            
        if not self.use_adaptive_weighting or len(surrogate_indices) == 1:
            # Fall back to simple averaging or single surrogate
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
        
        # Normalize weights to sum to 1
        weights = torch.tensor(weights, device=self.device)
        weights = F.softmax(weights, dim=0)  # Softmax to handle varying scales
        
        # Weighted combination
        combined = torch.zeros_like(grads[0])
        for i, grad in enumerate(grads):
            combined += weights[i] * grad
            
        # Normalize
        combined_norm = torch.norm(combined)
        if combined_norm > 0:
            combined = combined / combined_norm
            
        return combined
    
    def update_trust_scores(
        self,
        successful_indices: List[int],
        failed_indices: List[int]
    ):
        """
        Update trust scores based on which surrogates' gradients led to success/failure.
        """
        if not self.use_adaptive_weighting:
            return
            
        # Increase trust for successful surrogates
        for idx in successful_indices:
            self.surrogate_trust[idx] += self.trust_lr
            self.surrogate_success_counts[idx] += 1
            self.surrogate_attempt_counts[idx] += 1
            
        # Decrease trust for failed surrogates
        for idx in failed_indices:
            self.surrogate_trust[idx] *= self.trust_decay
            self.surrogate_attempt_counts[idx] += 1
            
        # Clamp trust scores to reasonable range
        self.surrogate_trust = torch.clamp(self.surrogate_trust, min=0.1, max=5.0)
    
    def get_smart_ods_direction(
        self,
        x: torch.Tensor,
        surrogate: nn.Module,
        num_classes: int,
        true_class: int,
        target_class: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        IMPROVEMENT #4: Smarter ODS Fallback
        
        Instead of purely random ODS sampling, use information from:
        1. Previously successful ODS weight vectors (momentum)
        2. Margin-aware class weighting (focus on classes near decision boundary)
        3. Anti-correlated sampling (avoid directions that failed)
        
        Returns: (ods_direction, weight_vector)
        """
        x_input = x.clone().detach().requires_grad_(True)
        
        # Get current logits to inform smart sampling
        with torch.no_grad():
            logits = surrogate(x_input)
            probs = F.softmax(logits, dim=1)
        
        if self.use_smart_ods:
            # Strategy 1: Margin-aware weighting
            # Give higher weight to classes near the decision boundary
            w = torch.empty(num_classes, device=self.device).uniform_(-1, 1)
            
            # Boost weights for top-k classes (more likely to affect decision)
            top_k = min(10, num_classes)
            top_indices = probs[0].topk(top_k).indices
            w[top_indices] *= 2.0
            
            # Reduce weight on true class (we want to move away from it)
            w[true_class] = -abs(w[true_class]) - 0.5
            
            # If targeted, boost target class
            if self.targeted and target_class is not None:
                w[target_class] = abs(w[target_class]) + 0.5
            
            # Strategy 2: Apply momentum from successful ODS directions
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
            
        # Apply momentum from successful ODS steps
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
        
        Enhanced Algorithm:
        1. Try weighted gradient from all surrogates (adaptive weighting)
        2. If that fails, try individual surrogate gradients (ordered by trust)
        3. If all gradients fail, use smart ODS (momentum + margin-aware)
        4. Update trust scores and ODS memory based on outcomes
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
        
        # Initial victim query to check if already adversarial
        logits = self.query_victim(x_adv)
        current_loss = self.margin_loss(logits, true_class, target_class).item()
        
        if self.is_adversarial(logits, true_class, target_class):
            return x_adv, {
                'success': True,
                'total_queries': self.query_count,
                'gradient_queries': 0,
                'coimage_queries': 0,
                'final_loss': current_loss
            }
        
        # Track which surrogates to try (ordered by trust)
        surrogate_order = torch.argsort(self.surrogate_trust, descending=True).tolist()
        remaining_surrogates = set(surrogate_order)
        
        while self.query_count < self.max_queries:
            step_successful = False
            
            # PHASE 1: Try weighted gradient from remaining surrogates
            if remaining_surrogates:
                # Get weighted gradient using adaptive weighting
                q = self.get_weighted_gradient(
                    x_adv, true_class, target_class, 
                    list(remaining_surrogates)
                )
                
                # Try both directions (SimBA-style)
                for alpha in [self.epsilon, -self.epsilon]:
                    x_candidate = self.project_onto_ball(x_adv + alpha * q, x_orig, norm_bound)
                    logits = self.query_victim(x_candidate)
                    self.gradient_queries += 1
                    new_loss = self.margin_loss(logits, true_class, target_class).item()
                    
                    if new_loss > current_loss:
                        x_adv = x_candidate
                        current_loss = new_loss
                        step_successful = True
                        
                        # Update trust scores - all remaining surrogates contributed
                        self.update_trust_scores(list(remaining_surrogates), [])
                        
                        # Reset remaining surrogates for next iteration
                        remaining_surrogates = set(surrogate_order)
                        break
                    
                    if self.is_adversarial(logits, true_class, target_class):
                        self.update_trust_scores(list(remaining_surrogates), [])
                        return x_adv, {
                            'success': True,
                            'total_queries': self.query_count,
                            'gradient_queries': self.gradient_queries,
                            'coimage_queries': self.coimage_queries,
                            'final_loss': current_loss,
                            'trust_scores': self.surrogate_trust.cpu().numpy().tolist()
                        }
                
                if not step_successful:
                    # Weighted gradient didn't work - try individual surrogates by trust order
                    for surr_idx in surrogate_order:
                        if surr_idx not in remaining_surrogates:
                            continue
                            
                        q = self.get_surrogate_gradient(
                            x_adv, self.surrogates[surr_idx], true_class, target_class
                        )
                        
                        for alpha in [self.epsilon, -self.epsilon]:
                            x_candidate = self.project_onto_ball(x_adv + alpha * q, x_orig, norm_bound)
                            logits = self.query_victim(x_candidate)
                            self.gradient_queries += 1
                            new_loss = self.margin_loss(logits, true_class, target_class).item()
                            
                            if new_loss > current_loss:
                                x_adv = x_candidate
                                current_loss = new_loss
                                step_successful = True
                                
                                # This surrogate worked!
                                self.update_trust_scores([surr_idx], [])
                                remaining_surrogates = set(surrogate_order)
                                break
                                
                            if self.is_adversarial(logits, true_class, target_class):
                                self.update_trust_scores([surr_idx], [])
                                return x_adv, {
                                    'success': True,
                                    'total_queries': self.query_count,
                                    'gradient_queries': self.gradient_queries,
                                    'coimage_queries': self.coimage_queries,
                                    'final_loss': current_loss,
                                    'trust_scores': self.surrogate_trust.cpu().numpy().tolist()
                                }
                        
                        if step_successful:
                            break
                        else:
                            # This surrogate failed
                            remaining_surrogates.discard(surr_idx)
                            self.update_trust_scores([], [surr_idx])
            
            # PHASE 2: Smart ODS Fallback (all gradients exhausted)
            if not step_successful:
                # Pick surrogate for ODS (weighted by trust)
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
                        
                        # Update ODS memory with successful direction
                        self.update_ods_memory(w, q)
                        
                        # Reset surrogates for gradient phase
                        remaining_surrogates = set(surrogate_order)
                        break
                        
                    if self.is_adversarial(logits, true_class, target_class):
                        self.update_ods_memory(w, q)
                        return x_adv, {
                            'success': True,
                            'total_queries': self.query_count,
                            'gradient_queries': self.gradient_queries,
                            'coimage_queries': self.coimage_queries,
                            'final_loss': current_loss,
                            'trust_scores': self.surrogate_trust.cpu().numpy().tolist()
                        }
        
        # Attack failed
        return x_adv, {
            'success': False,
            'total_queries': self.query_count,
            'gradient_queries': self.gradient_queries,
            'coimage_queries': self.coimage_queries,
            'final_loss': current_loss,
            'trust_scores': self.surrogate_trust.cpu().numpy().tolist()
        }


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