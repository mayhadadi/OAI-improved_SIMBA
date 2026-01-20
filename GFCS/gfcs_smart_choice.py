"""
GFCS-SmartChoice: Gradient First, Coimage Second with Smart Surrogate Selection
================================================================================
Modification of GFCS that selects surrogates based on output similarity to victim
instead of random selection.

Key change from original GFCS:
- Original: Randomly sample surrogate from S_rem (line 7)
- SmartChoice: Pick surrogate with highest output similarity to victim

This requires just 1 extra victim query at the start to compute similarities.
The surrogate selection is then deterministic based on similarity ranking.

Based on: "Attacking Deep Networks with Surrogate-Based Adversarial Black-Box Methods is Easy"
(Lord et al., ICLR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class GFCSSmartChoice:
    """
    GFCS with Smart Surrogate Selection based on output similarity.
    
    Instead of randomly sampling surrogates, we:
    1. At the start, query victim and all surrogates
    2. Compute cosine similarity between victim and each surrogate's output
    3. Try surrogates in order of similarity (most similar first)
    
    Args:
        victim_model: The black-box victim model
        surrogate_models: List of surrogate models with accessible gradients
        epsilon: Step size for perturbations (default: 2.0)
        norm_bound: L2 norm bound for total perturbation
        max_queries: Maximum number of queries to victim model
        targeted: Whether this is a targeted attack
        device: torch device
    """
    
    def __init__(
        self,
        victim_model: nn.Module,
        surrogate_models: List[nn.Module],
        epsilon: float = 2.0,
        norm_bound: float = None,
        max_queries: int = 10000,
        targeted: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.victim = victim_model.to(device).eval()
        self.surrogates = [s.to(device).eval() for s in surrogate_models]
        self.epsilon = epsilon
        self.norm_bound = norm_bound
        self.max_queries = max_queries
        self.targeted = targeted
        self.device = device
        
        # Statistics tracking
        self.query_count = 0
        self.gradient_queries = 0
        self.coimage_queries = 0
        
    def margin_loss(
        self, 
        logits: torch.Tensor, 
        true_class: int, 
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """Compute margin loss: L = f(c_t) - f(c_s)"""
        if self.targeted and target_class is not None:
            return logits[0, target_class] - logits[0, true_class]
        else:
            logits_copy = logits.clone()
            logits_copy[0, true_class] = -float('inf')
            second_highest_class = logits_copy.argmax(dim=1).item()
            return logits[0, second_highest_class] - logits[0, true_class]
    
    def query_victim(self, x: torch.Tensor) -> torch.Tensor:
        """Query victim model and increment counter."""
        self.query_count += 1
        with torch.no_grad():
            return self.victim(x)
    
    def compute_surrogate_similarities(
        self,
        victim_logits: torch.Tensor,
        x: torch.Tensor
    ) -> List[Tuple[int, float]]:
        """
        Compute similarity between victim's output and each surrogate's output.
        
        Returns:
            List of (surrogate_index, similarity) tuples, sorted by similarity (descending)
        """
        victim_logits_flat = victim_logits.view(-1)
        victim_norm = torch.norm(victim_logits_flat)
        if victim_norm > 0:
            victim_logits_flat = victim_logits_flat / victim_norm
        
        similarities = []
        with torch.no_grad():
            for idx, surrogate in enumerate(self.surrogates):
                surr_logits = surrogate(x).view(-1)
                surr_norm = torch.norm(surr_logits)
                if surr_norm > 0:
                    surr_logits = surr_logits / surr_norm
                
                sim = torch.dot(victim_logits_flat, surr_logits).item()
                similarities.append((idx, sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def get_surrogate_gradient(
        self, 
        x: torch.Tensor, 
        surrogate: nn.Module,
        true_class: int,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """Compute normalized loss gradient from a surrogate model."""
        x_input = x.clone().detach().requires_grad_(True)
        
        logits = surrogate(x_input)
        loss = self.margin_loss(logits, true_class, target_class)
        loss.backward()
        
        grad = x_input.grad.detach()
        grad_norm = torch.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm
            
        return grad
    
    def get_ods_direction(
        self,
        x: torch.Tensor,
        surrogate: nn.Module,
        num_classes: int = 1000
    ) -> torch.Tensor:
        """Compute ODS direction from surrogate's Jacobian row space."""
        x_input = x.clone().detach().requires_grad_(True)
        
        w = torch.empty(num_classes, device=self.device).uniform_(-1, 1)
        
        logits = surrogate(x_input)
        weighted_sum = (w * logits).sum()
        weighted_sum.backward()
        
        grad = x_input.grad.detach()
        grad_norm = torch.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm
            
        return grad
    
    def project_onto_ball(
        self,
        x_adv: torch.Tensor,
        x_orig: torch.Tensor,
        norm_bound: float
    ) -> torch.Tensor:
        """Project onto L2 ball and clamp to valid range."""
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
        return pred_class != true_class
    
    def attack(
        self,
        x: torch.Tensor,
        true_class: int,
        target_class: Optional[int] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Run GFCS-SmartChoice attack.
        
        Same as original GFCS Algorithm 1, but:
        - Line 7: Instead of randomly sampling, pick surrogate with highest similarity
        - Line 11: For ODS, also prefer higher-similarity surrogates
        
        Args:
            x: Input image tensor
            true_class: True class label
            target_class: Target class for targeted attacks
            
        Returns:
            x_adv: Adversarial example
            stats: Dictionary with attack statistics
        """
        # Reset statistics
        self.query_count = 0
        self.gradient_queries = 0
        self.coimage_queries = 0
        
        # Set norm bound if not specified
        if self.norm_bound is None:
            D = x.numel()
            self.norm_bound = np.sqrt(0.001 * D)
        
        # Initialize
        x_orig = x.clone().to(self.device)
        x_adv = x_orig.clone()
        
        # Get number of classes
        with torch.no_grad():
            num_classes = self.surrogates[0](x_adv).shape[1]
        
        # Initial victim query to compute similarities
        victim_logits = self.query_victim(x_adv)
        
        # Check if already adversarial
        if self.is_adversarial(victim_logits, true_class, target_class):
            return x_adv, self._get_stats(success=True)
        
        # Compute surrogate similarities (sorted by similarity, highest first)
        surrogate_ranking = self.compute_surrogate_similarities(victim_logits, x_adv)
        
        # S_rem: indices of remaining surrogates to try, in similarity order
        S_rem = [idx for idx, _ in surrogate_ranking]
        
        # Main loop
        while self.query_count < self.max_queries:
            # Query victim and check adversarial
            victim_logits = self.query_victim(x_adv)
            
            if self.is_adversarial(victim_logits, true_class, target_class):
                return x_adv, self._get_stats(success=True)
            
            current_loss = self.margin_loss(victim_logits, true_class, target_class).item()
            
            # Get candidate direction
            if len(S_rem) > 0:
                # GRADIENT FIRST: Pick most similar remaining surrogate
                surrogate_idx = S_rem.pop(0)  # Take first (most similar)
                surrogate = self.surrogates[surrogate_idx]
                
                q = self.get_surrogate_gradient(x_adv, surrogate, true_class, target_class)
                is_gradient_step = True
            else:
                # COIMAGE SECOND: Use most similar surrogate for ODS
                surrogate_idx = surrogate_ranking[0][0]  # Most similar overall
                surrogate = self.surrogates[surrogate_idx]
                
                q = self.get_ods_direction(x_adv, surrogate, num_classes)
                is_gradient_step = False
            
            # Try both step directions
            for alpha in [self.epsilon, -self.epsilon]:
                x_candidate = self.project_onto_ball(
                    x_adv + alpha * q,
                    x_orig,
                    self.norm_bound
                )
                
                candidate_logits = self.query_victim(x_candidate)
                candidate_loss = self.margin_loss(candidate_logits, true_class, target_class).item()
                
                if is_gradient_step:
                    self.gradient_queries += 1
                else:
                    self.coimage_queries += 1
                
                if candidate_loss > current_loss:
                    x_adv = x_candidate
                    
                    # Reset S_rem to full ranking order
                    S_rem = [idx for idx, _ in surrogate_ranking]
                    break
        
        # Final check
        final_logits = self.query_victim(x_adv)
        success = self.is_adversarial(final_logits, true_class, target_class)
        
        return x_adv, self._get_stats(success=success)
    
    def _get_stats(self, success: bool) -> dict:
        """Return attack statistics."""
        return {
            'success': success,
            'total_queries': self.query_count,
            'gradient_queries': self.gradient_queries,
            'coimage_queries': self.coimage_queries,
            'gradient_queries_ratio': self.gradient_queries / max(1, self.query_count)
        }