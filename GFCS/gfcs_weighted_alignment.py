"""
GFCS-WeightedAlignment: Gradient First, Coimage Second with Adaptive Improvements
==================================================================================
Enhanced implementation of GFCS with two key improvements:

1. ADAPTIVE SURROGATE WEIGHTING (Improvement #3):
   - For each image: query victim and surrogates, compute output similarity
   - Initialize trust scores based on cosine similarity of logit vectors
   - Surrogates with similar outputs to victim get higher initial weight
   - Update trust during attack based on gradient alignment with success

2. SMARTER ODS FALLBACK (Improvement #4):
   - Track which ODS directions worked/failed
   - Bias ODS sampling towards successful class-weight patterns
   - Use momentum from successful ODS steps

Key insight: If a surrogate's output is similar to the victim's on a specific image,
its gradients are more likely to transfer well for that image.

Based on: "Attacking Deep Networks with Surrogate-Based Adversarial Black-Box Methods is Easy"
(Lord et al., ICLR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


class GFCSWeightedAlignment:
    """
    GFCS with Adaptive Surrogate Weighting and Smarter ODS Fallback.
    
    v3: Weighted combination with alignment-based trust updates
    
    Args:
        victim_model: The black-box victim model
        surrogate_models: List of surrogate models with accessible gradients
        epsilon: Step size for perturbations (default: 2.0)
        norm_bound: L2 norm bound for total perturbation
        max_queries: Maximum number of queries to victim model
        targeted: Whether this is a targeted attack
        device: torch device
        trust_learning_rate: How fast trust scores adapt (default: 0.2)
        trust_decay: Decay factor for low-alignment surrogates (default: 0.98)
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
        trust_learning_rate: float = 0.2,
        trust_decay: float = 0.98,
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
        
        # Initialize trust scores (start at 1.0)
        self.surrogate_trust = torch.ones(self.num_surrogates, device=device)
        
        # Track alignment statistics
        self.surrogate_alignment_sum = torch.zeros(self.num_surrogates, device=device)
        self.surrogate_update_count = torch.zeros(self.num_surrogates, device=device)
        
        # ODS memory
        self.ods_success_weights = None
        self.ods_momentum_direction = None
        
        # Statistics
        self.query_count = 0
        self.gradient_queries = 0
        self.coimage_queries = 0
        
    def reset_statistics(self):
        """Reset per-attack statistics."""
        self.query_count = 0
        self.gradient_queries = 0
        self.coimage_queries = 0
        self.ods_success_weights = None
        self.ods_momentum_direction = None
        
    def reset_trust_scores(self):
        """Reset trust scores to initial values."""
        self.surrogate_trust = torch.ones(self.num_surrogates, device=self.device)
        self.surrogate_alignment_sum = torch.zeros(self.num_surrogates, device=self.device)
        self.surrogate_update_count = torch.zeros(self.num_surrogates, device=self.device)
        
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
    
    def get_all_gradients(
        self,
        x: torch.Tensor,
        true_class: int,
        target_class: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Get normalized gradients from all surrogates."""
        grads = []
        for surrogate in self.surrogates:
            grad = self.get_surrogate_gradient(x, surrogate, true_class, target_class)
            grads.append(grad)
        return grads
    
    def get_weighted_gradient(
        self,
        gradients: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute weighted combination of gradients based on trust scores."""
        if not self.use_adaptive_weighting:
            # Simple average
            combined = torch.stack(gradients).mean(dim=0)
        else:
            # Weighted by trust with temperature for differentiation
            weights = F.softmax(self.surrogate_trust / 0.5, dim=0)
            combined = torch.zeros_like(gradients[0])
            for i, grad in enumerate(gradients):
                combined += weights[i] * grad
        
        # Normalize
        combined_norm = torch.norm(combined)
        if combined_norm > 0:
            combined = combined / combined_norm
        return combined
    
    def update_trust_by_alignment(
        self,
        gradients: List[torch.Tensor],
        successful_direction: torch.Tensor
    ):
        """
        Update trust scores based on how well each surrogate's gradient
        aligned with the successful update direction.
        
        Surrogates whose gradients pointed in the same direction as the
        successful update get increased trust.
        """
        if not self.use_adaptive_weighting:
            return
        
        successful_direction_flat = successful_direction.view(-1)
        successful_norm = torch.norm(successful_direction_flat)
        if successful_norm == 0:
            return
        successful_direction_flat = successful_direction_flat / successful_norm
        
        for i, grad in enumerate(gradients):
            grad_flat = grad.view(-1)
            # Cosine similarity: how aligned is this gradient with success?
            alignment = torch.dot(grad_flat, successful_direction_flat).item()
            
            # Track statistics
            self.surrogate_alignment_sum[i] += alignment
            self.surrogate_update_count[i] += 1
            
            # Update trust based on alignment
            if alignment > 0.1:  # Positively aligned
                self.surrogate_trust[i] += self.trust_lr * alignment
            elif alignment < -0.1:  # Negatively aligned (bad surrogate)
                self.surrogate_trust[i] *= self.trust_decay
        
        # Clamp trust scores
        self.surrogate_trust = torch.clamp(self.surrogate_trust, min=0.1, max=3.0)
    
    def decay_all_trust(self):
        """Slightly decay all trust scores when no progress is made."""
        if self.use_adaptive_weighting:
            self.surrogate_trust *= 0.995
            self.surrogate_trust = torch.clamp(self.surrogate_trust, min=0.1, max=3.0)
    
    def get_smart_ods_direction(
        self,
        x: torch.Tensor,
        surrogate: nn.Module,
        num_classes: int,
        true_class: int,
        target_class: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Smart ODS with margin-aware sampling and momentum."""
        x_input = x.clone().detach().requires_grad_(True)
        
        with torch.no_grad():
            logits = surrogate(x_input)
            probs = F.softmax(logits, dim=1)
        
        if self.use_smart_ods:
            w = torch.empty(num_classes, device=self.device).uniform_(-1, 1)
            
            # Boost top-k classes
            top_k = min(10, num_classes)
            top_indices = probs[0].topk(top_k).indices
            w[top_indices] *= 2.0
            
            # Bias against true class
            w[true_class] = -abs(w[true_class]) - 0.5
            
            # Boost target for targeted attacks
            if self.targeted and target_class is not None:
                w[target_class] = abs(w[target_class]) + 0.5
            
            # Apply momentum from successful ODS
            if self.ods_success_weights is not None:
                w = (1 - self.ods_momentum) * w + self.ods_momentum * self.ods_success_weights
        else:
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
    
    def update_ods_memory(self, weights: torch.Tensor, direction: torch.Tensor):
        """Update ODS memory with successful direction."""
        if not self.use_smart_ods:
            return
        
        if self.ods_success_weights is None:
            self.ods_success_weights = weights.clone()
            self.ods_momentum_direction = direction.clone()
        else:
            self.ods_success_weights = 0.7 * self.ods_success_weights + 0.3 * weights
            self.ods_momentum_direction = 0.7 * self.ods_momentum_direction + 0.3 * direction
    
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
    
    def query_victim(self, x: torch.Tensor) -> torch.Tensor:
        """Query victim model."""
        self.query_count += 1
        with torch.no_grad():
            return self.victim(x)
    
    def compute_output_similarity(
        self,
        victim_logits: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between victim's output and each surrogate's output.
        Uses cosine similarity on the logit vectors.
        
        Returns:
            Tensor of similarity scores, one per surrogate
        """
        victim_logits_flat = victim_logits.view(-1)
        victim_norm = torch.norm(victim_logits_flat)
        if victim_norm > 0:
            victim_logits_flat = victim_logits_flat / victim_norm
        
        similarities = []
        with torch.no_grad():
            for surrogate in self.surrogates:
                surr_logits = surrogate(x).view(-1)
                surr_norm = torch.norm(surr_logits)
                if surr_norm > 0:
                    surr_logits = surr_logits / surr_norm
                
                # Cosine similarity
                sim = torch.dot(victim_logits_flat, surr_logits).item()
                similarities.append(sim)
        
        return torch.tensor(similarities, device=self.device)
    
    def attack(
        self,
        x: torch.Tensor,
        true_class: int,
        target_class: Optional[int] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Run GFCS-WeightedAlignment attack.
        
        Algorithm:
        1. Query victim and all surrogates, compute output similarity
        2. Initialize trust scores based on similarity (per-image)
        3. Get gradients from all surrogates
        4. Compute weighted combination based on trust
        5. Try SimBA update with combined gradient
        6. If success: update trust based on alignment
        7. If fail: fall back to smart ODS
        8. Repeat until adversarial or max queries
        """
        self.reset_statistics()
        
        x = x.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x_orig = x.clone()
        x_adv = x.clone()
        
        # Set norm bound
        norm_bound = self.norm_bound
        if norm_bound is None:
            D = x.numel()
            norm_bound = np.sqrt(0.001 * D)
        
        # Get num classes
        with torch.no_grad():
            num_classes = self.surrogates[0](x_adv).shape[1]
        
        # Initial query
        logits = self.query_victim(x_adv)
        current_loss = self.margin_loss(logits, true_class, target_class).item()
        
        # NEW: Initialize per-image trust based on output similarity
        if self.use_adaptive_weighting:
            similarities = self.compute_output_similarity(logits, x_adv)
            # Convert similarities to trust scores: higher similarity = higher trust
            # Shift to positive range and scale
            self.surrogate_trust = torch.clamp(similarities + 1.0, min=0.1, max=3.0)
            # Reset alignment tracking for this image
            self.surrogate_alignment_sum = torch.zeros(self.num_surrogates, device=self.device)
            self.surrogate_update_count = torch.zeros(self.num_surrogates, device=self.device)
        
        if self.is_adversarial(logits, true_class, target_class):
            return x_adv, self._make_stats(True, current_loss)
        
        # Track if we're in ODS mode (all gradients exhausted for this iteration)
        use_ods = False
        ods_surrogate_idx = 0
        
        while self.query_count < self.max_queries:
            step_successful = False
            
            if not use_ods:
                # PHASE 1: Gradient transfer with weighted combination
                gradients = self.get_all_gradients(x_adv, true_class, target_class)
                q = self.get_weighted_gradient(gradients)
                
                for alpha in [self.epsilon, -self.epsilon]:
                    x_candidate = self.project_onto_ball(x_adv + alpha * q, x_orig, norm_bound)
                    logits = self.query_victim(x_candidate)
                    self.gradient_queries += 1
                    new_loss = self.margin_loss(logits, true_class, target_class).item()
                    
                    if new_loss > current_loss:
                        # Success! Update trust based on alignment
                        successful_direction = alpha * q
                        self.update_trust_by_alignment(gradients, successful_direction)
                        
                        x_adv = x_candidate
                        current_loss = new_loss
                        step_successful = True
                        use_ods = False  # Reset to gradient mode
                        break
                    
                    if self.is_adversarial(logits, true_class, target_class):
                        successful_direction = alpha * q
                        self.update_trust_by_alignment(gradients, successful_direction)
                        return x_adv, self._make_stats(True, current_loss)
                
                if not step_successful:
                    # Gradient didn't work, switch to ODS
                    use_ods = True
                    self.decay_all_trust()
            
            else:
                # PHASE 2: Smart ODS fallback
                # Select surrogate weighted by trust
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
                        
                        # Update ODS memory and switch back to gradient mode
                        self.update_ods_memory(w, q)
                        use_ods = False
                        break
                    
                    if self.is_adversarial(logits, true_class, target_class):
                        self.update_ods_memory(w, q)
                        return x_adv, self._make_stats(True, current_loss)
        
        # Failed
        return x_adv, self._make_stats(False, current_loss)
    
    def _make_stats(self, success: bool, final_loss: float) -> dict:
        """Create stats dictionary."""
        return {
            'success': success,
            'total_queries': self.query_count,
            'gradient_queries': self.gradient_queries,
            'coimage_queries': self.coimage_queries,
            'final_loss': final_loss,
            'trust_scores': self.surrogate_trust.cpu().numpy().tolist(),
            'surrogate_avg_alignment': self._get_avg_alignments()
        }
    
    def _get_avg_alignments(self) -> List[float]:
        """Get average alignment per surrogate."""
        alignments = []
        for i in range(self.num_surrogates):
            if self.surrogate_update_count[i] > 0:
                avg = (self.surrogate_alignment_sum[i] / self.surrogate_update_count[i]).item()
            else:
                avg = 0.0
            alignments.append(avg)
        return alignments


class GFCSWeightedAlignmentAblation(GFCSWeightedAlignment):
    """Ablation variants."""
    
    @classmethod
    def only_adaptive_weighting(cls, *args, **kwargs):
        kwargs['use_adaptive_weighting'] = True
        kwargs['use_smart_ods'] = False
        return cls(*args, **kwargs)
    
    @classmethod
    def only_smart_ods(cls, *args, **kwargs):
        kwargs['use_adaptive_weighting'] = False
        kwargs['use_smart_ods'] = True
        return cls(*args, **kwargs)
    
    @classmethod
    def baseline(cls, *args, **kwargs):
        kwargs['use_adaptive_weighting'] = False
        kwargs['use_smart_ods'] = False
        return cls(*args, **kwargs)