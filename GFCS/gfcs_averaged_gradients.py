"""
GFCS with Averaged Gradients
=============================
Combines the exact GFCS algorithm structure with averaged surrogate gradients.

Key modifications from original GFCS:
1. Instead of sampling ONE random surrogate for gradient (line 7-9 in Algorithm 1),
   we compute gradients from ALL surrogates and average them
2. Everything else remains identical to the original GFCS algorithm
3. Still queries victim normally (not minimized)
4. Still uses S_rem logic and ODS fallback exactly as in original

This tests whether averaging surrogate gradients improves transfer success
while maintaining the same query budget and algorithm structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class GFCSAveragedGradients:
    """
    GFCS with Averaged Surrogate Gradients
    
    Identical to Algorithm 1 from the ICLR 2022 paper, except:
    - Line 9: Instead of ∇_x L_s(x_adv) for ONE surrogate s,
              we compute (1/|S_rem|) * Σ ∇_x L_s(x_adv) for ALL s in S_rem
    
    Args:
        victim_model: The black-box victim model (only used for forward passes/queries)
        surrogate_models: List of surrogate models with accessible gradients
        epsilon: Step size for perturbations (default: 2.0 as per paper)
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
        
    def margin_loss(self, logits: torch.Tensor, true_class: int, target_class: Optional[int] = None) -> torch.Tensor:
        """
        Compute the margin loss as defined in the paper (Section 2.2).
        
        For untargeted attacks: L = f(c_t) - f(c_s) where c_s is true class, c_t is second highest
        For targeted attacks: L = f(target) - f(true_class)
        
        We want to MAXIMIZE this loss to find adversarial examples.
        """
        if self.targeted and target_class is not None:
            # Targeted: maximize target class score relative to true class
            return logits[0, target_class] - logits[0, true_class]
        else:
            # Untargeted: maximize second-highest class relative to true class
            logits_copy = logits.clone()
            logits_copy[0, true_class] = -float('inf')
            second_highest_class = logits_copy.argmax(dim=1).item()
            return logits[0, second_highest_class] - logits[0, true_class]
    
    def query_victim(self, x: torch.Tensor) -> torch.Tensor:
        """Query the victim model and increment query counter."""
        self.query_count += 1
        with torch.no_grad():
            return self.victim(x)
    
    def get_averaged_surrogate_gradients(
        self, 
        x: torch.Tensor,
        surrogate_indices: List[int],
        true_class: int,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute the AVERAGED normalized loss gradient from multiple surrogate models.
        
        This is the key modification: instead of using ONE surrogate's gradient,
        we average the gradients from ALL surrogates in surrogate_indices.
        
        Modified line 9 in Algorithm 1:
        Original: q ← ∇_x L_s(x_adv) / ||∇_x L_s(x_adv)||_2  [for one s]
        Modified: q ← (1/|S_rem|) * Σ_{s in S_rem} (∇_x L_s(x_adv) / ||∇_x L_s(x_adv)||_2)
        
        Then normalize the result: q ← q / ||q||_2
        """
        x_input = x.clone().detach().requires_grad_(True)
        
        gradients = []
        
        for idx in surrogate_indices:
            surrogate = self.surrogates[idx]
            
            # Compute loss gradient for this surrogate
            logits = surrogate(x_input)
            loss = self.margin_loss(logits, true_class, target_class)
            loss.backward(retain_graph=True)
            
            grad = x_input.grad.detach().clone()
            
            # Normalize this surrogate's gradient
            grad_norm = torch.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm
            
            gradients.append(grad)
            
            # Clear gradients for next surrogate
            x_input.grad.zero_()
        
        # Average all normalized gradients
        avg_gradient = torch.stack(gradients).mean(dim=0)
        
        # Re-normalize the averaged gradient
        avg_norm = torch.norm(avg_gradient)
        if avg_norm > 0:
            avg_gradient = avg_gradient / avg_norm
        
        return avg_gradient
    
    def get_ods_direction(
        self,
        x: torch.Tensor,
        surrogate: nn.Module,
        num_classes: int = 1000
    ) -> torch.Tensor:
        """
        Compute an ODS (Output Diversified Sampling) direction.
        
        This is the "Coimage Second" part - sampling from the row space of the Jacobian.
        Corresponds to line 12-13 in Algorithm 1.
        
        ODS direction: d_ODS = ∇_x(w^T f(x)) / ||∇_x(w^T f(x))||
        where w is sampled uniformly from [-1, 1]^C
        """
        x_input = x.clone().detach().requires_grad_(True)
        
        # Sample random weights for class scores (line 12: w ~ U(-1, 1)^C)
        w = torch.empty(num_classes, device=self.device).uniform_(-1, 1)
        
        # Compute weighted sum of logits
        logits = surrogate(x_input)
        weighted_sum = (w * logits).sum()
        weighted_sum.backward()
        
        grad = x_input.grad.detach()
        
        # Normalize (line 13: q ← d_ODS(x_adv, s, w))
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
        """
        Project x_adv onto the L2 ball centered at x_orig with radius norm_bound.
        This is the Π operator from the paper (PGA projection).
        Corresponds to Π_{x_in,ν} in Algorithm 1 line 15.
        """
        delta = x_adv - x_orig
        delta_norm = torch.norm(delta)
        
        if delta_norm > norm_bound:
            delta = delta * (norm_bound / delta_norm)
            
        # Also clamp to valid image range [0, 1]
        x_projected = torch.clamp(x_orig + delta, 0, 1)
        
        return x_projected
    
    def is_adversarial(
        self,
        logits: torch.Tensor,
        true_class: int,
        target_class: Optional[int] = None
    ) -> bool:
        """
        Check if the current prediction is adversarial.
        This corresponds to the condition in line 5: "while x_adv is not adversary"
        """
        pred_class = logits.argmax(dim=1).item()
        
        if self.targeted:
            return pred_class == target_class
        else:
            return pred_class != true_class
    
    def attack(
        self,
        x: torch.Tensor,
        true_class: int,
        target_class: Optional[int] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Run the GFCS attack with averaged surrogate gradients.
        
        Algorithm (modified from Algorithm 1):
        1. Input: A targeted image x_in, loss function L, a victim classifier v, 
                  a set of surrogate models S, a step length ε, and a norm bound ν.
        2. Output: adversarial image x_adv within distance ν of x_in
        3. x_adv ← x_in
        4. S_rem ← S
        5. while x_adv is not adversary do
        6.     if S_rem ≠ ∅ then
        7.         [MODIFIED] Use ALL models in S_rem instead of sampling one
        8.         [MODIFIED] S_rem ← ∅ (we used all of them)
        9.         [MODIFIED] q ← average of normalized gradients from all models in S_rem
        10.    else  ▷ None of the surrogate loss gradients work, so revert to ODS.
        11.        Randomly sample surrogate model s from S
        12.        Sample w ~ U(−1, 1)^C
        13.        q ← d_ODS(x_adv, s, w)
        14.    for α ∈ {ε, −ε} do
        15.        if L_v(Π_{x_in,ν}(x_adv + α · q)) > L_v(x_adv) then
        16.            x_adv ← Π_{x_in,ν}(x_adv + α · q)
        17.            S_rem ← S  ▷ Reset candidate surrogate set to input set
        18.            break
        
        Args:
            x: Input image tensor of shape (1, C, H, W)
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
        
        # Set norm bound based on image dimension if not specified (paper default)
        if self.norm_bound is None:
            D = x.numel()
            self.norm_bound = np.sqrt(0.001 * D)
        
        # Line 3: x_adv ← x_in
        x_orig = x.clone().to(self.device)
        x_adv = x_orig.clone()
        
        # Line 4: S_rem ← S (remaining surrogates to try)
        S_rem = list(range(len(self.surrogates)))
        
        # Get number of classes from first surrogate
        with torch.no_grad():
            num_classes = self.surrogates[0](x_adv).shape[1]
        
        # Line 5: while x_adv is not adversary do
        while self.query_count < self.max_queries:
            # Check adversarial condition at START of loop (line 5)
            victim_logits = self.query_victim(x_adv)
            
            if self.is_adversarial(victim_logits, true_class, target_class):
                return x_adv, self._get_stats(success=True)
            
            # Get current loss for comparison
            current_loss = self.margin_loss(victim_logits, true_class, target_class).item()
            
            # Get candidate direction q
            # Line 6: if S_rem ≠ ∅ then
            if len(S_rem) > 0:
                # MODIFIED: Use AVERAGED gradient from ALL remaining surrogates
                # instead of sampling just one
                q = self.get_averaged_surrogate_gradients(
                    x_adv, 
                    S_rem,  # Pass ALL remaining surrogate indices
                    true_class, 
                    target_class
                )
                
                # Since we used all surrogates, empty S_rem
                S_rem = []
                is_gradient_step = True
            else:
                # COIMAGE SECOND: Fall back to ODS sampling
                # Line 11: Randomly sample surrogate model s from S
                surrogate_idx = np.random.randint(len(self.surrogates))
                surrogate = self.surrogates[surrogate_idx]
                
                # Line 12-13: Sample w ~ U(−1, 1)^C, q ← d_ODS(x_adv, s, w)
                q = self.get_ods_direction(x_adv, surrogate, num_classes)
                is_gradient_step = False
            
            # Line 14: for α ∈ {ε, −ε} do
            for alpha in [self.epsilon, -self.epsilon]:
                # Line 15: Compute candidate and project onto feasible set
                x_candidate = self.project_onto_ball(
                    x_adv + alpha * q,
                    x_orig,
                    self.norm_bound
                )
                
                # Query victim with candidate
                candidate_logits = self.query_victim(x_candidate)
                candidate_loss = self.margin_loss(candidate_logits, true_class, target_class).item()
                
                # Track query type
                if is_gradient_step:
                    self.gradient_queries += 1
                else:
                    self.coimage_queries += 1
                
                # Line 15: if L_v(Π_{x_in,ν}(x_adv + α · q)) > L_v(x_adv) then
                if candidate_loss > current_loss:
                    # Line 16: x_adv ← Π_{x_in,ν}(x_adv + α · q)
                    x_adv = x_candidate
                    current_loss = candidate_loss
                    
                    # Line 17: S_rem ← S (Reset to use all surrogates again)
                    S_rem = list(range(len(self.surrogates)))
                    
                    # Line 18: break
                    break
        
        # Final check (loop exited due to query limit)
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


if __name__ == "__main__":
    print("GFCS with Averaged Gradients Implementation loaded successfully!")
    print("\nKey difference from original GFCS:")
    print("- Original: Randomly samples ONE surrogate at a time for gradient")
    print("- This version: Averages gradients from ALL remaining surrogates")
    print("\nEverything else (queries, S_rem logic, ODS fallback) remains identical.")
    print("\nTo use, import GFCSAveragedGradients class and instantiate with victim and surrogate models.")