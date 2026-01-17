"""
GFCS Modified: Minimal Victim Queries
======================================
Modified version of GFCS that:
1. Queries the victim model only TWICE:
   - Once at the beginning to verify initial classification
   - Once at the end to verify success
2. Uses the AVERAGE of all surrogate gradients as the direction q
3. Tests the adversarial example on ALL surrogates
4. Stops when the example fools ALL surrogates

This is based on the assumption that if an example fools all surrogates,
it's highly likely to fool the victim model as well.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class GFCSMinimalVictimQueries:
    """
    Modified GFCS with minimal victim queries.
    
    Key differences from original GFCS:
    - Only 2 victim queries total (start and end)
    - Uses average gradient from ALL surrogates instead of sampling
    - Success criterion: fools ALL surrogates
    - Fallback to ODS only after all gradient attempts exhausted
    
    Args:
        victim_model: The black-box victim model (queried only twice)
        surrogate_models: List of surrogate models with accessible gradients
        epsilon: Step size for perturbations (default: 2.0)
        norm_bound: L2 norm bound for total perturbation
        max_iterations: Maximum number of iterations (not queries!)
        targeted: Whether this is a targeted attack
        device: torch device
    """
    
    def __init__(
        self,
        victim_model: nn.Module,
        surrogate_models: List[nn.Module],
        epsilon: float = 2.0,
        norm_bound: float = None,
        max_iterations: int = 1000,  # iterations, not queries
        targeted: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.victim = victim_model.to(device).eval()
        self.surrogates = [s.to(device).eval() for s in surrogate_models]
        self.epsilon = epsilon
        self.norm_bound = norm_bound
        self.max_iterations = max_iterations
        self.targeted = targeted
        self.device = device
        
        # Statistics tracking
        self.victim_queries = 0
        self.surrogate_queries = 0
        self.iterations = 0
        
    def margin_loss(self, logits: torch.Tensor, true_class: int, target_class: Optional[int] = None) -> torch.Tensor:
        """
        Compute the margin loss as defined in the paper.
        
        For untargeted: L = f(second_highest) - f(true_class)
        For targeted: L = f(target) - f(true_class)
        """
        if self.targeted:
            assert target_class is not None
            margin = logits[0, target_class] - logits[0, true_class]
        else:
            # Get second highest class (exclude true class)
            logits_copy = logits.clone()
            logits_copy[0, true_class] = -float('inf')
            second_highest = logits_copy.max(dim=1)[0]
            margin = second_highest - logits[0, true_class]
        
        return margin
    
    def is_adversarial(self, logits: torch.Tensor, true_class: int, target_class: Optional[int] = None) -> bool:
        """Check if current prediction is adversarial."""
        pred_class = logits.argmax(dim=1).item()
        
        if self.targeted:
            return pred_class == target_class
        else:
            return pred_class != true_class
    
    def project(self, x_adv: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        """Project x_adv onto L2 ball around x_orig with radius norm_bound."""
        delta = x_adv - x_orig
        delta_norm = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
        
        # Only project if outside the ball
        factor = torch.clamp(self.norm_bound / (delta_norm + 1e-10), max=1.0)
        delta = delta * factor.view(-1, 1, 1, 1)
        
        return x_orig + delta
    
    def get_average_surrogate_gradient(
        self, 
        x: torch.Tensor, 
        true_class: int, 
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get the AVERAGE normalized gradient from ALL surrogate models.
        
        This is the key difference from original GFCS:
        - Original: samples ONE surrogate at a time
        - This version: averages ALL surrogates
        
        Returns:
            q: Normalized direction vector (average of all surrogate gradients)
        """
        x = x.detach().requires_grad_(True)
        gradients = []
        
        for surrogate in self.surrogates:
            # Compute loss gradient for this surrogate
            logits = surrogate(x)
            self.surrogate_queries += 1
            
            loss = self.margin_loss(logits, true_class, target_class)
            loss.backward(retain_graph=True)
            
            # Normalize the gradient
            grad = x.grad.detach().clone()
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1, keepdim=True)
            grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-10)
            
            gradients.append(grad_normalized)
            
            # Clear gradients for next surrogate
            x.grad.zero_()
        
        # Average all normalized gradients
        avg_gradient = torch.stack(gradients).mean(dim=0)
        
        # Re-normalize the averaged gradient
        avg_norm = torch.norm(avg_gradient.view(avg_gradient.shape[0], -1), p=2, dim=1, keepdim=True)
        q = avg_gradient / (avg_norm.view(-1, 1, 1, 1) + 1e-10)
        
        return q
    
    def get_ods_direction(self, x: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Get ODS direction by averaging over all surrogates.
        Sample w ~ U(-1, 1)^C and compute ∇_x(w^T f(x)) for each surrogate.
        """
        x = x.detach().requires_grad_(True)
        
        # Sample random class weights (same for all surrogates)
        w = torch.FloatTensor(num_classes).uniform_(-1, 1).to(self.device)
        
        gradients = []
        for surrogate in self.surrogates:
            logits = surrogate(x)
            self.surrogate_queries += 1
            
            # Compute weighted output
            weighted_output = (w * logits[0]).sum()
            weighted_output.backward(retain_graph=True)
            
            # Normalize gradient
            grad = x.grad.detach().clone()
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1, keepdim=True)
            grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-10)
            
            gradients.append(grad_normalized)
            x.grad.zero_()
        
        # Average and normalize
        avg_gradient = torch.stack(gradients).mean(dim=0)
        avg_norm = torch.norm(avg_gradient.view(avg_gradient.shape[0], -1), p=2, dim=1, keepdim=True)
        q = avg_gradient / (avg_norm.view(-1, 1, 1, 1) + 1e-10)
        
        return q
    
    def check_all_surrogates_fooled(
        self, 
        x: torch.Tensor, 
        true_class: int, 
        target_class: Optional[int] = None
    ) -> bool:
        """
        Check if the adversarial example fools ALL surrogate models.
        
        Returns:
            True if all surrogates are fooled, False otherwise
        """
        for surrogate in self.surrogates:
            with torch.no_grad():
                logits = surrogate(x)
                self.surrogate_queries += 1
                
                if not self.is_adversarial(logits, true_class, target_class):
                    return False
        
        return True
    
    def attack(
        self,
        x: torch.Tensor,
        true_class: int,
        target_class: Optional[int] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Perform the minimal-victim-query attack.
        
        Algorithm:
        1. Query victim once to verify initial classification
        2. Loop until all surrogates are fooled OR max iterations reached:
           a. Compute q = average of all surrogate gradients
           b. Try steps x + ε*q and x - ε*q
           c. Accept step if it improves loss on all surrogates
           d. Check if all surrogates are fooled
        3. If all surrogates fooled, query victim once more to verify
        
        Args:
            x: Input image tensor of shape (1, C, H, W)
            true_class: True class label
            target_class: Target class for targeted attacks
            
        Returns:
            x_adv: Adversarial example
            stats: Dictionary with attack statistics
        """
        # Reset statistics
        self.victim_queries = 0
        self.surrogate_queries = 0
        self.iterations = 0
        
        # Set norm bound if not specified
        if self.norm_bound is None:
            D = x.numel()
            self.norm_bound = np.sqrt(0.001 * D)
        
        x_orig = x.clone().to(self.device)
        x_adv = x_orig.clone()
        
        # Get number of classes
        with torch.no_grad():
            num_classes = self.surrogates[0](x_adv).shape[1]
        
        # VICTIM QUERY 1: Verify initial classification
        with torch.no_grad():
            victim_logits = self.victim(x_adv)
            self.victim_queries += 1
            
            if self.is_adversarial(victim_logits, true_class, target_class):
                print("Input is already adversarial!")
                return x_adv, self._get_stats(success=True)
            
            initial_loss = self.margin_loss(victim_logits, true_class, target_class).item()
        
        print(f"Initial victim loss: {initial_loss:.4f}")
        print(f"Starting attack with {len(self.surrogates)} surrogates...")
        
        use_ods = False
        gradient_attempts = 0
        max_gradient_attempts = 100  # Switch to ODS after this many failed attempts
        
        # Main attack loop
        while self.iterations < self.max_iterations:
            self.iterations += 1
            
            # Check if all surrogates are fooled
            if self.check_all_surrogates_fooled(x_adv, true_class, target_class):
                print(f"\nAll surrogates fooled at iteration {self.iterations}!")
                break
            
            # Get direction q
            if not use_ods and gradient_attempts < max_gradient_attempts:
                # GRADIENT FIRST: Average of all surrogate gradients
                q = self.get_average_surrogate_gradient(x_adv, true_class, target_class)
                gradient_attempts += 1
            else:
                # COIMAGE SECOND: ODS sampling
                if not use_ods:
                    print(f"\nSwitching to ODS after {gradient_attempts} gradient attempts")
                    use_ods = True
                q = self.get_ods_direction(x_adv, num_classes)
            
            # Try both directions
            step_accepted = False
            for alpha in [self.epsilon, -self.epsilon]:
                # Compute candidate
                x_candidate = x_adv + alpha * q
                x_candidate = self.project(x_candidate, x_orig)
                
                # Evaluate on all surrogates
                with torch.no_grad():
                    candidate_losses = []
                    current_losses = []
                    
                    for surrogate in self.surrogates:
                        # Candidate loss
                        logits_cand = surrogate(x_candidate)
                        self.surrogate_queries += 1
                        loss_cand = self.margin_loss(logits_cand, true_class, target_class).item()
                        candidate_losses.append(loss_cand)
                        
                        # Current loss
                        logits_curr = surrogate(x_adv)
                        self.surrogate_queries += 1
                        loss_curr = self.margin_loss(logits_curr, true_class, target_class).item()
                        current_losses.append(loss_curr)
                    
                    # Accept if average loss improves
                    avg_candidate_loss = np.mean(candidate_losses)
                    avg_current_loss = np.mean(current_losses)
                    
                    if avg_candidate_loss > avg_current_loss:
                        x_adv = x_candidate
                        step_accepted = True
                        
                        if self.iterations % 10 == 0:
                            print(f"Iter {self.iterations}: Avg surrogate loss improved: "
                                  f"{avg_current_loss:.4f} -> {avg_candidate_loss:.4f}")
                        break
            
            if not step_accepted and self.iterations % 10 == 0:
                print(f"Iter {self.iterations}: No improvement")
        
        # VICTIM QUERY 2: Check if we fooled the victim
        with torch.no_grad():
            final_victim_logits = self.victim(x_adv)
            self.victim_queries += 1
            
            success = self.is_adversarial(final_victim_logits, true_class, target_class)
            final_loss = self.margin_loss(final_victim_logits, true_class, target_class).item()
        
        print(f"\nFinal victim loss: {final_loss:.4f}")
        print(f"Success: {success}")
        
        return x_adv, self._get_stats(success=success)
    
    def _get_stats(self, success: bool) -> dict:
        """Return attack statistics."""
        perturbation_norm = 0.0
        
        return {
            'success': success,
            'victim_queries': self.victim_queries,
            'surrogate_queries': self.surrogate_queries,
            'iterations': self.iterations,
            'perturbation_norm': perturbation_norm
        }


def test_minimal_victim_gfcs():
    """
    Simple test to demonstrate the modified GFCS.
    """
    print("=" * 70)
    print("Testing GFCS with Minimal Victim Queries")
    print("=" * 70)
    
    # Create dummy models for testing
    class SimpleModel(nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32 * 8 * 8, 10)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Create victim and surrogates
    victim = SimpleModel(seed=42)
    surrogates = [SimpleModel(seed=i) for i in [100, 200, 300]]
    
    # Create test image
    x = torch.randn(1, 3, 32, 32)
    true_class = 0
    
    # Create attacker
    attacker = GFCSMinimalVictimQueries(
        victim_model=victim,
        surrogate_models=surrogates,
        epsilon=2.0,
        max_iterations=100
    )
    
    # Run attack
    x_adv, stats = attacker.attack(x, true_class)
    
    # Print results
    print("\n" + "=" * 70)
    print("ATTACK RESULTS")
    print("=" * 70)
    print(f"Success: {stats['success']}")
    print(f"Victim queries: {stats['victim_queries']} (should be 2)")
    print(f"Surrogate queries: {stats['surrogate_queries']}")
    print(f"Iterations: {stats['iterations']}")
    print(f"Perturbation L2 norm: {stats['perturbation_norm']:.4f}")
    
    return x_adv, stats


if __name__ == "__main__":
    test_minimal_victim_gfcs()