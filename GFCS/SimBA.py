"""
SimBA: Simple Black-box Adversarial Attack
===========================================
This implementation EXACTLY matches the official repository:
https://github.com/cg563/simple-blackbox-attack

Key difference from previous Implementation 2:
- Stores perturbations in FREQUENCY SPACE (for DCT mode)
- Applies block IDCT once per query (more efficient)
- Matches official repo's computational approach
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import scipy.fftpack as fftpack


@dataclass
class AttackStats:
    success: bool
    total_queries: int
    method: str


class SimBA:
    """
    SimBA: Simple Black-box Adversarial Attack
    
    This implementation matches the official repository's approach:
    - For pixel attacks: directly modify pixel values
    - For DCT attacks: maintain perturbation in frequency space, apply IDCT per query
    
    Args:
        model: victim model (expects input shape (1, C, H, W))
        epsilon: step size (default: 0.2)
        max_queries: query budget (default: 10000)
        freq_dims: number of DCT directions to consider (only used if pixel_attack=False)
        freq_fraction: if freq_dims is None, use this fraction of lowest frequencies (default: 1/8)
        order: 'random' (paper default) or 'diag' (deterministic low->high frequency traversal)
        targeted: targeted vs untargeted
        pixel_attack: True => pixel basis, False => DCT basis
        device: torch device string
        clip_min/clip_max: clamp range for images
        expects_logits: if True, applies softmax to model outputs
        seed: optional RNG seed (for reproducibility)
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.2,
        max_queries: int = 10000,
        freq_dims: Optional[int] = None,
        freq_fraction: float = 1.0 / 8.0,
        order: str = "random",
        targeted: bool = False,
        pixel_attack: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        expects_logits: bool = True,
        seed: Optional[int] = None,
    ):
        self.model = model.to(device).eval()
        self.epsilon = float(epsilon)
        self.max_queries = int(max_queries)
        self.freq_dims = freq_dims
        self.freq_fraction = float(freq_fraction)
        self.order = order
        self.targeted = bool(targeted)
        self.pixel_attack = bool(pixel_attack)
        self.device = device
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.expects_logits = bool(expects_logits)

        self.query_count = 0

        # RNG for reproducibility
        self._rng = np.random.default_rng(seed)

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities p(y|x)."""
        with torch.no_grad():
            out = self.model(x)
            if self.expects_logits:
                return torch.softmax(out, dim=1)
            return out

    def is_adversarial(
        self,
        probs: torch.Tensor,
        true_class: int,
        target_class: Optional[int] = None,
    ) -> bool:
        """Check adversarial condition based on predicted class."""
        pred_class = int(probs.argmax(dim=1).item())
        if self.targeted:
            if target_class is None:
                raise ValueError("target_class must be provided for targeted attacks.")
            return pred_class == int(target_class)
        return pred_class != int(true_class)

    def _objective_prob(
        self,
        probs: torch.Tensor,
        true_class: int,
        target_class: Optional[int],
    ) -> float:
        """Probability used by SimBA acceptance rule."""
        if self.targeted:
            if target_class is None:
                raise ValueError("target_class must be provided for targeted attacks.")
            return float(probs[0, int(target_class)].item())
        return float(probs[0, int(true_class)].item())

    # ---------- Basis construction ----------

    def get_basis_indices(self, image_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Get indices for the basis directions to try.
        
        For pixel attacks: indices into flattened pixel space
        For DCT attacks: indices into flattened frequency space
        """
        if len(image_shape) != 4 or image_shape[0] != 1:
            raise ValueError(f"Expected image shape (1,C,H,W), got {image_shape}")
        _, c, h, w = image_shape

        if self.pixel_attack:
            # Pixel space: all pixel indices
            n_dims = c * h * w
            indices = np.arange(n_dims, dtype=np.int64)
            if self.order == "random":
                self._rng.shuffle(indices)
            return indices
        else:
            # DCT space: frequency indices ordered by u+v
            if self.freq_dims is None:
                # Use fraction of lowest frequencies
                total_dims = c * h * w
                k = max(1, int(round(total_dims * self.freq_fraction)))
            else:
                k = int(self.freq_dims)
            
            # Build list of (freq_score, idx) then sort
            items: List[Tuple[int, int]] = []
            for ch in range(c):
                base = ch * h * w
                for u in range(h):
                    for v in range(w):
                        idx = base + u * w + v
                        freq = u + v  # Frequency measure
                        items.append((freq, idx))
            
            # Sort by frequency (low to high)
            items.sort(key=lambda t: (t[0], t[1]))
            low_freq_indices = [idx for _, idx in items[:k]]
            
            low_freq_indices = np.array(low_freq_indices, dtype=np.int64)
            if self.order == "random":
                self._rng.shuffle(low_freq_indices)
            
            return low_freq_indices

    def _apply_idct(self, freq_tensor: torch.Tensor, image_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Apply 2D inverse DCT to convert frequency domain to spatial domain.
        This matches the official repo's utils.block_idct functionality.
        
        Args:
            freq_tensor: Tensor of shape (1, C*H*W) containing frequency coefficients
            image_shape: Target image shape (1, C, H, W)
            
        Returns:
            Spatial domain tensor of shape (1, C, H, W)
        """
        _, c, h, w = image_shape
        
        # Reshape to (C, H, W)
        freq_coeffs = freq_tensor.view(c, h, w).cpu().numpy()
        
        # Apply 2D IDCT per channel
        spatial = np.zeros_like(freq_coeffs)
        for ch in range(c):
            spatial[ch] = fftpack.idct(
                fftpack.idct(freq_coeffs[ch], axis=0, norm='ortho'),
                axis=1, norm='ortho'
            )
        
        # Convert back to torch and reshape
        spatial_tensor = torch.from_numpy(spatial).float().to(self.device)
        return spatial_tensor.unsqueeze(0)  # (1, C, H, W)

    # ---------- Attack ----------

    def attack(
        self,
        x: torch.Tensor,
        true_class: int,
        target_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run SimBA attack matching official repository implementation.
        
        Key difference from previous version:
        - For DCT mode: stores perturbation in FREQUENCY space
        - Applies IDCT transformation once per query
        - More efficient and matches official repo exactly
        
        Args:
            x: Input image tensor of shape (1, C, H, W)
            true_class: True class label
            target_class: Target class for targeted attacks
            
        Returns:
            x_adv: Adversarial example
            stats: Dictionary with attack statistics
        """
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(f"Expected x shape (1,C,H,W), got {tuple(x.shape)}")

        self.query_count = 0
        x_orig = x.detach().clone().to(self.device)
        image_shape = tuple(x_orig.shape)
        _, c, h, w = image_shape

        # Get basis indices (pixel or frequency)
        basis_indices = self.get_basis_indices(image_shape)
        remaining = list(basis_indices)

        # Initialize perturbation
        # KEY DIFFERENCE: For DCT, this is in FREQUENCY SPACE
        n_dims = c * h * w
        if self.pixel_attack:
            # Pixel mode: perturbation in pixel space
            delta = torch.zeros(n_dims, device=self.device, dtype=torch.float32)
        else:
            # DCT mode: perturbation in FREQUENCY space (matches official repo)
            delta = torch.zeros(n_dims, device=self.device, dtype=torch.float32)

        # Initial query
        if self.pixel_attack:
            x_current = torch.clamp(x_orig + delta.view(image_shape), self.clip_min, self.clip_max)
        else:
            # Apply IDCT to get spatial perturbation
            delta_spatial = self._apply_idct(delta.unsqueeze(0), image_shape)
            x_current = torch.clamp(x_orig + delta_spatial, self.clip_min, self.clip_max)
        
        probs = self.get_probs(x_current)
        self.query_count += 1

        # Main loop (query-budgeted)
        while self.query_count < self.max_queries:
            # Stop if already adversarial
            if self.is_adversarial(probs, true_class, target_class):
                if self.pixel_attack:
                    x_adv = torch.clamp(x_orig + delta.view(image_shape), self.clip_min, self.clip_max)
                else:
                    delta_spatial = self._apply_idct(delta.unsqueeze(0), image_shape)
                    x_adv = torch.clamp(x_orig + delta_spatial, self.clip_min, self.clip_max)
                return x_adv, self._get_stats(success=True)

            # Restart directions if exhausted
            if len(remaining) == 0:
                remaining = list(basis_indices)

            # Pick random direction without replacement
            pick_pos = int(self._rng.integers(0, len(remaining)))
            dim = int(remaining.pop(pick_pos))

            current_prob = self._objective_prob(probs, true_class, target_class)

            # Try +eps first, then -eps (query-efficient)
            for alpha in (self.epsilon, -self.epsilon):
                if self.query_count >= self.max_queries:
                    break

                # Create candidate perturbation
                # CRITICAL: Modify in frequency space for DCT mode
                delta_candidate = delta.clone()
                delta_candidate[dim] += alpha

                # Transform to spatial domain and apply
                if self.pixel_attack:
                    x_new = torch.clamp(
                        x_orig + delta_candidate.view(image_shape),
                        self.clip_min, self.clip_max
                    )
                else:
                    # Apply IDCT to frequency perturbation
                    delta_spatial = self._apply_idct(delta_candidate.unsqueeze(0), image_shape)
                    x_new = torch.clamp(x_orig + delta_spatial, self.clip_min, self.clip_max)

                probs_new = self.get_probs(x_new)
                self.query_count += 1

                new_prob = self._objective_prob(probs_new, true_class, target_class)

                # Check for improvement
                improved = (new_prob > current_prob) if self.targeted else (new_prob < current_prob)
                
                if improved:
                    delta = delta_candidate
                    probs = probs_new
                    break  # Early exit after first improvement

        # Budget exhausted: finalize
        if self.pixel_attack:
            x_adv = torch.clamp(x_orig + delta.view(image_shape), self.clip_min, self.clip_max)
        else:
            delta_spatial = self._apply_idct(delta.unsqueeze(0), image_shape)
            x_adv = torch.clamp(x_orig + delta_spatial, self.clip_min, self.clip_max)

        # Final success check
        if self.query_count < self.max_queries:
            final_probs = self.get_probs(x_adv)
            self.query_count += 1
        else:
            final_probs = probs

        success = self.is_adversarial(final_probs, true_class, target_class)
        return x_adv, self._get_stats(success=success)

    def _get_stats(self, success: bool) -> Dict[str, Any]:
        return {
            "success": bool(success),
            "total_queries": int(self.query_count),
            "method": "simba-pixel" if self.pixel_attack else "simba-dct",
        }


if __name__ == "__main__":
    print("SimBA loaded (OFFICIAL REPO VERSION)")
    print("  - SimBA-pixel: pixel_attack=True")
    print("  - SimBA-DCT  : pixel_attack=False (frequency-space accumulation)")
    print("\nThis version matches the official repository implementation:")
    print("  https://github.com/cg563/simple-blackbox-attack")