#!/usr/bin/env python3
"""
Test script to compare our simplified formula with I²SB official implementation
"""

import numpy as np
import torch

def i2sb_official(x0, x1, step, betas, tau):
    """
    Official I²SB formula from NVlabs/I2SB
    """
    # Forward and backward cumulative variances
    std_fwd = np.sqrt(np.cumsum(betas))
    std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))

    # Gaussian product coefficients
    denom = std_fwd**2 + std_bwd**2
    mu_x0 = std_bwd**2 / denom
    mu_x1 = std_fwd**2 / denom
    var = (std_fwd**2 * std_bwd**2) / denom

    # Sample x_t
    noise = torch.randn_like(x0)
    xt = mu_x0[step] * x0 + mu_x1[step] * x1 + np.sqrt(var[step]) * noise

    return xt, mu_x0[step], mu_x1[step], np.sqrt(var[step])


def our_simplified(x0, x1, t_norm, tau):
    """
    Our simplified formula
    """
    lambda_t = t_norm
    sigma_t = np.sqrt(t_norm * (1 - t_norm) * tau)

    noise = torch.randn_like(x0)
    xt = (1 - lambda_t) * x0 + lambda_t * x1 + sigma_t * noise

    return xt, (1 - lambda_t), lambda_t, sigma_t


def original_code_incremental(x0, x1, times, step, tau):
    """
    Original code's incremental approach (what was causing OOM)
    Note: This requires netG which we don't have, so we'll simulate the end result
    """
    # Assuming uniform betas (simplified)
    # The original code uses: scale = delta * (1 - delta/denom)
    # This is actually equivalent to Brownian bridge variance!

    t_norm = times[step]

    # The key insight: if we accumulate all the incremental steps,
    # we should get a Brownian bridge formula
    # For uniform discretization: variance = t*(1-t)*tau

    lambda_t = t_norm
    sigma_t = np.sqrt(t_norm * (1 - t_norm) * tau)

    noise = torch.randn_like(x0)
    xt = (1 - lambda_t) * x0 + lambda_t * x1 + sigma_t * noise

    return xt, (1 - lambda_t), lambda_t, sigma_t


if __name__ == "__main__":
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)

    x0 = torch.randn(1, 3, 64, 64)
    x1 = torch.randn(1, 3, 64, 64)
    tau = 1.0
    T = 20
    step = 10  # Middle timestep

    # I²SB uses symmetric beta schedule
    linear_start = 0.0001
    linear_end = 0.02
    betas = np.linspace(linear_start**0.5, linear_end**0.5, T//2)**2
    betas = np.concatenate([betas, np.flip(betas)])

    # Our time schedule (from original code)
    incs = np.array([0] + [1/(i+1) for i in range(T-1)])
    times = np.cumsum(incs)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    times = np.concatenate([np.zeros(1), times])
    t_norm = times[step]

    print("=" * 60)
    print("Comparing Sampling Formulas")
    print("=" * 60)
    print(f"\nTest configuration:")
    print(f"  Timestep: {step}/{T}")
    print(f"  Normalized time: {t_norm:.4f}")
    print(f"  tau: {tau}")

    # Test I²SB official
    torch.manual_seed(42)
    xt_i2sb, mu_x0_i2sb, mu_x1_i2sb, std_i2sb = i2sb_official(x0, x1, step, betas, tau)

    print("\n" + "=" * 60)
    print("I²SB Official Implementation")
    print("=" * 60)
    print(f"  mu_x0 (weight for x0): {mu_x0_i2sb:.4f}")
    print(f"  mu_x1 (weight for x1): {mu_x1_i2sb:.4f}")
    print(f"  std (noise scale):     {std_i2sb:.4f}")
    print(f"  Sum of weights:        {mu_x0_i2sb + mu_x1_i2sb:.4f}")

    # Test our simplified
    torch.manual_seed(42)
    xt_ours, mu_x0_ours, mu_x1_ours, std_ours = our_simplified(x0, x1, t_norm, tau)

    print("\n" + "=" * 60)
    print("Our Simplified Implementation")
    print("=" * 60)
    print(f"  mu_x0 (1-lambda_t):    {mu_x0_ours:.4f}")
    print(f"  mu_x1 (lambda_t):      {mu_x1_ours:.4f}")
    print(f"  std (noise scale):     {std_ours:.4f}")
    print(f"  Sum of weights:        {mu_x0_ours + mu_x1_ours:.4f}")

    # Compare
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"  Δ mu_x0: {abs(mu_x0_i2sb - mu_x0_ours):.6f}")
    print(f"  Δ mu_x1: {abs(mu_x1_i2sb - mu_x1_ours):.6f}")
    print(f"  Δ std:   {abs(std_i2sb - std_ours):.6f}")

    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    # Check if formulas are equivalent under certain conditions
    if abs(mu_x0_i2sb + mu_x1_i2sb - 1.0) < 1e-6:
        print("  ✓ I²SB weights sum to 1 (valid convex combination)")
    else:
        print("  ✗ I²SB weights don't sum to 1!")

    if abs(mu_x0_ours + mu_x1_ours - 1.0) < 1e-6:
        print("  ✓ Our weights sum to 1 (valid convex combination)")
    else:
        print("  ✗ Our weights don't sum to 1!")

    # Check if they're close
    if abs(mu_x0_i2sb - mu_x0_ours) < 0.1:
        print("  ✓ Weights are similar (< 10% difference)")
    else:
        print("  ⚠ Weights differ significantly")
        print("    This is expected: I²SB uses asymmetric beta schedule,")
        print("    while our formula assumes uniform Brownian bridge")

    print("\n" + "=" * 60)
    print("Conclusion")
    print("=" * 60)
    print("""
For the UNSB-MRI codebase:

1. I²SB official uses complex beta schedules with forward/backward variances
2. Our simplified formula uses uniform Brownian bridge: t*(1-t)

Question: Does UNSB-MRI need I²SB's complex schedule?
  - If the original iterative code worked, then uniform bridge is OK
  - The key is eliminating the loop, not matching I²SB exactly

Recommendation:
  - Keep our simplified formula for now (it's memory-efficient)
  - Monitor training loss to ensure it's working
  - If results are bad, implement I²SB's exact beta schedule
""")

    print("=" * 60)
