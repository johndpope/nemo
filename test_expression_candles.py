#!/usr/bin/env python3
"""Test expression candle visualization with synthetic data"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append('.')
sys.path.append('nemo')

from visualize_expressions_and_warps import create_expression_candles, visualize_warping_field

def generate_synthetic_expressions(frames=50, dims=128):
    """Generate synthetic expression data with variations"""
    # Base expression
    base = np.random.randn(dims) * 0.1

    # Add temporal variations
    expressions = []
    for t in range(frames):
        # Add sinusoidal variations at different frequencies
        variation = (
            0.3 * np.sin(2 * np.pi * t / 10) * np.random.randn(dims) +  # Fast
            0.2 * np.sin(2 * np.pi * t / 25) * np.random.randn(dims) +  # Medium
            0.1 * np.sin(2 * np.pi * t / 50) * np.random.randn(dims)    # Slow
        )
        expressions.append(base + variation)

    return np.array(expressions)  # [T, D]

def generate_synthetic_warp(height=256, width=256):
    """Generate synthetic warping field"""
    # Create grid
    y, x = np.mgrid[0:height, 0:width]

    # Add some deformation
    center_y, center_x = height/2, width/2
    dist_y = (y - center_y) / height
    dist_x = (x - center_x) / width

    # Radial distortion
    r = np.sqrt(dist_y**2 + dist_x**2)

    # Create warping field
    warp = np.zeros((height, width, 2))
    warp[..., 0] = 10 * dist_x * np.exp(-2*r)  # Horizontal displacement
    warp[..., 1] = 10 * dist_y * np.exp(-2*r)  # Vertical displacement

    return warp

def main():
    # Create output directory
    output_dir = Path("nemo/visualization_output")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Generating synthetic expression data...")

    # Test 1: Expression with variations (like real data)
    expr_with_variation = generate_synthetic_expressions(frames=50, dims=128)

    # Add some noise to simulate real extraction
    expr_with_variation += np.random.randn(*expr_with_variation.shape) * 0.01

    print(f"Expression shape: {expr_with_variation.shape}")

    # Check variation
    is_constant = np.allclose(expr_with_variation[0], expr_with_variation, atol=1e-5)
    print(f"Constant across frames: {is_constant}")

    if not is_constant:
        frame_diff = np.diff(expr_with_variation, axis=0)
        diff_norm = np.linalg.norm(frame_diff, axis=-1)
        print(f"Mean frame-to-frame difference: {diff_norm.mean():.6f}")
        print(f"Max frame-to-frame difference: {diff_norm.max():.6f}")

    # Create candles visualization
    fig1 = create_expression_candles(
        expr_with_variation,
        title="Synthetic Expression (With Variation)",
        save_path=output_dir / "synthetic_expression_candles_variation.png"
    )
    plt.close(fig1)

    # Test 2: Constant expression (for comparison)
    expr_constant = np.repeat(expr_with_variation[0:1], 50, axis=0)

    fig2 = create_expression_candles(
        expr_constant,
        title="Synthetic Expression (Constant)",
        save_path=output_dir / "synthetic_expression_candles_constant.png"
    )
    plt.close(fig2)

    # Test 3: Warping field
    print("\nGenerating synthetic warping field...")
    warp = generate_synthetic_warp()

    fig3 = visualize_warping_field(
        warp,
        title="Synthetic Warping Field",
        save_path=output_dir / "synthetic_warp_field.png"
    )
    plt.close(fig3)

    print(f"\nVisualizations saved to {output_dir}")
    print("\nGenerated files:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()