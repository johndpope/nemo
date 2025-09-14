#!/usr/bin/env python3
"""Visualize expressions from actual training checkpoint"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append('.')
sys.path.append('nemo')

from visualize_expressions_and_warps import create_expression_candles

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize expressions from checkpoint')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints_overfit/best_checkpoint.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--output-dir', type=str, default='nemo/visualization_output',
                       help='Directory to save visualizations')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading checkpoint: {args.checkpoint}")

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found at {args.checkpoint}")
        # Try alternative path
        alt_path = "checkpoints_overfit/checkpoint_epoch_10.pt"
        if Path(alt_path).exists():
            print(f"Using alternative checkpoint: {alt_path}")
            args.checkpoint = alt_path
        else:
            print("No checkpoint found. Please train the model first.")
            return

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Look for expression embeddings in validation outputs
    expr_data = None
    if 'validation_outputs' in checkpoint:
        print("Found validation_outputs in checkpoint")
        if 'expression_embed' in checkpoint['validation_outputs']:
            expr_data = checkpoint['validation_outputs']['expression_embed']
            print(f"Found expression embeddings: {expr_data.shape}")
        else:
            print("Available keys in validation_outputs:", list(checkpoint['validation_outputs'].keys()))

    # Also check training batch
    if expr_data is None and 'last_batch' in checkpoint:
        print("Checking last_batch for expression data...")
        if 'expression_embed' in checkpoint['last_batch']:
            expr_data = checkpoint['last_batch']['expression_embed']
            print(f"Found expression embeddings in last_batch: {expr_data.shape}")

    if expr_data is not None:
        # Convert to numpy if tensor
        if isinstance(expr_data, torch.Tensor):
            expr_data = expr_data.detach().cpu().numpy()

        # Handle different shapes
        if len(expr_data.shape) == 3:  # [B, T, D]
            # Take first sample
            expr_data = expr_data[0]
            print(f"Using first sample from batch, shape: {expr_data.shape}")

        # Check variation
        is_constant = np.allclose(expr_data[0], expr_data, atol=1e-5)
        print(f"\nExpression variation analysis:")
        print(f"  Constant across frames: {is_constant}")

        if not is_constant:
            frame_diff = np.diff(expr_data, axis=0)
            diff_norm = np.linalg.norm(frame_diff, axis=-1)
            print(f"  Mean frame-to-frame difference: {diff_norm.mean():.6f}")
            print(f"  Max frame-to-frame difference: {diff_norm.max():.6f}")
            print(f"  Min frame-to-frame difference: {diff_norm.min():.6f}")
        else:
            print("  WARNING: Expressions are constant across all frames!")
            print("  This suggests expressions are only extracted from identity image.")

        # Create visualization
        fig = create_expression_candles(
            expr_data,
            title="Checkpoint Expression Embeddings",
            save_path=output_dir / "checkpoint_expression_candles.png"
        )
        plt.close(fig)

        print(f"\nVisualization saved to {output_dir}/checkpoint_expression_candles.png")

        # Also save raw statistics for debugging
        stats_file = output_dir / "expression_stats.txt"
        with open(stats_file, 'w') as f:
            f.write(f"Expression shape: {expr_data.shape}\n")
            f.write(f"Constant across frames: {is_constant}\n")
            if not is_constant:
                f.write(f"Mean frame-to-frame diff: {diff_norm.mean():.6f}\n")
                f.write(f"Max frame-to-frame diff: {diff_norm.max():.6f}\n")
                f.write(f"Min frame-to-frame diff: {diff_norm.min():.6f}\n")
            f.write(f"\nPer-dimension statistics:\n")
            f.write(f"Mean: {expr_data.mean(axis=0)[:10]}...\n")  # First 10 dims
            f.write(f"Std: {expr_data.std(axis=0)[:10]}...\n")
        print(f"Statistics saved to {stats_file}")

    else:
        print("\nNo expression embeddings found in checkpoint.")
        print("Available top-level keys:", list(checkpoint.keys()))

if __name__ == "__main__":
    main()