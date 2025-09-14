#!/usr/bin/env python3
"""Visualize expression embeddings as candles and warping fields"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys
import os

sys.path.append('.')
sys.path.append('nemo')

def create_expression_candles(expression_data, title="Expression Embeddings", save_path=None):
    """
    Create candlestick-like visualization for expression embeddings
    Similar to wandb's expression candles
    """
    if isinstance(expression_data, torch.Tensor):
        expression_data = expression_data.detach().cpu().numpy()

    # Handle different shapes
    if len(expression_data.shape) == 3:  # [B, T, D]
        expression_data = expression_data[0]  # Take first batch

    T, D = expression_data.shape  # [T, D] where T=frames, D=128

    # Create figure with appropriate size
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    # First subplot: Candlestick visualization
    ax1 = axes[0]

    # For each frame, compute statistics across dimensions
    for t in range(T):
        frame_values = expression_data[t]

        # Compute statistics
        mean_val = np.mean(frame_values)
        std_val = np.std(frame_values)
        min_val = np.min(frame_values)
        max_val = np.max(frame_values)
        q25 = np.percentile(frame_values, 25)
        q75 = np.percentile(frame_values, 75)

        # Draw candlestick
        # Box (25th to 75th percentile)
        box = patches.Rectangle((t - 0.3, q25), 0.6, q75 - q25,
                               linewidth=1, edgecolor='blue',
                               facecolor='lightblue', alpha=0.7)
        ax1.add_patch(box)

        # Whiskers (min to max)
        ax1.plot([t, t], [min_val, q25], 'b-', linewidth=0.5)
        ax1.plot([t, t], [q75, max_val], 'b-', linewidth=0.5)
        ax1.plot([t - 0.1, t + 0.1], [min_val, min_val], 'b-', linewidth=0.5)
        ax1.plot([t - 0.1, t + 0.1], [max_val, max_val], 'b-', linewidth=0.5)

        # Mean line
        ax1.plot([t - 0.3, t + 0.3], [mean_val, mean_val], 'r-', linewidth=2)

    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Expression Value')
    ax1.set_title(f'{title} - Candlestick View (per frame statistics)')
    ax1.grid(True, alpha=0.3)

    # Second subplot: Heatmap of all dimensions over time
    ax2 = axes[1]
    im = ax2.imshow(expression_data.T, aspect='auto', cmap='RdBu_r',
                    interpolation='nearest')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Expression Dimension')
    ax2.set_title(f'{title} - All Dimensions Heatmap')
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved expression candles to {save_path}")

    return fig

def visualize_warping_field(warp_field, title="Warping Field", save_path=None):
    """
    Visualize warping/deformation field as a grid or flow field
    """
    if isinstance(warp_field, torch.Tensor):
        warp_field = warp_field.detach().cpu().numpy()

    # Handle different shapes
    if len(warp_field.shape) == 5:  # [B, T, H, W, 2]
        warp_field = warp_field[0, 0]  # Take first batch, first frame
    elif len(warp_field.shape) == 4:  # [B, H, W, 2]
        warp_field = warp_field[0]  # Take first batch
    elif len(warp_field.shape) == 3:  # [H, W, 2]
        pass  # Already in right shape

    H, W = warp_field.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: Horizontal displacement
    ax1 = axes[0]
    im1 = ax1.imshow(warp_field[..., 0], cmap='RdBu_r')
    ax1.set_title(f'{title} - Horizontal Displacement')
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Height')
    plt.colorbar(im1, ax=ax1)

    # Subplot 2: Vertical displacement
    ax2 = axes[1]
    im2 = ax2.imshow(warp_field[..., 1], cmap='RdBu_r')
    ax2.set_title(f'{title} - Vertical Displacement')
    ax2.set_xlabel('Width')
    ax2.set_ylabel('Height')
    plt.colorbar(im2, ax=ax2)

    # Subplot 3: Quiver plot (flow field)
    ax3 = axes[2]

    # Downsample for better visualization
    step = max(H // 20, 1)
    Y, X = np.mgrid[0:H:step, 0:W:step]
    U = warp_field[::step, ::step, 0]
    V = warp_field[::step, ::step, 1]

    # Create quiver plot
    magnitude = np.sqrt(U**2 + V**2)
    ax3.quiver(X, Y, U, -V, magnitude, cmap='viridis', scale_units='xy', scale=1)
    ax3.set_title(f'{title} - Flow Field')
    ax3.set_xlabel('Width')
    ax3.set_ylabel('Height')
    ax3.set_aspect('equal')
    ax3.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved warping field to {save_path}")

    return fig

def main():
    """
    Load and visualize expression embeddings and warps from a checkpoint or dataset
    """
    import argparse
    parser = argparse.ArgumentParser(description='Visualize expressions and warps')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--dataset', action='store_true', help='Load from dataset instead')
    parser.add_argument('--output-dir', type=str, default='visualization_output',
                       help='Directory to save visualizations')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.dataset:
        # Load from dataset
        print("Loading from dataset...")

        # Import after parsing args to avoid loading models unnecessarily
        from vasa_dataset import VASAIntegratedDataset
        from omegaconf import OmegaConf
        import importlib

        # Load volumetric avatar
        emo_config = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
        va_module = importlib.import_module('models.stage_1.volumetric_avatar.va')
        volumetric_avatar = va_module.Model(emo_config, training=False)

        model_path = './logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
        if os.path.exists(model_path):
            model_dict = torch.load(model_path, map_location='cuda', weights_only=False)
            volumetric_avatar.load_state_dict(model_dict, strict=False)
            volumetric_avatar = volumetric_avatar.cuda()
            volumetric_avatar.eval()

        # Create dataset
        dataset = VASAIntegratedDataset(
            video_folder="./test_videos/",
            emo_model=volumetric_avatar,
            max_videos=1,
            window_size=50,
            stride=50,
            context_size=10,
            random_seed=42,
            cache_dir='cache'
        )

        # Get first window
        print("Extracting first window...")
        window_data = dataset[0]

        # Visualize expressions
        if 'expression_embed' in window_data:
            expr = window_data['expression_embed']
            print(f"Expression shape: {expr.shape}")

            # Create candles visualization
            fig = create_expression_candles(
                expr,
                title="Dataset Expression Embeddings",
                save_path=output_dir / "expression_candles_dataset.png"
            )
            plt.show()

            # Check variation
            if isinstance(expr, torch.Tensor):
                expr_np = expr.numpy()
            else:
                expr_np = expr

            is_constant = np.allclose(expr_np[0], expr_np, atol=1e-5)
            print(f"\nExpression variation analysis:")
            print(f"  Constant across frames: {is_constant}")

            if not is_constant:
                frame_diff = np.diff(expr_np, axis=0)
                diff_norm = np.linalg.norm(frame_diff, axis=-1)
                print(f"  Mean frame-to-frame difference: {diff_norm.mean():.6f}")
                print(f"  Max frame-to-frame difference: {diff_norm.max():.6f}")

        # Check for warping data
        if 'warp_field' in window_data:
            warp = window_data['warp_field']
            print(f"\nWarp field shape: {warp.shape}")

            # Visualize first frame's warp
            fig = visualize_warping_field(
                warp,
                title="Warping Field - Frame 0",
                save_path=output_dir / "warp_field_frame0.png"
            )
            plt.show()

    elif args.checkpoint:
        # Load from checkpoint
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

        # Look for expression embeddings in different possible locations
        expr_data = None
        if 'model_state_dict' in checkpoint:
            # Check if there are any saved expressions in the model
            for key in checkpoint['model_state_dict'].keys():
                if 'expression' in key.lower():
                    print(f"Found expression-related key: {key}")

        if 'validation_outputs' in checkpoint:
            if 'expression_embed' in checkpoint['validation_outputs']:
                expr_data = checkpoint['validation_outputs']['expression_embed']
                print(f"Found validation expressions: {expr_data.shape}")

        if expr_data is not None:
            fig = create_expression_candles(
                expr_data,
                title="Checkpoint Expression Embeddings",
                save_path=output_dir / "expression_candles_checkpoint.png"
            )
            plt.show()

    else:
        print("Please specify --checkpoint or --dataset")
        return

    print(f"\nVisualizations saved to {output_dir}")

if __name__ == "__main__":
    main()