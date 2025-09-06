import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import glob 
import re

def save_expression_embed(expression_embed, frame_idx, name, save_dir='expression_comparison'):
    """
    Save expression embedding tensor to disk for comparison.
    
    Args:
        expression_embed (torch.Tensor): Expression embedding tensor
        frame_idx (int): Frame index
        name (str): Identifier (e.g., 'vasa' or 'wrapper')
        save_dir (str): Directory to save files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure we're working with CPU tensor
    if isinstance(expression_embed, torch.Tensor):
        expression_embed = expression_embed.cpu().detach()
    
    # Save raw tensor
    torch.save(expression_embed, 
              os.path.join(save_dir, f'expression_{name}_frame_{frame_idx}.pt'))

def visualize_expression_comparison(frame_idx, save_dir='expression_comparison'):
    """
    Create visualization comparing VASA and InferenceWrapper expression embeddings.
    
    Args:
        frame_idx (int): Frame index to compare
        save_dir (str): Directory containing saved embeddings
    """
    # Load saved tensors
    vasa_path = os.path.join(save_dir, f'expression_vasa_frame_{frame_idx}.pt')
    wrapper_path = os.path.join(save_dir, f'expression_wrapper_target_frame_{frame_idx}.pt')
    
    if not (os.path.exists(vasa_path) and os.path.exists(wrapper_path)):
        print(f"Missing expression embeddings for frame {frame_idx}")
        return
        
    vasa_embed = torch.load(vasa_path)
    wrapper_embed = torch.load(wrapper_path)
    
    # Convert to numpy for visualization
    vasa_np = vasa_embed.numpy()
    wrapper_np = wrapper_embed.numpy()
    
    # Create figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))
    
    # Plot VASA embedding
    im1 = ax1.imshow(vasa_np.reshape(1, -1), aspect='auto', cmap='viridis')
    ax1.set_title(f'VASA Expression Embedding (Frame {frame_idx})')
    plt.colorbar(im1, ax=ax1)
    ax1.set_yticks([])
    
    # Plot Wrapper embedding
    im2 = ax2.imshow(wrapper_np.reshape(1, -1), aspect='auto', cmap='viridis')
    ax2.set_title(f'InferenceWrapper Expression Embedding (Frame {frame_idx})')
    plt.colorbar(im2, ax=ax2)
    ax2.set_yticks([])
    
    # Plot difference
    diff = vasa_np - wrapper_np
    im3 = ax3.imshow(diff.reshape(1, -1), aspect='auto', cmap='RdBu', norm=plt.Normalize(vmin=-np.abs(diff).max(), vmax=np.abs(diff).max()))
    ax3.set_title('Difference (VASA - Wrapper)')
    plt.colorbar(im3, ax=ax3)
    ax3.set_yticks([])
    
    # Add statistics as text
    stats_text = f'Max Diff: {np.abs(diff).max():.4f}\n'
    stats_text += f'Mean Diff: {np.abs(diff).mean():.4f}\n'
    stats_text += f'Correlation: {np.corrcoef(vasa_np.flatten(), wrapper_np.flatten())[0,1]:.4f}'
    fig.text(0.02, 0.02, stats_text, fontsize=10, va='bottom')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_frame_{frame_idx}.png'))
    plt.close()

def analyze_expressions(frame_idx, save_dir='expression_comparison'):
    """
    Analyze and print detailed statistics about the expression embeddings.
    
    Args:
        frame_idx (int): Frame index to analyze
        save_dir (str): Directory containing saved embeddings
    """
    # Load saved tensors
    vasa_path = os.path.join(save_dir, f'expression_vasa_frame_{frame_idx}.pt')
    wrapper_path = os.path.join(save_dir, f'expression_wrapper_target_frame_{frame_idx}.pt')
    
    vasa_embed = torch.load(vasa_path)
    wrapper_embed = torch.load(wrapper_path)
    
    # Convert to numpy
    vasa_np = vasa_embed.numpy()
    wrapper_np = wrapper_embed.numpy()
    
    # Compute statistics
    print(f"\nExpression Embedding Analysis for Frame {frame_idx}")
    print("-" * 50)
    print(f"VASA Embedding:")
    print(f"  Shape: {vasa_np.shape}")
    print(f"  Mean: {vasa_np.mean():.4f}")
    print(f"  Std: {vasa_np.std():.4f}")
    print(f"  Min: {vasa_np.min():.4f}")
    print(f"  Max: {vasa_np.max():.4f}")
    
    print(f"\nWrapper Embedding:")
    print(f"  Shape: {wrapper_np.shape}")
    print(f"  Mean: {wrapper_np.mean():.4f}")
    print(f"  Std: {wrapper_np.std():.4f}")
    print(f"  Min: {wrapper_np.min():.4f}")
    print(f"  Max: {wrapper_np.max():.4f}")
    
    diff = vasa_np - wrapper_np
    print(f"\nDifferences:")
    print(f"  Max Absolute Diff: {np.abs(diff).max():.4f}")
    print(f"  Mean Absolute Diff: {np.abs(diff).mean():.4f}")
    print(f"  Correlation: {np.corrcoef(vasa_np.flatten(), wrapper_np.flatten())[0,1]:.4f}")
    
    # Save detailed analysis
    with open(os.path.join(save_dir, f'analysis_frame_{frame_idx}.txt'), 'w') as f:
        f.write(f"Expression Embedding Analysis for Frame {frame_idx}\n")
        f.write("-" * 50 + "\n")
        f.write(f"VASA Embedding Statistics:\n")
        f.write(f"  Shape: {vasa_np.shape}\n")
        f.write(f"  Mean: {vasa_np.mean():.4f}\n")
        f.write(f"  Std: {vasa_np.std():.4f}\n")
        f.write(f"  Min: {vasa_np.min():.4f}\n")
        f.write(f"  Max: {vasa_np.max():.4f}\n\n")
        
        f.write(f"Wrapper Embedding Statistics:\n")
        f.write(f"  Shape: {wrapper_np.shape}\n")
        f.write(f"  Mean: {wrapper_np.mean():.4f}\n")
        f.write(f"  Std: {wrapper_np.std():.4f}\n")
        f.write(f"  Min: {wrapper_np.min():.4f}\n")
        f.write(f"  Max: {wrapper_np.max():.4f}\n\n")
        
        f.write(f"Differences:\n")
        f.write(f"  Max Absolute Diff: {np.abs(diff).max():.4f}\n")
        f.write(f"  Mean Absolute Diff: {np.abs(diff).mean():.4f}\n")
        f.write(f"  Correlation: {np.corrcoef(vasa_np.flatten(), wrapper_np.flatten())[0,1]:.4f}\n")

def analyze_single_file(filepath, frame_num, file_type, save_dir):
    """
    Analyze a single .pt file and create visualization.
    """
    # Load tensor
    tensor = torch.load(filepath)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().detach()
    
    # Convert to numpy
    np_array = tensor.numpy() if hasattr(tensor, 'numpy') else np.array(tensor)
    
    # Flatten if multi-dimensional
    if len(np_array.shape) > 1:
        np_flat = np_array.flatten()
    else:
        np_flat = np_array
    
    # Print statistics
    print(f"\n  {file_type} embedding (frame {frame_num}):")
    print(f"    Shape: {np_array.shape}")
    print(f"    Mean: {np_flat.mean():.4f}")
    print(f"    Std: {np_flat.std():.4f}")
    print(f"    Min: {np_flat.min():.4f}")
    print(f"    Max: {np_flat.max():.4f}")
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    im = ax.imshow(np_flat.reshape(1, -1), aspect='auto', cmap='viridis')
    ax.set_title(f'{file_type} Expression Embedding (Frame {frame_num})')
    ax.set_yticks([])
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{file_type}_frame_{frame_num}.png'))
    plt.close()

def visualize_file_comparison(frame_num, files, save_dir):
    """
    Compare multiple .pt files for the same frame.
    """
    # Load all available tensors
    tensors = {}
    for file_type, filepath in files.items():
        tensor = torch.load(filepath)
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().detach()
        tensors[file_type] = tensor.numpy() if hasattr(tensor, 'numpy') else np.array(tensor)
    
    # Create comparison visualization
    n_files = len(tensors)
    fig, axes = plt.subplots(n_files + 1, 1, figsize=(15, 3 * (n_files + 1)))
    
    if n_files == 1:
        axes = [axes]
    
    # Plot each tensor
    for idx, (file_type, np_array) in enumerate(tensors.items()):
        np_flat = np_array.flatten() if len(np_array.shape) > 1 else np_array
        
        im = axes[idx].imshow(np_flat.reshape(1, -1), aspect='auto', cmap='viridis')
        axes[idx].set_title(f'{file_type} Expression Embedding (Frame {frame_num})')
        axes[idx].set_yticks([])
        plt.colorbar(im, ax=axes[idx])
        
        # Print stats
        print(f"\n  {file_type}:")
        print(f"    Shape: {np_array.shape}")
        print(f"    Mean: {np_flat.mean():.4f}, Std: {np_flat.std():.4f}")
        print(f"    Range: [{np_flat.min():.4f}, {np_flat.max():.4f}]")
    
    # If we have exactly 2 files, plot their difference
    if n_files == 2:
        keys = list(tensors.keys())
        diff = tensors[keys[0]].flatten() - tensors[keys[1]].flatten()
        
        im = axes[-1].imshow(diff.reshape(1, -1), aspect='auto', cmap='RdBu', 
                            norm=plt.Normalize(vmin=-np.abs(diff).max(), vmax=np.abs(diff).max()))
        axes[-1].set_title(f'Difference ({keys[0]} - {keys[1]})')
        axes[-1].set_yticks([])
        plt.colorbar(im, ax=axes[-1])
        
        # Compute correlation if same size
        if tensors[keys[0]].shape == tensors[keys[1]].shape:
            corr = np.corrcoef(tensors[keys[0]].flatten(), tensors[keys[1]].flatten())[0,1]
            print(f"\n  Comparison statistics:")
            print(f"    Max Absolute Diff: {np.abs(diff).max():.4f}")
            print(f"    Mean Absolute Diff: {np.abs(diff).mean():.4f}")
            print(f"    Correlation: {corr:.4f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_frame_{frame_num}.png'))
    plt.close()

def analyze_all_expressions(save_dir='expression_comparison'):
    """
    Find all saved expression embeddings and analyze/visualize them.
    Handles any .pt files in the directory.
    """
    # Check if directory exists
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} not found!")
        return
    
    # Find all .pt files
    all_pt_files = glob.glob(os.path.join(save_dir, '*.pt'))
    
    if not all_pt_files:
        print(f"No .pt files found in {save_dir}")
        return
    
    # Group files by type and frame number
    file_groups = {}
    for file in all_pt_files:
        basename = os.path.basename(file)
        # Extract frame number from filename
        match = re.search(r'frame_(\d+)\.pt', basename)
        if match:
            frame_num = int(match.group(1))
            # Determine file type (vasa, wrapper, wrapper_target, etc.)
            if 'vasa' in basename:
                file_type = 'vasa'
            elif 'wrapper_target' in basename:
                file_type = 'wrapper_target'
            elif 'wrapper' in basename:
                file_type = 'wrapper'
            else:
                file_type = 'unknown'
            
            if frame_num not in file_groups:
                file_groups[frame_num] = {}
            file_groups[frame_num][file_type] = file
    
    print(f"Found {len(all_pt_files)} .pt files")
    print(f"Frame numbers: {sorted(file_groups.keys())}")
    
    # Analyze individual files or pairs
    for frame_num in sorted(file_groups.keys()):
        files = file_groups[frame_num]
        print(f"\n{'='*60}")
        print(f"Frame {frame_num}:")
        print(f"  Available files: {list(files.keys())}")
        
        # If we have pairs, compare them
        if len(files) >= 2:
            visualize_file_comparison(frame_num, files, save_dir)
        else:
            # Analyze single file
            for file_type, filepath in files.items():
                analyze_single_file(filepath, frame_num, file_type, save_dir)

def create_summary_report(save_dir='expression_comparison'):
    """
    Create a summary report of all analyzed expressions.
    """
    # Find all .pt files
    all_pt_files = glob.glob(os.path.join(save_dir, '*.pt'))
    
    if not all_pt_files:
        return
    
    # Collect statistics
    file_stats = {}
    for filepath in all_pt_files:
        basename = os.path.basename(filepath)
        tensor = torch.load(filepath)
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().detach().numpy()
        else:
            tensor = np.array(tensor)
        
        flat = tensor.flatten()
        file_stats[basename] = {
            'shape': tensor.shape,
            'mean': flat.mean(),
            'std': flat.std(),
            'min': flat.min(),
            'max': flat.max()
        }
    
    # Write summary report
    with open(os.path.join(save_dir, 'summary_report.txt'), 'w') as f:
        f.write("Expression Embedding Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total files analyzed: {len(all_pt_files)}\n\n")
        
        # Group by type
        types = {}
        for filename in file_stats.keys():
            if 'wrapper_target' in filename:
                type_name = 'wrapper_target'
            elif 'wrapper' in filename:
                type_name = 'wrapper'
            elif 'vasa' in filename:
                type_name = 'vasa'
            else:
                type_name = 'other'
            
            if type_name not in types:
                types[type_name] = []
            types[type_name].append(filename)
        
        # Report by type
        for type_name, files in types.items():
            f.write(f"\n{type_name.upper()} Files ({len(files)} total):\n")
            f.write("-" * 40 + "\n")
            
            # Calculate aggregate stats
            all_means = [file_stats[f]['mean'] for f in files]
            all_stds = [file_stats[f]['std'] for f in files]
            
            f.write(f"  Average mean: {np.mean(all_means):.4f}\n")
            f.write(f"  Average std: {np.mean(all_stds):.4f}\n")
            f.write(f"  Mean range: [{np.min(all_means):.4f}, {np.max(all_means):.4f}]\n")
            f.write(f"  Std range: [{np.min(all_stds):.4f}, {np.max(all_stds):.4f}]\n")
    
    print(f"\nSummary report saved to {os.path.join(save_dir, 'summary_report.txt')}")

# Run the analysis
if __name__ == "__main__":
    analyze_all_expressions()
    create_summary_report()

#     visualize_expression_comparison(0)
#     analyze_expressions(0) 
