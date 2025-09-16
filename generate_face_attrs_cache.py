#!/usr/bin/env python3
"""
Generate face attributes cache from a video file for use with pipeline_face_attr.py
This extracts expressions and head poses from driving videos and saves them to H5 files.
"""

import os
import h5py
import torch
import numpy as np
from PIL import Image
import cv2
import argparse
from tqdm import tqdm
import hashlib

from infer import InferenceWrapper
from decord import VideoReader

def extract_face_attributes_from_video(video_path, inferer, max_frames=None):
    """Extract face attributes from a video using the model's embedders"""

    # Load video
    vr = VideoReader(video_path)
    num_frames = len(vr) if max_frames is None else min(len(vr), max_frames)

    print(f"Processing {num_frames} frames from video: {video_path}")

    all_expressions = []
    all_thetas = []
    all_head_poses = []

    for idx in tqdm(range(num_frames), desc="Extracting face attributes"):
        frame = vr[idx].asnumpy()
        # Convert BGR to RGB if needed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Resize to 512x512
        frame_pil = frame_pil.resize((512, 512), Image.LANCZOS)

        # Convert to tensor
        frame_tensor = torch.from_numpy(np.array(frame_pil)).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

        with torch.no_grad():
            # Create data dict for expression embedder
            data_dict = {
                'source_img': frame_tensor,
                'target_img': frame_tensor  # Use same frame as both source and target
            }

            # Extract expression embedding
            expression_embed = inferer.model.expression_embedder_nw(data_dict)
            # Take only the target expression (second half)
            t = expression_embed.shape[0] // 2
            target_expr = expression_embed[t:] if expression_embed.shape[0] > 1 else expression_embed
            all_expressions.append(target_expr.cpu().numpy())

            # Extract head pose (theta) - simplified version
            # In real implementation, this would use the pose detector
            # For now, we'll use a placeholder
            theta = torch.zeros(1, 6).cpu().numpy()  # 6D pose representation
            all_thetas.append(theta)

            # Head pose embedding (if available)
            head_pose = torch.zeros(1, 512).cpu().numpy()  # Placeholder
            all_head_poses.append(head_pose)

    return {
        'expressions': np.concatenate(all_expressions, axis=0),
        'thetas': np.concatenate(all_thetas, axis=0),
        'head_poses': np.concatenate(all_head_poses, axis=0),
        'num_frames': num_frames
    }


def save_face_attrs_to_h5(face_attrs, output_path, window_size=4):
    """Save face attributes to H5 file with window organization"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Save metadata
        f.attrs['num_frames'] = face_attrs['num_frames']
        f.attrs['window_size'] = window_size

        # Calculate number of windows
        num_windows = face_attrs['num_frames'] - window_size + 1
        f.attrs['num_windows'] = num_windows

        # Save attributes by window
        for window_idx in range(num_windows):
            window_group = f.create_group(f'window_{window_idx}')

            # Get window slice
            start_idx = window_idx
            end_idx = window_idx + window_size

            # Save expressions for this window
            window_group.create_dataset('expressions',
                                       data=face_attrs['expressions'][start_idx:end_idx])

            # Save thetas for this window
            window_group.create_dataset('thetas',
                                       data=face_attrs['thetas'][start_idx:end_idx])

            # Save head poses for this window
            window_group.create_dataset('head_poses',
                                       data=face_attrs['head_poses'][start_idx:end_idx])

            # Save frame indices
            window_group.create_dataset('frame_indices',
                                       data=np.arange(start_idx, end_idx))

    print(f"Saved face attributes to: {output_path}")
    print(f"  - Total frames: {face_attrs['num_frames']}")
    print(f"  - Number of windows: {num_windows}")
    print(f"  - Window size: {window_size}")


def main():
    parser = argparse.ArgumentParser(description='Generate face attributes cache from video')
    parser.add_argument('--video_path', type=str, default='../junk/15.mp4',
                       help='Path to input video')
    parser.add_argument('--output_dir', type=str, default='../cache_single_bucket',
                       help='Output directory for H5 files')
    parser.add_argument('--max_frames', type=int, default=100,
                       help='Maximum number of frames to process')
    parser.add_argument('--window_size', type=int, default=4,
                       help='Window size for temporal context')

    args = parser.parse_args()

    # Generate output filename based on video hash
    video_hash = hashlib.md5(open(args.video_path, 'rb').read()).hexdigest()
    output_path = os.path.join(args.output_dir, f'face_attrs_{video_hash}.h5')

    # Initialize model
    project_dir = os.path.dirname(os.path.abspath(__file__))
    args_overwrite = {'l1_vol_rgb': 0}

    print("Loading model...")
    inferer = InferenceWrapper(
        experiment_name='Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1',
        model_file_name='328_model.pth',
        project_dir=project_dir,
        folder='logs',
        state_dict=None,
        args_overwrite=args_overwrite,
        pose_momentum=0.1,
        print_model=False,
        print_params=True
    )

    # Extract face attributes
    print("Extracting face attributes from video...")
    face_attrs = extract_face_attributes_from_video(args.video_path, inferer, args.max_frames)

    # Save to H5
    save_face_attrs_to_h5(face_attrs, output_path, args.window_size)

    print(f"\nDone! You can now use the cached attributes with:")
    print(f"  python pipeline_face_attr.py --face_attrs_h5 {output_path}")


if __name__ == '__main__':
    main()