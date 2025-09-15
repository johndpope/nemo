#!/usr/bin/env python
"""
Simple test to verify face_attrs H5 loading works correctly
"""

import h5py
import torch
import numpy as np
from pathlib import Path

def test_load_face_attrs():
    """Test loading face attributes from H5 cache."""

    h5_path = Path('../cache_single_bucket/all_windows_cache.h5')

    if not h5_path.exists():
        print(f"Cache file not found: {h5_path}")
        return

    print(f"Loading face attributes from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        print(f"Total windows: {f.attrs.get('total_windows', 0)}")
        print(f"Successful windows: {f.attrs.get('successful_windows', 0)}")

        # Get first available window
        windows = sorted([k for k in f.keys() if k.startswith('window_')],
                        key=lambda x: int(x.split('_')[1]))

        if not windows:
            print("No windows found in cache")
            return

        window_key = windows[0]
        print(f"\nLoading {window_key}...")
        window = f[window_key]

        # Check what's available
        print(f"Available keys: {list(window.keys())}")

        # Load key face attributes
        face_attrs = {}

        # Load gaze
        if 'gaze' in window:
            gaze_data = window['gaze'][()]
            face_attrs['gaze'] = torch.from_numpy(gaze_data)
            print(f"Gaze shape: {face_attrs['gaze'].shape}")

        # Load emotion
        if 'emotion' in window:
            emotion_data = window['emotion'][()]
            face_attrs['emotion'] = torch.from_numpy(emotion_data)
            print(f"Emotion shape: {face_attrs['emotion'].shape}")

        # Load head distance
        if 'head_distance' in window:
            head_dist_data = window['head_distance'][()]
            face_attrs['head_distance'] = torch.from_numpy(head_dist_data)
            print(f"Head distance shape: {face_attrs['head_distance'].shape}")

        # Load blink states
        if 'blink_state' in window:
            blink_data = window['blink_state'][()]
            face_attrs['blink_state'] = torch.from_numpy(blink_data)
            print(f"Blink state shape: {face_attrs['blink_state'].shape}")

        # Load identity frame
        if 'identity_frame' in window:
            identity_data = window['identity_frame'][()]
            face_attrs['identity_frame'] = torch.from_numpy(identity_data)
            print(f"Identity frame shape: {face_attrs['identity_frame'].shape}")

        # Load audio features
        if 'audio_features' in window:
            audio_data = window['audio_features'][()]
            face_attrs['audio_features'] = torch.from_numpy(audio_data)
            print(f"Audio features shape: {face_attrs['audio_features'].shape}")

        print(f"\nSuccessfully loaded face attributes for {window_key}")

        # Check if we can create synthetic frames
        if 'identity_frame' in face_attrs:
            # Expand identity frame to full sequence
            identity_frame = face_attrs['identity_frame']
            if identity_frame.dim() == 3:  # Single frame
                frames = identity_frame.unsqueeze(0).repeat(50, 1, 1, 1)
                print(f"Expanded identity frame to frames: {frames.shape}")

        return face_attrs

if __name__ == '__main__':
    test_load_face_attrs()