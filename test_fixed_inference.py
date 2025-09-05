#!/usr/bin/env python
"""Test script for fixed inference with correct args loading."""

from infer import InferenceWrapper, create_driven_video

if __name__ == "__main__":
    # Initialize with correct args loading
    args_overwrite = {'l1_vol_rgb': 0}
    inferer = InferenceWrapper(
        experiment_name='Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1',
        model_file_name='328_model.pth',
        project_dir='/media/2TB/VASA-1-hack/nemo',
        folder='logs',
        state_dict=None,
        args_overwrite=args_overwrite,
        pose_momentum=0.1,
        print_model=False,
        print_params=True,
        debug=False  # Disable debug for cleaner output
    )
    
    # Generate video with correct colors
    create_driven_video(
        source_path='data/IMG_1.png',
        video_path='data/VID_1.mp4',
        output_path='data/result_fixed.mp4',
        inferer=inferer,
        max_frames=30,  # Process 30 frames for testing
        fps=30.0
    )
    
    print("\nVideo generation complete! Output saved to data/result_fixed.mp4")