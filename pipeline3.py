#!/usr/bin/env python3
"""
Pipeline3: Deep trace version of pipeline2 with comprehensive debugging.
Breaks down the forward pass into helper methods with JSON logging and image saves.
"""

import argparse
import os
import cv2
import torch
import numpy as np
from torch import nn
from glob import glob
from PIL import Image
import json
from datetime import datetime
import traceback

from torchvision.transforms import transforms
from torch.nn import functional as F
from tqdm import trange, tqdm
from torchvision.transforms import ToTensor, ToPILImage

from networks.volumetric_avatar import FaceParsing
from repos.MODNet.src.models.modnet import MODNet
from ibug.face_detection import RetinaFacePredictor

to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
import os
import pathlib
import numpy as np
import importlib
import math
from scipy import linalg
import sys

import utils.args as args_utils
from utils import spectral_norm, stats_calc
from datasets.voxceleb2hq_pairs import LMDBDataset
from repos.MODNet.src.models.modnet import MODNet
from networks import volumetric_avatar
from torch.nn.modules.module import _addindent
import mediapipe as mp
from facenet_pytorch import MTCNN
import pickle
from utils import point_transforms
import contextlib
none_context = contextlib.nullcontext()
from typing import *
from PIL import Image

# Import DebugTracer from the new module
from debug_tracer import DebugTracer


from pathlib import Path

class InferenceWrapperDebug(nn.Module):
    """Debug version of InferenceWrapper with detailed tracing."""

    def __init__(self, experiment_name, which_epoch='latest', model_file_name='', use_gpu=True,
                 num_gpus=1, fixed_bounding_box=False, project_dir='./', folder='logs',
                 model_='va', torch_home='', debug=True, args_path=None):
        super(InferenceWrapperDebug, self).__init__()

        self.tracer = DebugTracer(output_dir="debug_pipeline3")
        self.use_gpu = use_gpu
        self.debug = debug
        self.num_gpus = num_gpus

        # Initialize model paths and config
        args_path = pathlib.Path(project_dir) / folder / experiment_name / 'args.txt' if args_path is None else args_path
        self.args = args_utils.parse_args(args_path)
        self.args.project_dir = project_dir

        if torch_home:
            os.environ['TORCH_HOME'] = torch_home

        # Set device
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Load model
        self._load_model(experiment_name, which_epoch, model_file_name, project_dir, folder, model_)

    def _load_model(self, experiment_name, which_epoch, model_file_name, project_dir, folder, model_):
        """Load the volumetric avatar model."""
        step = self.tracer.log_step("_load_model", "entry",
                                   experiment_name=experiment_name,
                                   which_epoch=which_epoch)

        # Model path
        if model_file_name:
            model_checkpoint = pathlib.Path(project_dir) / folder / experiment_name / 'checkpoints' / model_file_name
        else:
            model_checkpoint = pathlib.Path(project_dir) / folder / experiment_name / 'checkpoints' / f'{which_epoch}_model.pth'

        print(f"Loading model from: {model_checkpoint}")

        # Import and initialize model
        if model_ == 'va':
            model_name = 'volumetric_avatar'
        else:
            model_name = model_

        model = importlib.import_module(f'models.stage_1.{model_name}.{model_}')
        self.model = model.Model(self.args, training=False)

        # Load weights
        if model_checkpoint.exists():
            model_dict = torch.load(model_checkpoint, map_location='cpu')
            self.model.load_state_dict(model_dict, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()

        # MODNet for masking
        self.modnet = MODNet(backbone_pretrained=False)
        modnet_path = 'repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'
        if os.path.exists(modnet_path):
            self.modnet.load_state_dict(torch.load(modnet_path, map_location='cpu'), strict=False)
        self.modnet = self.modnet.to(self.device)
        self.modnet.eval()

        self.tracer.log_step("_load_model", "exit")

    def extract_identity_features(self, source_img: torch.Tensor) -> Dict:
        """Extract identity features with detailed logging."""
        step = self.tracer.log_step("extract_identity_features", "entry")

        self.tracer.save_image(source_img, "source_input", step)

        # Face mask extraction
        face_mask, _, _, cloth = self.model.face_idt.forward(source_img)
        self.tracer.log_step("face_idt", "exit",
                            face_mask=face_mask,
                            cloth_mask=cloth if cloth is not None else None)
        self.tracer.save_image(face_mask, "face_mask_raw", step)

        # Threshold mask
        face_mask = (face_mask > 0.6).float()
        self.tracer.save_image(face_mask, "face_mask_thresh", step)

        # Apply mask
        masked_source = source_img * face_mask
        self.tracer.save_image(masked_source, "masked_source", step)
        self.tracer.save_tensor(masked_source, "masked_source", step)

        # Identity embedding
        idt_embed = self.model.idt_embedder_nw(masked_source)
        self.tracer.log_step("idt_embedder_nw", "exit", idt_embed=idt_embed)
        self.tracer.save_tensor(idt_embed, "identity_embedding", step)

        # Head pose
        source_theta, scale, rotation, translation = self.model.head_pose_regressor.forward(source_img, True)
        self.tracer.log_step("head_pose_regressor", "exit",
                            theta=source_theta,
                            scale=scale,
                            rotation=rotation,
                            translation=translation)
        self.tracer.save_tensor(source_theta, "source_theta", step)

        # Create data dict
        data_dict = {
            'source_img': source_img,
            'source_mask': face_mask,
            'source_theta': source_theta,
            'target_img': source_img,
            'target_mask': face_mask,
            'target_theta': source_theta,
            'idt_embed': idt_embed
        }

        # Expression embedding
        data_dict = self.model.expression_embedder_nw(data_dict, True, False, False)
        self.tracer.log_step("expression_embedder_nw", "exit",
                            source_pose_embed=data_dict.get('source_pose_embed'))

        if 'source_pose_embed' in data_dict:
            self.tracer.save_tensor(data_dict['source_pose_embed'], "source_pose_embed", step)

        # Predict embeddings
        source_warp_embed, _, _, embed_dict = self.model.predict_embed(data_dict)
        self.tracer.log_step("predict_embed", "exit",
                            embed_dict_keys=list(embed_dict.keys()) if embed_dict else None)

        # Generate XY warps
        source_xy_warp, xy_conf = self.model.xy_generator_nw(source_warp_embed)
        self.tracer.log_step("xy_generator_nw", "exit",
                            xy_warp=source_xy_warp,
                            xy_conf=xy_conf if xy_conf is not None else None)
        self.tracer.save_tensor(source_xy_warp, "source_xy_warp", step)

        # Visualize XY warp
        if source_xy_warp is not None and source_xy_warp.shape[1] > 8:
            xy_slice = source_xy_warp[0, 8].detach().cpu().numpy()  # Middle slice
            xy_mag = np.sqrt(xy_slice[..., 0]**2 + xy_slice[..., 1]**2)
            self.tracer.save_image(torch.from_numpy(xy_mag).unsqueeze(0), "xy_warp_magnitude", step)

        # Local encoder
        source_latents = self.model.local_encoder_nw(masked_source)
        self.tracer.log_step("local_encoder_nw", "exit", source_latents=source_latents)
        self.tracer.save_tensor(source_latents, "source_latents", step)

        # Create source volume
        c = self.model.args.latent_volume_channels
        d = self.model.args.latent_volume_depth
        s = self.model.args.latent_volume_size

        source_volume = source_latents.view(1, c, d, s, s)
        self.tracer.log_step("reshape_volume", "exit",
                            source_volume=source_volume,
                            c=c, d=d, s=s)
        self.tracer.save_tensor(source_volume, "source_volume_raw", step)

        # Process source volume
        if self.model.args.source_volume_num_blocks > 0:
            source_volume = self.model.volume_source_nw(source_volume)
            self.tracer.log_step("volume_source_nw", "exit", source_volume=source_volume)
            self.tracer.save_tensor(source_volume, "source_volume_processed", step)

        # Create rotation warp
        grid = self.model.identity_grid_3d[:1]
        source_rot_warp = grid.bmm(source_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
        self.tracer.log_step("create_rotation_warp", "exit", rotation_warp=source_rot_warp)
        self.tracer.save_tensor(source_rot_warp, "source_rotation_warp", step)

        # Apply rotation
        rotated_source = self.model.grid_sample(source_volume, source_rot_warp)
        self.tracer.log_step("grid_sample_rotation", "exit", rotated_source=rotated_source)
        self.tracer.save_tensor(rotated_source, "rotated_source_volume", step)

        # Apply XY warp to get canonical
        canonical_volume = self.model.grid_sample(rotated_source, source_xy_warp)
        self.tracer.log_step("grid_sample_xy", "exit", canonical_volume=canonical_volume)
        self.tracer.save_tensor(canonical_volume, "canonical_volume", step)

        # Process canonical through volume network
        processed_canonical = self.model.volume_process_nw(canonical_volume, embed_dict)
        self.tracer.log_step("volume_process_nw", "exit", processed_canonical=processed_canonical)
        self.tracer.save_tensor(processed_canonical, "processed_canonical", step)

        result = {
            'idt_embed': idt_embed,
            'embed_dict': embed_dict,
            'canonical_volume': processed_canonical,
            'source_theta': source_theta,
            'source_mask': face_mask,
            'source_xy_warp': source_xy_warp,
            'source_latents': source_latents,
            'source_volume': source_volume
        }

        self.tracer.log_step("extract_identity_features", "exit", result_keys=list(result.keys()))

        return result

    def apply_target_expression(self, identity_info: Dict, target_img: torch.Tensor) -> torch.Tensor:
        """Apply target expression with detailed logging."""
        step = self.tracer.log_step("apply_target_expression", "entry")

        self.tracer.save_image(target_img, "target_input", step)

        # Target face mask
        target_mask, _, _, _ = self.model.face_idt.forward(target_img)
        self.tracer.log_step("face_idt_target", "exit", target_mask=target_mask)
        self.tracer.save_image(target_mask, "target_mask_raw", step)

        target_mask = (target_mask > 0.6).float()
        self.tracer.save_image(target_mask, "target_mask_thresh", step)

        # Target head pose
        target_theta, scale, rotation, translation = self.model.head_pose_regressor.forward(target_img, True)
        self.tracer.log_step("head_pose_target", "exit",
                            theta=target_theta,
                            scale=scale,
                            rotation=rotation,
                            translation=translation)
        self.tracer.save_tensor(target_theta, "target_theta", step)

        # Create target data dict WITH SOURCE IDENTITY
        data_dict = {
            'source_img': target_img,
            'source_mask': target_mask,
            'source_theta': target_theta,
            'target_img': target_img,
            'target_mask': target_mask,
            'target_theta': target_theta,
            'idt_embed': identity_info['idt_embed']  # SOURCE IDENTITY!
        }

        self.tracer.log_step("create_target_dict", "exit",
                            using_source_identity=True,
                            idt_embed_shape=list(identity_info['idt_embed'].shape))

        # Target expression
        data_dict = self.model.expression_embedder_nw(data_dict, True, False, False)
        self.tracer.log_step("expression_embedder_target", "exit",
                            target_pose_embed=data_dict.get('source_pose_embed'))

        if 'source_pose_embed' in data_dict:
            self.tracer.save_tensor(data_dict['source_pose_embed'], "target_pose_embed", step)

        # Target warps
        _, target_warp_embed, _, _ = self.model.predict_embed(data_dict)
        self.tracer.log_step("predict_embed_target", "exit")

        # UV warps
        target_uv_warp, uv_conf = self.model.uv_generator_nw(target_warp_embed)
        self.tracer.log_step("uv_generator_nw", "exit",
                            uv_warp=target_uv_warp,
                            uv_conf=uv_conf if uv_conf is not None else None)
        self.tracer.save_tensor(target_uv_warp, "target_uv_warp", step)

        # Visualize UV warp
        if target_uv_warp is not None and target_uv_warp.shape[1] > 8:
            uv_slice = target_uv_warp[0, 8].detach().cpu().numpy()
            uv_mag = np.sqrt(uv_slice[..., 0]**2 + uv_slice[..., 1]**2)
            self.tracer.save_image(torch.from_numpy(uv_mag).unsqueeze(0), "uv_warp_magnitude", step)

        # Apply UV warp to canonical
        volume_with_expression = self.model.grid_sample(identity_info['canonical_volume'], target_uv_warp)
        self.tracer.log_step("grid_sample_uv", "exit", volume_with_expression=volume_with_expression)
        self.tracer.save_tensor(volume_with_expression, "volume_with_expression", step)

        # Target rotation
        c = self.model.args.latent_volume_channels
        d = self.model.args.latent_volume_depth
        s = self.model.args.latent_volume_size

        grid = self.model.identity_grid_3d[:1]
        target_rot_warp = grid.bmm(target_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
        self.tracer.log_step("create_target_rotation", "exit", rotation_warp=target_rot_warp)
        self.tracer.save_tensor(target_rot_warp, "target_rotation_warp", step)

        # Apply final rotation
        final_volume = self.model.grid_sample(volume_with_expression, target_rot_warp)
        self.tracer.log_step("grid_sample_final_rotation", "exit", final_volume=final_volume)
        self.tracer.save_tensor(final_volume, "final_volume", step)

        # Prepare for decoder
        target_latent_feats = final_volume.view(1, c * d, s, s)
        self.tracer.log_step("reshape_for_decoder", "exit",
                            target_latent_feats=target_latent_feats)
        self.tracer.save_tensor(target_latent_feats, "target_latent_feats", step)

        # Decoder dict
        decode_dict = {
            'target_theta': target_theta,
            'target_pose_embed': data_dict.get('source_pose_embed')
        }

        # DECODE WITH SOURCE IDENTITY
        self.tracer.log_step("decoder_nw", "entry",
                            using_source_embed_dict=True,
                            embed_dict_keys=list(identity_info['embed_dict'].keys()))

        generated_img, _, deep_features, img_features = self.model.decoder_nw(
            decode_dict,
            identity_info['embed_dict'],  # SOURCE IDENTITY!
            target_latent_feats,
            False,
            stage_two=True
        )

        self.tracer.log_step("decoder_nw", "exit",
                            generated_img=generated_img,
                            generated_range=[float(generated_img.min().item()),
                                           float(generated_img.max().item())])
        self.tracer.save_image(generated_img, "generated_raw", step)
        self.tracer.save_tensor(generated_img, "generated_raw", step)

        # Check and fix range
        gen_min = generated_img.min().item()
        gen_max = generated_img.max().item()

        if gen_min >= 0 and gen_max <= 1.1:
            self.tracer.log_step("range_conversion", "detected",
                               from_range="[0,1]", to_range="[-1,1]")
            generated_img = generated_img * 2 - 1
            self.tracer.save_image(generated_img, "generated_converted", step)

        # Get mask for compositing
        gen_mask, _, _, _ = self.model.face_idt.forward(generated_img)
        self.tracer.save_image(gen_mask, "gen_mask_raw", step)

        gen_mask = (gen_mask > 0.65).float()
        self.tracer.save_image(gen_mask, "gen_mask_thresh", step)

        # Smooth mask
        for i in range(3):
            gen_mask = F.avg_pool2d(gen_mask, 3, stride=1, padding=1)
        self.tracer.save_image(gen_mask, "gen_mask_smooth", step)

        # Composite
        background = target_img * (1 - gen_mask)
        self.tracer.save_image(background, "background", step)

        foreground = generated_img * gen_mask
        self.tracer.save_image(foreground, "foreground", step)

        final_img = foreground + background
        final_img = torch.clamp(final_img, -1, 1)

        self.tracer.log_step("compositing", "exit",
                            final_range=[float(final_img.min().item()),
                                       float(final_img.max().item())])
        self.tracer.save_image(final_img, "final_composite", step)
        self.tracer.save_tensor(final_img, "final_composite", step)

        self.tracer.log_step("apply_target_expression", "exit")

        return final_img, generated_img

    def forward(self, source_image=None, driver_image=None, **kwargs):
        """Main forward pass with comprehensive debugging."""

        try:
            self.tracer.log_step("forward", "entry",
                               has_source=source_image is not None,
                               has_driver=driver_image is not None)

            # Convert images to tensors
            if source_image is not None:
                if isinstance(source_image, Image.Image):
                    source_image = np.array(source_image)
                if isinstance(source_image, np.ndarray):
                    source_image = torch.from_numpy(source_image).float() / 127.5 - 1.0
                    if len(source_image.shape) == 3:
                        source_image = source_image.permute(2, 0, 1).unsqueeze(0)
                    source_image = source_image.to(self.device)

            if driver_image is not None:
                if isinstance(driver_image, Image.Image):
                    driver_image = np.array(driver_image)
                if isinstance(driver_image, np.ndarray):
                    driver_image = torch.from_numpy(driver_image).float() / 127.5 - 1.0
                    if len(driver_image.shape) == 3:
                        driver_image = driver_image.permute(2, 0, 1).unsqueeze(0)
                    driver_image = driver_image.to(self.device)

            result = None

            # Process source
            if source_image is not None:
                self.tracer.log_step("process_source", "entry")
                self.source_features = self.extract_identity_features(source_image)
                self.tracer.log_step("process_source", "exit")

            # Process driver/target
            if driver_image is not None and hasattr(self, 'source_features'):
                self.tracer.log_step("process_driver", "entry")
                result, raw = self.apply_target_expression(self.source_features, driver_image)
                self.tracer.log_step("process_driver", "exit")

                # Convert result to PIL for compatibility
                if result is not None:
                    result_np = result[0].detach().cpu().permute(1, 2, 0).numpy()
                    result_np = ((result_np + 1) * 127.5).astype(np.uint8)
                    result_pil = Image.fromarray(result_np)

                    self.tracer.log_step("forward", "exit", success=True)
                    return [result_pil], result

            self.tracer.log_step("forward", "exit", success=False)
            return None, None

        except Exception as e:
            self.tracer.log_step("forward", "error",
                               error=str(e),
                               traceback=traceback.format_exc())
            raise

        finally:
            # Save complete trace
            self.tracer.save_final_trace()


def get_video_frames(video_path, max_frames=10):
    """Extract frames from video."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to 512x512
        frame_rgb = cv2.resize(frame_rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        # Convert to PIL Image
        frames.append(Image.fromarray(frame_rgb))
        count += 1

    cap.release()
    return frames


def create_comparison_grid(source_img, target_imgs, result_imgs, save_path="debug_pipeline3/comparison.png"):
    """Create a comparison grid showing source, targets, and results."""
    import matplotlib.pyplot as plt

    n_targets = min(len(target_imgs), len(result_imgs))
    if n_targets == 0:
        return

    fig, axes = plt.subplots(3, n_targets + 1, figsize=(3 * (n_targets + 1), 9))

    # If only one target, reshape axes
    if n_targets == 1:
        axes = axes.reshape(3, -1)

    # Show source
    axes[0, 0].imshow(source_img)
    axes[0, 0].set_title("Source\n(IMG_1)", fontsize=10, weight='bold')
    axes[0, 0].axis('off')

    axes[1, 0].text(0.5, 0.5, 'Identity\nSource', ha='center', va='center',
                   fontsize=11, weight='bold', color='blue')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')

    # Show targets and results
    for i in range(n_targets):
        col = i + 1

        # Target
        axes[0, col].imshow(target_imgs[i])
        axes[0, col].set_title(f"Target {i+1}", fontsize=9)
        axes[0, col].axis('off')

        # Result
        axes[1, col].imshow(result_imgs[i])
        axes[1, col].set_title(f"Result {i+1}", fontsize=9, weight='bold', color='green')
        axes[1, col].axis('off')

        # Difference
        target_np = np.array(target_imgs[i]).astype(np.float32) / 255.0
        result_np = np.array(result_imgs[i]).astype(np.float32) / 255.0
        diff = np.abs(result_np - target_np).mean(axis=2)

        axes[2, col].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2, col].set_title("Difference", fontsize=9)
        axes[2, col].axis('off')

    plt.suptitle("Pipeline3 Debug Face Swap Results", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison grid: {save_path}")


def test_debug_pipeline(source_path=None, target_path=None, use_video=False, max_frames=5, swap_identity=False):
    """Test the debug pipeline with video support.

    Args:
        source_path: Path to source image (default: ./data/IMG_1.png)
        target_path: Path to target image or video
        use_video: If True, process target as video
        max_frames: Maximum number of video frames to process
        swap_identity: If True, use driver as identity source (for face swap)
    """
    import argparse

    print("="*60)
    print("Testing Debug Pipeline3")
    print("="*60)

    # Initialize wrapper
    wrapper = InferenceWrapperDebug(
        experiment_name='Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1',
        which_epoch='328',
        model_file_name='328_model.pth',
        project_dir='./',
        folder='logs',
        model_='va'
    )

    # Load source image (default to IMG_1.png like pipeline2.py)
    if source_path is None:
        source_path = "nemo/data/IMG_1.png"

    print(f"\nLoading source: {source_path}")
    # FIXED: Use LANCZOS resize to match pipeline2.py
    source = Image.open(source_path).convert('RGB').resize((512, 512), Image.LANCZOS)

    # Process targets
    results = []
    result_imgs = []
    target_imgs = []

    if use_video and target_path:
        # Process video frames
        print(f"\nProcessing video: {target_path}")
        frames = get_video_frames(target_path, max_frames)

        if len(frames) == 0:
            print("Error: No frames extracted from video!")
            return

        if swap_identity:
            # Face swap mode: Use driver as identity source, source as expression
            print("\n[SWAP MODE] Using driver frame for identity, source for expression...")
            print("Extracting identity from first driver frame...")
            first_result, first_raw = wrapper.forward(source_image=frames[0], driver_image=source)
        else:
            # Standard mode: Use source for identity, driver for expression
            print("\nExtracting source identity with first driver frame (joint extraction)...")
            first_result, first_raw = wrapper.forward(source_image=source, driver_image=frames[0])

        if first_result:
            results.append(first_result)
            result_imgs.append(first_result[0])
            target_imgs.append(frames[0])
            first_result[0].save(f"debug_pipeline3/frame_000_result.png")
            print(f"  Saved frame_000_result.png")

        # Process remaining frames with cached embeddings
        for i in range(1, len(frames)):
            print(f"\nProcessing frame {i+1}/{len(frames)}...")
            if swap_identity:
                # Continue using source as expression driver
                frame_result, frame_raw = wrapper.forward(source_image=None, driver_image=source)
            else:
                # Continue using frames as expression driver
                frame_result, frame_raw = wrapper.forward(source_image=None, driver_image=frames[i])

            if frame_result:
                results.append(frame_result)
                result_imgs.append(frame_result[0])
                target_imgs.append(frames[i])
                frame_result[0].save(f"debug_pipeline3/frame_{i:03d}_result.png")
                print(f"  Saved frame_{i:03d}_result.png")

        # Create comparison grid for video
        if result_imgs:
            create_comparison_grid(source, target_imgs, result_imgs,
                                 "debug_pipeline3/video_comparison.png")
    else:
        # Process single image
        if target_path is None:
            target_path = "nemo/data/IMG_2.png"

        print(f"\nLoading target: {target_path}")
        # FIXED: Use LANCZOS resize for target too
        target = Image.open(target_path).convert('RGB').resize((512, 512), Image.LANCZOS)

        # Joint extraction for single image
        if swap_identity:
            print("\n[SWAP MODE] Using target for identity, source for expression...")
            result, raw = wrapper.forward(source_image=target, driver_image=source)
        else:
            print("\nProcessing with joint extraction (source + target)...")
            result, raw = wrapper.forward(source_image=source, driver_image=target)

        if result:
            result[0].save("debug_pipeline3/result.png")
            print(f"\nResult saved to debug_pipeline3/result.png")

            # Create comparison grid for single image
            create_comparison_grid(source, [target], [result[0]],
                                 "debug_pipeline3/comparison.png")

    print(f"\nDebug output saved to: {wrapper.tracer.output_dir}")
    print(f"Total steps traced: {wrapper.tracer.step_counter}")
    print(f"Processed {len(results) if use_video else 1} target(s)")

    print("\n" + "="*60)
    print("Debug Pipeline3 Complete!")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pipeline3 Debug Tool (nemo version)')
    parser.add_argument('--source', default='nemo/data/IMG_1.png',
                       help='Source image path (default: nemo/data/IMG_1.png)')
    parser.add_argument('--target', default=None,
                       help='Target image or video path')
    parser.add_argument('--video', action='store_true',
                       help='Process target as video (extracts frames)')
    parser.add_argument('--max-frames', type=int, default=5,
                       help='Maximum number of video frames to process (default: 5)')
    parser.add_argument('--default-video', action='store_true',
                       help='Use default video from pipeline2.py')
    parser.add_argument('--swap-identity', action='store_true',
                       help='Swap identity source (use driver for identity, source for expression)')
    args = parser.parse_args()

    # Handle default video option
    target_path = args.target
    use_video = args.video

    if args.default_video:
        # Use relative path to junk folder
        target_path = 'junk/15.mp4'
        use_video = True
        print(f"Using default video from pipeline2.py: {target_path}")
    elif target_path and target_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Auto-detect video files by extension
        use_video = True
        print(f"Detected video file: {target_path}")

    # Run the test
    test_debug_pipeline(
        source_path=args.source,
        target_path=target_path,
        use_video=use_video,
        max_frames=args.max_frames,
        swap_identity=args.swap_identity
    )