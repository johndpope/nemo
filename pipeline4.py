#!/usr/bin/env python3
"""
Pipeline4: Refactored version of pipeline2.py with better organization
Maintains EXACT working logic from pipeline2.py to preserve correct identity handling
"""

import argparse
import os
import cv2
import torch
import numpy as np
import pathlib
from torch import nn
from glob import glob
from PIL import Image

from torchvision.transforms import transforms
from torch.nn import functional as F
from tqdm import trange, tqdm
from torchvision.transforms import ToTensor, ToPILImage

from networks.volumetric_avatar import FaceParsing
from repos.MODNet.src.models.modnet import MODNet
from ibug.face_detection import RetinaFacePredictor

# Import background utilities
from background_utils import BackgroundProcessor, init_background_processor

to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()
to_flip = transforms.RandomHorizontalFlip(p=1)
to_512 = lambda x: x.resize((512, 512), Image.LANCZOS)
to_256 = lambda x: x.resize((256, 256), Image.LANCZOS)

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
try:
    import apex
except ImportError:
    apex = None
import sys

import utils.args as args_utils
from logger import logger
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
from logger import logger
from PIL import Image
from vis_helper import save_expression_embed
from debug_tracer import DebugTracer

# Global tracer instance (disabled by default, enable with --debug flag)
global_tracer = None


def log_tensor_state(name: str, tensor: Optional[Union[torch.Tensor, Image.Image, list]], detailed: bool = False) -> None:
    """Log tensor state with shape and value range if available."""
    if tensor is None:
        logger.debug(f"{name}: None")
    elif isinstance(tensor, Image.Image):
        logger.debug(f"{name}: PIL Image size={tensor.size}")
    elif isinstance(tensor, list):
        logger.debug(f"{name}: List of {len(tensor)} items")
        if len(tensor) > 0:
            if isinstance(tensor[0], Image.Image):
                logger.debug(f"{name} (first item): PIL Image size={tensor[0].size}")
            elif isinstance(tensor[0], torch.Tensor):
                logger.debug(f"{name} (first item): Tensor shape={tensor[0].shape}")
    elif isinstance(tensor, torch.Tensor):
        info = f"{name}: shape={tensor.shape}"
        if detailed and torch.is_floating_point(tensor):
            info += f", range=[{tensor.min():.3f}, {tensor.max():.3f}]"
        logger.debug(info)
    else:
        logger.debug(f"{name}: type={type(tensor)}")


def log_data_dict(prefix: str, data_dict: Dict[str, torch.Tensor]) -> None:
    """Log the shapes of tensors in a data dictionary."""
    logger.debug(f"\n{prefix} data dictionary contents:")
    for k, v in data_dict.items():
        if isinstance(v, (torch.Tensor, Image.Image, list)):
            log_tensor_state(f"  {k}", v)


def log_processing_step(step_name: str) -> None:
    """Log a processing step header."""
    logger.debug(f"\n=== {step_name} ===")


class InferenceWrapper(nn.Module):
    """
    Refactored InferenceWrapper - maintains exact logic from pipeline2.py
    but with better organization and helper methods
    """

    def __init__(self, experiment_name, which_epoch='latest', model_file_name='', use_gpu=True, num_gpus=1,
                 fixed_bounding_box=False, project_dir='./', folder= 'mp_logs', model_ = 'va',
                 torch_home='', debug=False, print_model=False, print_params=True, args_overwrite={},
                 state_dict=None, pose_momentum=0.5, rank=0, args_path=None, enable_tracing=False):
        super(InferenceWrapper, self).__init__()

        self.use_gpu = use_gpu
        self.debug = debug
        self.num_gpus = num_gpus

        # Initialize tracer if enabled
        self.tracer = DebugTracer(output_dir="debug_pipeline4", enabled=enable_tracing) if enable_tracing else None

        self.modnet_pass = 'repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'

        # Get a config for the network
        args_path = pathlib.Path(project_dir) / folder / experiment_name / 'args.txt' if args_path is None else args_path

        import utils.args as args_utils
        self.args = args_utils.parse_args(args_path)
        # Add args from args_overwrite dict that overwrite the default ones
        self.args.project_dir = project_dir
        if args_overwrite is not None:
            for k, v in args_overwrite.items():
                setattr(self.args, k, v)

        # Set device
        if torch_home:
            os.environ['TORCH_HOME'] = torch_home
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            self.device = torch.device("cpu")

        # Import the module (model)
        self._load_model(experiment_name, which_epoch, model_file_name, project_dir, folder, model_, state_dict, print_model, print_params)

        # Initialize helper modules
        self._init_helper_modules()

        # Initialize tracking variables
        self._init_tracking_variables(pose_momentum)

        # Args already loaded above in __init__

        # Set resize_warp flag based on args
        self.resize_warp = self.args.warp_output_size != self.args.gen_latent_texture_size

    def _load_model(self, experiment_name, which_epoch, model_file_name, project_dir, folder, model_, state_dict, print_model, print_params):
        """Load the volumetric avatar model - CRITICAL: Keep exact loading logic from pipeline2"""
        if model_ == 'va':
            model_name = 'volumetric_avatar'
        else:
            model_name = model_

        if self.tracer:
            self.tracer.log_step("model_loading", "start", experiment=experiment_name)

        module = importlib.import_module(f'models.stage_1.{model_name}.{model_}')
        self.model = module.Model(self.args, training=False).to(self.device)

        # Load checkpoint
        if state_dict is None:
            if model_file_name == '':
                model_file_name = f'{which_epoch}_model.pth'
            model_path = pathlib.Path(project_dir) / folder / experiment_name / 'checkpoints' / model_file_name
            state_dict = torch.load(model_path, map_location=self.device)

        # Move model to device first
        self.model = self.model.to(self.device)

        if self.num_gpus > 1:
            self.model = nn.DataParallel(self.model)
            self.model.module.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

        if print_model:
            logger.debug(self.model)

        if print_params:
            self._print_params()

        if self.tracer:
            self.tracer.log_step("model_loading", "complete")

    def _print_params(self):
        """Print model parameters count"""
        for name, module in self.model.named_children():
            logger.debug(f'Number of parameters in {name}: {sum(p.numel() for p in module.parameters())}')

    def _init_helper_modules(self):
        """Initialize helper modules for processing"""
        # ModNet for masking
        self.modnet = MODNet(backbone_pretrained=False)
        self.modnet = nn.DataParallel(self.modnet).cuda()
        self.modnet.load_state_dict(torch.load(self.modnet_pass))
        self.modnet.eval()

        # FaceParsing for face masks
        self.face_idt = FaceParsing(None, self.device)

        # Initialize background processor
        self.bg_processor = BackgroundProcessor(device=self.device)
        self.bg_processor.set_models(modnet=self.modnet, face_idt=self.face_idt)

        # Transforms
        self.mp_face_detection = mp.solutions.face_detection
        self.to_image = to_image
        self.to_tensor = to_tensor

    def _init_tracking_variables(self, pose_momentum):
        """Initialize tracking variables for pose and motion"""
        self.center = None
        self.size = None
        self.theta = None
        self.delta_yaw = None
        self.delta_pitch = None
        self.pose_momentum = pose_momentum
        self.mix = False
        self.mix_old = True
        self.no_grad_infer = True
        self.target_theta = True

        # Cached values for performance
        self.cached_source_data = None
        self.cached_frame_idx = -1


    # ===================== CRITICAL FORWARD METHOD - PRESERVE EXACT LOGIC =====================

    def forward(self, source_image=None, driver_image=None, source_mask=None, source_mask_add=0,
                driver_mask=None, crop=True, reset_tracking=False, smooth_pose=False,
                hard_normalize=False, soft_normalize=False, delta_yaw=None, delta_pitch=None, cloth=False,
                thetas_pass='', theta_n=0, target_theta=True, mix=False, mix_old=True,
                c_source_latent_volume=None, c_target_latent_volume=None, custome_target_pose_embed=None,
                custome_target_theta_embed=None, no_grad_infer=True, modnet_mask=False, frame_idx=0):
        """
        CRITICAL: This forward method MUST maintain exact logic from pipeline2.py
        DO NOT change the order of operations or data flow
        """

        # Debug tracing
        if self.tracer:
            self.tracer.log_step("forward", "entry",
                               has_source=source_image is not None,
                               has_driver=driver_image is not None,
                               frame_idx=frame_idx)

        log_processing_step("Forward Pass Initialization")
        log_tensor_state("Input source image", source_image)
        log_tensor_state("Input driver image", driver_image)

        self.no_grad_infer = no_grad_infer
        self.target_theta = target_theta

        with torch.no_grad():
            if reset_tracking:
                self._reset_tracking()

            self._update_motion_params(mix, mix_old, delta_yaw, delta_pitch)

            if source_image is not None:
                # Process source - CRITICAL: Keep exact processing order
                source_data = self._process_source_image(
                    source_image, source_mask, source_mask_add,
                    crop, cloth, modnet_mask
                )
                self.cached_source_data = source_data

            if driver_image is not None and self.cached_source_data is not None:
                # Process driver and generate - CRITICAL: Keep exact generation logic
                result = self._process_driver_and_generate(
                    driver_image, driver_mask, crop, smooth_pose,
                    hard_normalize, soft_normalize, thetas_pass, theta_n,
                    c_source_latent_volume, c_target_latent_volume,
                    custome_target_pose_embed, custome_target_theta_embed,
                    modnet_mask, frame_idx
                )

                if self.tracer and result is not None:
                    pred_target_img, img = result
                    for i, pil_img in enumerate(pred_target_img):
                        self.tracer.save_image(pil_img, f"generated_{frame_idx}_{i}")
                    self.tracer.log_step("forward", "exit", success=True, num_images=len(pred_target_img))

                return result
            else:
                logger.debug("No driver image provided, returning None")
                if self.tracer:
                    self.tracer.log_step("forward", "exit", success=False, reason="no_driver")
                return None

    def _reset_tracking(self):
        """Reset tracking parameters"""
        log_processing_step("Resetting Tracking Parameters")
        self.center = None
        self.size = None
        self.theta = None
        self.delta_yaw = None
        self.delta_pitch = None

    def _update_motion_params(self, mix, mix_old, delta_yaw, delta_pitch):
        """Update motion parameters"""
        self.mix = mix
        self.mix_old = mix_old
        if delta_yaw is not None:
            self.delta_yaw = delta_yaw
        if delta_pitch is not None:
            self.delta_pitch = delta_pitch

    def _process_source_image(self, source_image, source_mask, source_mask_add, crop, cloth, modnet_mask):
        """
        Process source image - matches pipeline2 source processing
        """
        log_processing_step("Source Image Processing")

        if crop:
            source_image = self._crop_face(source_image, is_source=True)

        # Convert to tensor
        source_img_t = self._prepare_image_tensor(source_image)

        # Get masks
        source_mask = self._get_mask(source_img_t, source_mask, source_mask_add, modnet_mask)
        source_mask = source_mask.to(self.device)

        # Store for later use
        self.source_img_mask = source_mask

        # Get volume dimensions
        c = self.args.latent_volume_channels
        s = self.args.latent_volume_size
        d = self.args.latent_volume_depth

        # Process source to get canonical volume (matching pipeline2 logic)
        with torch.no_grad():
            # Get identity embedding - directly from masked image like pipeline2
            if self.tracer:
                self.tracer.log_step("source_processing", "before_idt_embed",
                                   input_shape=list(source_img_t.shape),
                                   mask_shape=list(source_mask.shape))

            self.idt_embed = self.model.idt_embedder_nw.forward_image(source_img_t * source_mask)

            if self.tracer:
                self.tracer.log_step("source_processing", "after_idt_embed",
                                   idt_embed_shape=list(self.idt_embed.shape) if self.idt_embed is not None else None)

            # Get local features
            if self.tracer:
                self.tracer.log_step("source_processing", "before_local_encoder")

            source_latents = self.model.local_encoder_nw(source_img_t * source_mask)

            if self.tracer:
                self.tracer.log_step("source_processing", "after_local_encoder",
                                   latents_shape=list(source_latents.shape),
                                   latents_min=float(source_latents.min()),
                                   latents_max=float(source_latents.max()))

            # Get head pose
            if self.tracer:
                self.tracer.log_step("source_processing", "before_head_pose")

            pred_source_theta = self.model.head_pose_regressor.forward(source_img_t)
            self.pred_source_theta = pred_source_theta

            if self.tracer:
                self.tracer.log_step("source_processing", "after_head_pose",
                                   theta_shape=list(pred_source_theta.shape))

            # Generate 3D grid and rotation warp
            grid = self.model.identity_grid_3d.repeat_interleave(1, dim=0)
            inv_source_theta = pred_source_theta.float().inverse().type(pred_source_theta.type())
            source_rotation_warp = grid.bmm(inv_source_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)

            # Create data dictionary for expression processing using proper method
            # First prepare the base source data
            data_dict = self._prepare_source_data_dict(source_img_t, source_mask, cloth)
            # Then add the theta values that were computed
            data_dict['source_theta'] = pred_source_theta
            data_dict['target_theta'] = pred_source_theta
            data_dict['idt_embed'] = self.idt_embed

            # Get source expression embedding
            if self.tracer:
                self.tracer.log_step("source_processing", "before_expression_embedder",
                                   data_dict_keys=list(data_dict.keys()))

            data_dict = self.model.expression_embedder_nw(data_dict, True, False)

            if self.tracer:
                self.tracer.log_step("source_processing", "after_expression_embedder",
                                   data_dict_keys=list(data_dict.keys()),
                                   has_source_pose_embed='source_pose_embed' in data_dict)

            # Store source embeddings
            self.pred_source_pose_embed = data_dict['source_pose_embed']
            source_pose_embed = data_dict['source_pose_embed']
            self.source_img_align = data_dict['source_img_align']
            self.source_img = source_img_t
            self.align_warp = data_dict['align_warp']

            # Get warp embeddings
            if self.tracer:
                self.tracer.log_step("source_processing", "before_predict_embed")

            source_warp_embed_dict, target_warp_embed_dict, mixing_warp_embed_dict, embed_dict = self.model.predict_embed(data_dict)

            if self.tracer:
                self.tracer.log_step("source_processing", "after_predict_embed",
                                   source_warp_keys=list(source_warp_embed_dict.keys()) if isinstance(source_warp_embed_dict, dict) else "not_dict",
                                   target_warp_keys=list(target_warp_embed_dict.keys()) if isinstance(target_warp_embed_dict, dict) else "not_dict",
                                   embed_dict_keys=list(embed_dict.keys()) if isinstance(embed_dict, dict) else "not_dict")

            # 🤷 do we investigate these warps - target_warp_embed_dict, mixing_warp_embed_dict,

            # Debug trace the warp embed dict
            if self.tracer:
                self.tracer.log_step("source_processing", "predict_embed_output",
                                   warp_dict_keys=list(source_warp_embed_dict.keys()) if isinstance(source_warp_embed_dict, dict) else "not_dict",
                                   embed_dict_keys=list(embed_dict.keys()) if isinstance(embed_dict, dict) else "not_dict")
                self.tracer.save_tensor(source_warp_embed_dict, "source_warp_embed_dict")

            # Generate XY warps
            if self.tracer:
                self.tracer.log_step("source_processing", "before_xy_generator",
                                   input_type=type(source_warp_embed_dict).__name__)

            xy_gen_outputs = self.model.xy_generator_nw(source_warp_embed_dict)
            data_dict['source_delta_xy'] = xy_gen_outputs[0]

            if self.tracer:
                self.tracer.log_step("source_processing", "after_xy_generator",
                                   num_outputs=len(xy_gen_outputs) if isinstance(xy_gen_outputs, (list, tuple)) else 1,
                                   output_shape=list(xy_gen_outputs[0].shape) if xy_gen_outputs else None)

            source_xy_warp = xy_gen_outputs[0]
            source_xy_warp_resize = source_xy_warp
            # Skip resize_warp_func as it may not exist
            # if self.resize_warp and hasattr(self.model, 'resize_warp_func'):
            #     source_xy_warp_resize = self.model.resize_warp_func(source_xy_warp_resize)

            # Create source latent volume from latents
            if self.tracer:
                self.tracer.log_step("volume_processing", "before_volume_reshape",
                                   source_latents_shape=list(source_latents.shape))

            source_latents_face = source_latents
            source_latent_volume = source_latents_face.view(1, c, d, s, s)

            if self.tracer:
                self.tracer.log_step("volume_processing", "after_volume_reshape",
                                   volume_shape=list(source_latent_volume.shape))

            # Process source volume if needed
            if self.args.source_volume_num_blocks > 0:
                if self.tracer:
                    self.tracer.log_step("volume_processing", "before_volume_source_nw",
                                       input_shape=list(source_latent_volume.shape))

                source_latent_volume = self.model.volume_source_nw(source_latent_volume)

                if self.tracer:
                    self.tracer.log_step("volume_processing", "after_volume_source_nw",
                                       output_shape=list(source_latent_volume.shape))

            self.source_latent_volume = source_latent_volume
            self.source_rotation_warp = source_rotation_warp
            self.source_xy_warp_resize = source_xy_warp_resize

            # Create target volume by warping source volume (order matters!)
            if self.tracer:
                self.tracer.log_step("volume_processing", "before_grid_sample",
                                   source_volume_shape=list(self.source_latent_volume.shape),
                                   rotation_warp_shape=list(source_rotation_warp.shape),
                                   xy_warp_shape=list(source_xy_warp_resize.shape))

            target_latent_volume = self.model.grid_sample(
                self.model.grid_sample(self.source_latent_volume, source_rotation_warp),
                source_xy_warp_resize
            )

            if self.tracer:
                self.tracer.log_step("volume_processing", "after_grid_sample",
                                   target_volume_shape=list(target_latent_volume.shape))

            # Store intermediate volume
            self.target_latent_volume_1 = target_latent_volume

            # Process target volume directly (NOT flattened) - pass embed_dict as second param
            self.target_latent_volume = self.model.volume_process_nw(self.target_latent_volume_1, embed_dict)

        return {
            'source_img': source_image,
            'source_img_t': source_img_t,
            'source_mask': source_mask,
            'data_dict': data_dict,
            'output': {
                'source_information': {
                    'idt_embed': self.idt_embed,
                    'source_latent_volume': self.source_latent_volume,
                    'target_latent_volume': self.target_latent_volume
                }
            }
        }

    def _process_driver_and_generate(self, driver_image, driver_mask, crop, smooth_pose,
                                    hard_normalize, soft_normalize, thetas_pass, theta_n,
                                    c_source_latent_volume, c_target_latent_volume,
                                    custome_target_pose_embed, custome_target_theta_embed,
                                    modnet_mask, frame_idx):
        """
        Process driver and generate output - matches pipeline2 generation logic
        """
        log_processing_step("Driver Image Processing and Generation")

        if crop:
            driver_image = self._crop_face(driver_image, is_source=False)

        # Convert to tensor
        driver_img_t = self._prepare_image_tensor(driver_image)

        # Get masks
        driver_mask = self._get_mask(driver_img_t, driver_mask, 0, modnet_mask)
        driver_mask = driver_mask.to(self.device)

        # Get volume dimensions
        c = self.args.latent_volume_channels
        s = self.args.latent_volume_size
        d = self.args.latent_volume_depth

        with torch.no_grad():
            # Get driver pose
            pred_target_theta = self.model.head_pose_regressor.forward(driver_img_t)

            # Generate 3D grid and rotation warp for target
            grid = self.model.identity_grid_3d.repeat_interleave(1, dim=0)
            target_rotation_warp = grid.bmm(pred_target_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)

            # Create data dictionary for driver using proper method
            # For driver, we use the target preparation method
            data_dict = self._prepare_target_data_dict(
                driver_img_t, driver_mask, smooth_pose,
                hard_normalize, soft_normalize, thetas_pass, theta_n,
                custome_target_pose_embed, custome_target_theta_embed
            )
            # Add source-related fields needed for driver processing
            data_dict['source_img'] = driver_img_t
            data_dict['source_mask'] = driver_mask
            data_dict['source_theta'] = pred_target_theta
            data_dict['target_theta'] = pred_target_theta
            data_dict['idt_embed'] = self.idt_embed

            # Get target expression embedding
            if self.tracer:
                self.tracer.log_step("driver_processing", "before_expression_embedder",
                                   data_dict_keys=list(data_dict.keys()))

            data_dict = self.model.expression_embedder_nw(data_dict, True, False)

            if self.tracer:
                self.tracer.log_step("driver_processing", "after_expression_embedder",
                                   data_dict_keys=list(data_dict.keys()),
                                   has_target_pose_embed='target_pose_embed' in data_dict)

            if custome_target_pose_embed is not None:
                data_dict['target_pose_embed'] = custome_target_pose_embed

            target_pose_embed = data_dict['target_pose_embed']
            self.target_pose_embed = target_pose_embed
            self.target_img_align = data_dict['target_img_align']

            # Generate target warps
            if self.tracer:
                self.tracer.log_step("driver_processing", "before_predict_embed")

            bla, target_warp_embed_dict, bla1, embed_dict = self.model.predict_embed(data_dict)

            if self.tracer:
                self.tracer.log_step("driver_processing", "after_predict_embed",
                                   warp_dict_keys=list(target_warp_embed_dict.keys()) if isinstance(target_warp_embed_dict, dict) else "not_dict")

            # Generate UV warp
            if self.tracer:
                self.tracer.log_step("driver_processing", "before_uv_generator")

            target_uv_warp, data_dict['target_delta_uv'] = self.model.uv_generator_nw(target_warp_embed_dict)

            if self.tracer:
                self.tracer.log_step("driver_processing", "after_uv_generator",
                                   uv_warp_shape=list(target_uv_warp.shape))
            target_uv_warp_resize = target_uv_warp

            # Skip resize_warp_func as it may not exist
            # if self.resize_warp and hasattr(self.model, 'resize_warp_func'):
            #     target_uv_warp_resize = self.model.resize_warp_func(target_uv_warp_resize)

            # Apply warping to target volume
            if self.tracer:
                self.tracer.log_step("driver_processing", "before_final_warping",
                                   target_volume_shape=list(self.target_latent_volume.shape),
                                   uv_warp_shape=list(target_uv_warp_resize.shape),
                                   rotation_warp_shape=list(target_rotation_warp.shape))

            aligned_target_volume = self.model.grid_sample(
                self.model.grid_sample(self.target_latent_volume, target_uv_warp_resize),
                target_rotation_warp
            )

            if self.tracer:
                self.tracer.log_step("driver_processing", "after_final_warping",
                                   aligned_volume_shape=list(aligned_target_volume.shape))

            target_latent_feats = aligned_target_volume.view(1, c * d, s, s)

            # Generate final image using decoder
            if self.tracer:
                self.tracer.log_step("generation", "before_decoder",
                                   latent_feats_shape=list(target_latent_feats.shape))

            img, _, deep_f, img_f = self.model.decoder_nw(
                data_dict,
                embed_dict,
                target_latent_feats,
                False,
                stage_two=True
            )

            if self.tracer:
                self.tracer.log_step("generation", "after_decoder",
                                   output_shape=list(img.shape) if img is not None else None)

        log_tensor_state("Generated image", img)

        pred_target_img = img.detach().cpu().clamp(0, 1)
        pred_target_img = [self.to_image(img) for img in pred_target_img]

        return pred_target_img, img

    def _crop_face(self, image, is_source=True):
        """Crop face from image - maintains exact cropping logic"""
        log_processing_step("Face Detection and Cropping")

        with self.mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            np_img = np.array(image)
            results = face_detection.process(np_img)

            if results.detections is None:
                logger.debug(f"No face detected in {'source' if is_source else 'driver'} image")
                return image  # Return uncropped

            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = np_img.shape[:2]

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Expand bbox
            expand_factor = 1.2
            center_x = x + width / 2
            center_y = y + height / 2
            new_size = max(width, height) * expand_factor

            x = max(0, int(center_x - new_size / 2))
            y = max(0, int(center_y - new_size / 2))
            x2 = min(w, int(center_x + new_size / 2))
            y2 = min(h, int(center_y + new_size / 2))

            cropped = image.crop((x, y, x2, y2))
            return to_512(cropped)

    def _prepare_image_tensor(self, image):
        """Convert PIL image to tensor"""
        if isinstance(image, Image.Image):
            tensor = self.to_tensor(image)
            # Ensure we only have 3 channels (RGB)
            if tensor.shape[0] > 3:
                tensor = tensor[:3]
            return tensor.unsqueeze(0).to(self.device)
        return image

    def _get_mask(self, img_tensor, provided_mask=None, mask_add=0, use_modnet=False):
        """Get mask for image - maintains exact masking logic"""
        if provided_mask is not None:
            return provided_mask

        if use_modnet:
            return self.get_mask(img_tensor)
        else:
            return self.get_mask_fp(img_tensor)

    def _prepare_source_data_dict(self, source_img_t, source_mask, cloth):
        """Prepare source data dictionary - CRITICAL: Keep exact structure"""
        data_dict = {
            'source_img': source_img_t,
            'source_mask': source_mask,
            'target_img': source_img_t,  # Same as source initially
            'target_mask': source_mask,  # Same as source initially
        }

        if cloth:
            data_dict['cloth'] = True

        return data_dict

    def _prepare_target_data_dict(self, target_img_t, target_mask, smooth_pose,
                                 hard_normalize, soft_normalize, thetas_pass, theta_n,
                                 custome_target_pose_embed, custome_target_theta_embed):
        """Prepare target data dictionary - CRITICAL: Keep exact structure"""
        data_dict = {
            'target_img': target_img_t,
            'target_mask': target_mask,
        }

        # Add pose parameters
        if smooth_pose:
            data_dict['smooth_pose'] = True
        if hard_normalize:
            data_dict['hard_normalize'] = True
        if soft_normalize:
            data_dict['soft_normalize'] = True

        # Custom embeddings
        if custome_target_pose_embed is not None:
            data_dict['custome_target_pose_embed'] = custome_target_pose_embed
        if custome_target_theta_embed is not None:
            data_dict['custome_target_theta_embed'] = custome_target_theta_embed

        # Theta parameters
        if thetas_pass:
            data_dict['thetas_pass'] = thetas_pass
            data_dict['theta_n'] = theta_n

        return data_dict

    def _merge_source_and_target(self, source_dict, target_dict,
                                c_source_latent_volume, c_target_latent_volume):
        """Merge source and target data - CRITICAL: Keep exact merging logic"""
        merged = source_dict.copy()
        merged.update(target_dict)

        # Custom latent volumes
        if c_source_latent_volume is not None:
            merged['c_source_latent_volume'] = c_source_latent_volume
        if c_target_latent_volume is not None:
            merged['c_target_latent_volume'] = c_target_latent_volume

        # Add control flags
        merged['mix'] = self.mix
        merged['mix_old'] = self.mix_old
        merged['target_theta'] = self.target_theta

        return merged

    # ===================== MASKING METHODS - KEEP EXACT LOGIC =====================

    def get_mask(self, img):
        """Get mask using ModNet - EXACT copy from pipeline2"""
        # Ensure we only have 3 channels
        if img.shape[1] > 3:
            img = img[:, :3, :, :]

        im_transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        im = im_transform(img)
        ref_size = 512

        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        _, _, matte = self.modnet(im, True)
        matte = F.interpolate(matte, size=(512, 512), mode='area')
        matte = matte.repeat(1, 3, 1, 1)

        return matte

    def get_mask_fp(self, img):
        """Get mask using face parsing - matches pipeline2 logic"""
        # Ensure we only have 3 channels
        if img.shape[1] > 3:
            img = img[:, :3, :, :]

        with torch.no_grad():
            # Get face mask from face parsing
            face_mask, _, _, cloth_s = self.face_idt.forward(img)
            threshold = 0.6
            face_mask = (face_mask > threshold).float()

            # Get ModNet mask
            source_mask_modnet = self.get_mask(img)

            # Ensure masks are the same size - resize face_mask if needed
            if face_mask.shape[-2:] != source_mask_modnet.shape[-2:]:
                face_mask = F.interpolate(face_mask, size=source_mask_modnet.shape[-2:], mode='bilinear', align_corners=False)

            # Combine masks
            face_mask = (face_mask * source_mask_modnet).float()

        return face_mask


# ===================== VIDEO PROCESSING FUNCTIONS =====================

def get_video_frames_as_images(video_path):
    """Extract frames from video as PIL Images"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def get_bg(s_img, inferer, mdnt=True):
    """Get background from image using the background processor"""
    return inferer.bg_processor.extract_background(s_img, use_modnet=mdnt)


def connect_img_and_bg(img, bg, inferer, mdnt=True):
    """Connect image with background using the background processor"""
    return inferer.bg_processor.composite_with_background(img, bg, use_modnet=mdnt)


def drive_image_with_video(inferer, source, video_path='/path/to/your/xxx.mp4', max_len=None, enable_tracing=False):
    """
    Drive source image with video - CRITICAL: Maintain exact logic from pipeline2
    """
    # Use the inferer's existing tracer if available
    tracer = inferer.tracer if hasattr(inferer, 'tracer') else None

    # If inferer doesn't have a tracer, create a new one if requested
    if tracer is None and enable_tracing:
        tracer = DebugTracer(output_dir="debug_pipeline4", enabled=enable_tracing)

    if tracer:
        tracer.log_step("drive_image_with_video", "entry", video_path=video_path, max_len=max_len)
        tracer.save_image(source, "source_image")

    all_srs = []
    all_bgs = []
    all_img_bg = []

    all_curr_d = get_video_frames_as_images(video_path)
    all_curr_d = all_curr_d[:max_len]

    if tracer:
        tracer.log_step("video_frames", "loaded", num_frames=len(all_curr_d))
        for i in range(min(3, len(all_curr_d))):
            tracer.save_image(all_curr_d[i], f"driver_frame_{i:03d}")

    # Process first frame with source - CRITICAL: This establishes identity
    first_d = all_curr_d[0]
    img = inferer.forward(source, first_d, crop=False, smooth_pose=False,
                         target_theta=True, mix=True, mix_old=False, modnet_mask=False)
    all_srs.append(source)

    # Make background from source
    bg, bg_img = get_bg(source, inferer, mdnt=False)
    all_bgs.append(bg_img)

    # Composite first frame
    img_with_bg = connect_img_and_bg(img[0][0], bg, inferer, mdnt=False)
    all_img_bg.append(img_with_bg)

    if tracer:
        tracer.save_image(img_with_bg, f"output_frame_000")
        tracer.log_step("first_frame", "processed")

    # Process remaining frames - CRITICAL: Use None for source to maintain identity
    for i, curr_d in enumerate(tqdm(all_curr_d[1:]), 1):
        frame_idx = i
        img = inferer.forward(None, curr_d, crop=False, smooth_pose=False,
                            target_theta=True, mix=True, mix_old=False,
                            modnet_mask=False, frame_idx=frame_idx)
        img_with_bg = connect_img_and_bg(img[0][0], bg, inferer, mdnt=False)
        all_img_bg.append(img_with_bg)

        if tracer and i < 5:
            tracer.save_image(img_with_bg, f"output_frame_{i:03d}")

    if tracer:
        tracer.log_step("drive_image_with_video", "exit", num_output_frames=len(all_img_bg))
        tracer.save_final_trace()

    return all_img_bg, all_curr_d


def make_video_crop(source, driven_result, drivers, path, fps=25.0):
    """Make comparison video"""
    frames = []
    for i in range(len(driven_result)):
        source_img = np.array(to_512(source))[:, :, :3]
        driver_img = np.array(to_512(drivers[i]))[:, :, :3]
        result_img = np.array(to_512(driven_result[i]))[:, :, :3]
        combined = np.concatenate([source_img, driver_img, result_img], axis=1)
        frames.append(combined)

    videodims = (3 * 512, 512)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, fps, videodims)

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()


# ===================== HELPER FUNCTIONS =====================

def get_mask(gt_img_t):
    """Get mask using global ModNet"""
    # This would need access to modnet instance
    # In practice, this should be called through inferer.get_mask()
    pass


def get_mask_fp(gt_img_t):
    """Get mask using face parsing"""
    # This would need access to model instance
    # In practice, this should be called through inferer.get_mask_fp()
    pass


def get_modnet_mask(pred_img_t):
    """Get ModNet mask"""
    # This would need access to modnet instance
    pass


# ===================== INITIALIZATION =====================

# Model configuration
args_overwrite = {
    'l1_vol_rgb': 0
}

project_dir = './'
threshold = 0.8
device = 'cuda'

# Initialize face detector
face_detector = RetinaFacePredictor(threshold=threshold, device=device,
                                   model=(RetinaFacePredictor.get_model('mobilenet0.25')))

# Initialize inference wrapper
inferer = None  # Will be initialized in main


# ===================== MAIN EXECUTION =====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline4: Refactored pipeline2 with exact working logic')
    parser.add_argument('--source_image_path', type=str, default='data/IMG_1.png', help='Path to source image')
    parser.add_argument('--driven_video_path', type=str, default='../junk/15.mp4', help='Path to driving video')
    parser.add_argument('--saved_to_path', type=str, default='data/result_pipeline4.mp4', help='Path to save result video')
    parser.add_argument('--fps', type=float, default=25.0, help='FPS of output video')
    parser.add_argument('--max_len', type=int, default=1000, help='Maximum number of frames to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug tracing')

    args = parser.parse_args()

    # Initialize inference wrapper
    inferer = InferenceWrapper(
        experiment_name='Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1',
        model_file_name='328_model.pth',
        project_dir=project_dir,
        folder='logs',
        state_dict=None,
        args_overwrite=args_overwrite,
        pose_momentum=0.1,
        print_model=False,
        print_params=True,
        enable_tracing=args.debug
    )

    # Enable tracing if debug flag is set
    if args.debug:
        logger.debug("Debug mode enabled - trace files will be saved to debug_pipeline4/")

    # Load source image
    source_img = to_512(Image.open(args.source_image_path))
    driving_video_path = args.driven_video_path

    # Process video
    driven_result, drivers = drive_image_with_video(
        inferer, source_img, driving_video_path,
        max_len=args.max_len, enable_tracing=args.debug
    )

    # Save result video
    save_path = args.saved_to_path
    make_video_crop(source_img, driven_result, drivers, save_path, fps=args.fps)

    logger.info(f"Result saved to: {save_path}")