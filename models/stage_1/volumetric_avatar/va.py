import torch
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
import numpy as np
import itertools
from torch.cuda import amp
import sys
sys.path.append('.')
from networks import basic_avatar, volumetric_avatar
from utils import args as args_utils
from utils import spectral_norm, weight_init, point_transforms
from skimage.measure import label
from .va_losses_and_visuals import calc_train_losses, calc_test_losses, prepare_input_data, MODNET, init_losses
from .va_losses_and_visuals import visualize_data, get_visuals, draw_stickman
from .va_arguments import VolumetricAvatarConfig
from networks.volumetric_avatar.utils import requires_grad, d_logistic_loss, d_r1_loss, g_nonsaturating_loss, \
    _calc_r1_penalty
from scipy import linalg
from dataclasses import dataclass
from torch.autograd import Variable
import math
from utils.non_specific import calculate_obj_params, FaceParsingBUG, get_mixing_theta, align_keypoints_torch


from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
from ibug.face_parsing.utils import label_colormap
from ibug.roi_tanh_warping import roi_tanh_polar_restore, roi_tanh_polar_warp
from torchvision import transforms
from .va_arguments import VolumetricAvatarConfig
from utils import point_transforms
from omegaconf import OmegaConf
from typing import Dict, Optional, Tuple, List
import time
from collections import defaultdict
import logging
from logger import logger  
import traceback

to_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

import contextlib
from rich.console import Console
# from textual_image.renderable import Image  # Commented out - not essential
from mem import memory_stats
console = Console()


class TimingStats:
    def __init__(self, verbose=0):
        """
        Initialize TimingStats with configurable verbosity.
        
        Args:
            verbose (int): 
                0 = No logging
                1 = Log only total time every N iterations
                2 = Log major stage timings every N iterations
                3 = Log all stage timings every N iterations
                4 = Log all timings every iteration + running averages
        """
        self.timings = defaultdict(list)
        self.verbose = verbose
        self.running_avgs = defaultdict(lambda: [0, 0])  # [sum, count]
        
    def update(self, stage, duration):
        self.timings[stage].append(duration)
        # Update running average
        self.running_avgs[stage][0] += duration
        self.running_avgs[stage][1] += 1
        
        # Immediate logging for verbose=4
        avg = self.running_avgs[stage][0] / self.running_avgs[stage][1]
        print(f"{stage}: {duration*1000:.2f}ms (avg: {avg*1000:.2f}ms)")
    
    def get_stats(self, major_stages_only=False):
        stats = {}
        major_stages = {'total', 'feature_extraction', 'volume_processing', 'target_generation'}
        
        for stage, times in self.timings.items():
            if major_stages_only and stage not in major_stages:
                continue
            avg_time = sum(times) / len(times) if times else 0
            stats[stage] = {
                'avg_ms': avg_time * 1000,
                'last_ms': times[-1] * 1000 if times else 0,
                'count': len(times)
            }
        return stats
    


class Model(nn.Module):

    def __init__(self, cfg, training=True, rank=0, exp_dir=None):
        super(Model, self).__init__()
        
        self.exp_dir = exp_dir
       
        # Use the passed config instead of loading from hardcoded path
        self.cfg = cfg if cfg is not None else OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
        self.args = self.cfg
        args = self.cfg
        self.va_config = VolumetricAvatarConfig(args)
        self.weights = self.va_config.get_weights()


        
        self.rank=rank
        self.num_source_frames = args.num_source_frames
        self.num_target_frames = args.num_target_frames

        self.resize_d = self._resize_down
        self.resize_u = self._resize_up



        self.embed_size = args.gen_embed_size
        self.num_source_frames = args.num_source_frames  # number of identities per batch
        self.embed_size = args.gen_embed_size
        self.pred_seg = args.dec_pred_seg
        self.use_stylegan_d = args.use_stylegan_d
        self.bn = nn.BatchNorm1d(512, affine=False)
        self.thetas_pool = []
        if self.pred_seg:
            self.seg_loss = nn.BCELoss()
        self.pred_flip = args.gen_pred_flip
        self.pred_mixing = args.gen_pred_mixing
        assert self.num_source_frames == 1, 'No support for multiple sources'
        self.background_net_input_channels = 64


        # if self.args.w_eyes_loss_l1>0 or self.args.w_mouth_loss_l1>0 or self.args.w_ears_loss_l1>0:
        self.face_parsing_bug = FaceParsingBUG()


        self.m_key_diff = 0
        self.init_networks(args, training)


        self.prev_targets = None
        self.autocast = args.use_amp_autocast
        self.apply(weight_init.weight_init(args.init_type, args.init_gain))
        self.dec_pred_conf = args.dec_pred_conf
        self.sep_train_losses = args.sep_train_losses
        self.resize_warp = args.warp_output_size != args.gen_latent_texture_size
        self.warp_resize_stride = (
            1, args.warp_output_size // args.gen_latent_texture_size,
            args.warp_output_size // args.gen_latent_texture_size)
        
        self.resize_d = self._resize_down
        self.resize_u = self._resize_up


        grid_s = torch.linspace(-1, 1, self.args.aug_warp_size)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer('identity_grid_2d', torch.stack([u, v], dim=2).view(1, -1, 2), persistent=False)

        grid_s = torch.linspace(-1, 1, self.args.latent_volume_size)
        grid_z = torch.linspace(-1, 1, self.args.latent_volume_depth)
        w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
        e = torch.ones_like(u)
        self.register_buffer('identity_grid_3d', torch.stack([u, v, w, e], dim=3).view(1, -1, 4), persistent=False)
        self.only_cycle_embed = args.only_cycle_embed

        self.use_masked_aug = args.use_masked_aug
        self.num_b_negs = self.args.num_b_negs
        self.pred_cycle = args.pred_cycle

        # # Apply spectral norm
        if args.use_sn:
            spectral_norm.apply_sp_to_nets(self)

        # Apply weight standartization 
        if args.use_ws:
            volumetric_avatar.utils.apply_ws_to_nets(self)

        # Calculate params
        calculate_obj_params(self)

        if training:
            self.init_losses(args)

    def _resize_down(self, img):
        """Replacement for resize_d lambda"""
        return F.interpolate(img, mode='bilinear',
                           size=(224, 224),
                           align_corners=False)

    def _resize_up(self, img):
        """Replacement for resize_u lambda"""
        return F.interpolate(img, mode='bilinear',
                           size=(256, 256),
                           align_corners=False)

    def _grid_sample_fn(self, inputs, grid):
        """Replacement for grid_sample lambda"""
        return F.grid_sample(inputs.float(), grid.float(),
                           padding_mode=self.args.grid_sample_padding_mode)
    
    def init_networks(self, args, training):
        
        ##################################
        #### Encoders ####
        ##################################

        ## Define image encoder
        self.local_encoder_nw = volumetric_avatar.LocalEncoder(self.va_config.local_encoder_cfg)

        ## Define background nets; default = False
        if self.args.use_back:
            in_u = self.args.background_net_input_channels
            c = self.args.latent_volume_channels
            d = self.args.latent_volume_depth
            self.background_net_out_channels = self.args.latent_volume_depth * self.args.latent_volume_channels
            u = self.background_net_out_channels
            
            self.local_encoder_back_nw = volumetric_avatar.LocalEncoderBack(self.va_config.local_encoder_back_cfg)
            
            self.backgroung_adding_nw = nn.Sequential(*[nn.Conv2d(
                in_channels=c * d + u,
                out_channels=c * d,
                kernel_size=(1, 1),
                padding=0,
                bias=False),
                nn.ReLU(),
            ])

            self.background_process_nw = volumetric_avatar.UNet(in_u, u, base=self.args.back_unet_base, max_ch=self.args.back_unet_max_ch, norm='gn')

        # Define volume rendering net; default = False
        if self.args.volume_rendering:
            self.volume_renderer_nw = volumetric_avatar.VolumeRenderer(self.va_config.volume_renderer_cfg)

        # Define idt embedder net - for adaptivity of networks and for face warping help
        self.idt_embedder_nw = volumetric_avatar.IdtEmbed(self.va_config.idt_embedder_cfg)


        # Define expression embedder net - derive latent vector of emotions
        self.expression_embedder_nw = volumetric_avatar.ExpressionEmbed(self.va_config.exp_embedder_cfg)

        ##################################
        #### Warp ####
        ##################################

        # Operator that transform exp_emb to extended exp_emb (to match idt_emb size)
        self.pose_unsqueeze_nw = nn.Linear(args.lpe_output_channels_expression, args.gen_max_channels * self.embed_size ** 2,
                                        bias=False)


        # Operator that combine idt_imb and extended exp_emb together (a "+" sign of a scheme)
        self.warp_embed_head_orig_nw = nn.Conv2d(
            in_channels=args.gen_max_channels*(2 if self.args.cat_em else 1),
            out_channels=args.gen_max_channels,
            kernel_size=(1, 1),
            bias=False)
        
        # Define networks from warping to (xy) and from (uv) canical volume cube
        self.xy_generator_nw = volumetric_avatar.WarpGenerator(self.va_config.warp_generator_cfg)
        self.uv_generator_nw = volumetric_avatar.WarpGenerator(self.va_config.warp_generator_cfg)

        ##################################
        #### Volume process ####
        ##################################

        ## Define 3D net that goes right after image encoder
        if self.args.source_volume_num_blocks>0:
            
            if self.args.unet_first:
                print('aaaaaaaaaaaaaaaa')
                self.volume_source_nw = volumetric_avatar.Unet3D(self.va_config.unet3d_cfg_s)
            else:
                print('bbbbbbbbbbb')
                # self.volume_source_nw = volumetric_avatar.Unet3D(self.va_config.unet3d_cfg_s)
                self.volume_source_nw = volumetric_avatar.VPN_ResBlocks(self.va_config.VPN_resblocks_source_cfg)



        # If we want to use additional learnable tensor - like avarage person
        if self.args.use_tensor:
            d = self.args.gen_latent_texture_depth
            s = self.args.gen_latent_texture_size
            c = self.args.gen_latent_texture_channels
            self.avarage_tensor_ts = nn.Parameter(Variable(  (torch.rand((1,c,d,s,s), requires_grad = True)*2 - 1)*math.sqrt(6./(d*s*s*c))  ).cuda(), requires_grad = True)

        # Net that process volume after first duble-warping
        self.volume_process_nw = volumetric_avatar.Unet3D(self.va_config.unet3d_cfg)


        ## Define 3D net that goes before image decoder
        if self.args.pred_volume_num_blocks>0:
            
            if self.args.unet_first:
                self.volume_pred_nw = volumetric_avatar.Unet3D(self.va_config.unet3d_cfg_s)
            else:
                self.volume_pred_nw = volumetric_avatar.VPN_ResBlocks(self.va_config.VPN_resblocks_pred_cfg)

        ##################################
        #### Decoding ####
        ##################################
        self.decoder_nw = volumetric_avatar.Decoder(self.va_config.decoder_cfg)


            
        ##################################
        #### Discriminators ####
        ##################################
        if training:
            self.discriminator_ds = basic_avatar.MultiScaleDiscriminator(self.va_config.dis_cfg)
            self.discriminator_ds.apply(weight_init.weight_init(args.dis_init_type, args.dis_init_gain))

            if self.args.use_mix_dis:
                self.discriminator2_ds = basic_avatar.MultiScaleDiscriminator(self.va_config.dis_2_cfg)
                self.discriminator2_ds.apply(weight_init.weight_init(args.dis_init_type, args.dis_init_gain))

            if self.use_stylegan_d:
                self.r1_loss = torch.tensor(0.0)
                self.stylegan_discriminator_ds = basic_avatar.DiscriminatorStyleGAN2(size=self.args.image_size,
                                                                                  channel_multiplier=1, my_ch=2)

        
        ###########################################
        #### Non-trainable additional networks ####
        ###########################################
        self.face_idt = volumetric_avatar.FaceParsing(None, 'cuda', project_dir = self.args.project_dir)



        if self.args.use_mix_losses or self.pred_seg:
            self.get_mask = MODNET(project_dir = self.args.project_dir)

        if self.args.estimate_head_pose_from_keypoints:
            self.head_pose_regressor = volumetric_avatar.HeadPoseRegressor(args.head_pose_regressor_path, args.num_gpus)


        if args.warp_norm_grad:
            self.grid_sample = volumetric_avatar.GridSample(args.gen_latent_texture_size)
        else:
            self.grid_sample = self._grid_sample_fn

        self.get_face_vector = volumetric_avatar.utils.Face_vector(self.head_pose_regressor, half=False)
        self.get_face_vector_resnet = volumetric_avatar.utils.Face_vector_resnet(half=False, project_dir=self.args.project_dir)
        self.face_parsing_bug = FaceParsingBUG()

        grid_s = torch.linspace(-1, 1, self.args.aug_warp_size)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer('identity_grid_2d', torch.stack([u, v], dim=2).view(1, -1, 2), persistent=False)

        grid_s = torch.linspace(-1, 1, self.args.latent_volume_size)
        grid_z = torch.linspace(-1, 1, self.args.latent_volume_depth)
        w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
        e = torch.ones_like(u)
        self.register_buffer('identity_grid_3d', torch.stack([u, v, w, e], dim=3).view(1, -1, 4), persistent=False)



    def init_losses(self, args):
        return init_losses(self, args)



    def _get_batch_dimensions(self, data_dict):
        """Extract batch dimensions from input data."""
        b = data_dict['source_img'].shape[0]
        c = self.args.latent_volume_channels
        s = self.args.latent_volume_size
        d = self.args.latent_volume_depth
        return b, c, s, d

    def _process_face_masks(self, data_dict):
        """Process face parsing masks for source and target images."""
        if self.args.use_ibug_mask:
            return self._process_ibug_masks(data_dict)
        return self._process_default_masks(data_dict)


    def _extract_features_and_embeddings(self, data_dict: Dict[str, torch.Tensor], modnet) -> torch.Tensor:
        """Extract source identity features and canonical volume for Stage 2 VASA training."""
        try:
            # 1. Generate masks using MODNet
            # img = data_dict['source_img'].cpu().detach()
            # console.print(img)
            _, _, source_mask_modnet = modnet(data_dict['source_img'].cuda(), True)
            source_mask_modnet = source_mask_modnet.to(data_dict['source_img'].device)
            
            # Get face parsing mask as fallback
            face_mask_source, _, _, _ = self.face_idt.forward(data_dict['source_img'])
            face_mask_source = (face_mask_source > 0.6).float()
            
            # Combine masks
            source_mask = source_mask_modnet if self.args.use_modnet_mask else face_mask_source
            
            # Apply mask to source image
            source_masked = data_dict['source_img'] * source_mask
            
            # 2. Extract identity embeddings
            data_dict['idt_embed'] = self.idt_embedder_nw(source_masked)
            
            # 3. Extract base latent features
            source_latents = self.local_encoder_nw(source_masked)
            
            # 4. Process source volume
            B = source_latents.shape[0]
            source_latent_volume = source_latents.view(
                B,
                self.args.latent_volume_channels,
                self.args.latent_volume_depth,
                self.args.latent_volume_size,
                self.args.latent_volume_size
            )
            
            if self.args.source_volume_num_blocks > 0:
                source_latent_volume = self.volume_source_nw(source_latent_volume)
                
            # 5. Process canonical volume - skip expression embedding
            if self.args.unet_first:
                canonical_volume = self.volume_process_nw(source_latent_volume)
            else:
                canonical_volume = source_latent_volume

            return canonical_volume

        except Exception as e:
            logger.error(f"Error in _extract_features_and_embeddings: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    def _generate_warping_fields(self, source_warp_embed_dict, target_warp_embed_dict):
        """Generate warping fields for both source and target."""
        xy_gen_warp, _ = self.xy_generator_nw(source_warp_embed_dict)
        target_uv_warp, _ = self.uv_generator_nw(target_warp_embed_dict)
        
        if self.resize_warp:
            xy_gen_warp = self.resize_warp_func(xy_gen_warp)
            target_uv_warp = self.resize_warp_func(target_uv_warp)
            
        return xy_gen_warp, target_uv_warp

    def _process_latent_volume(self, latent_volume, embed_dict, b, c, d, s):
        """Process latent volume through 3D CNNs."""
        latent_volume = latent_volume.view(b, c, d, s, s)

        if self.args.unet_first:
            latent_volume = self.volume_process_nw(latent_volume, embed_dict)
        elif self.args.source_volume_num_blocks > 0:
            latent_volume = self.volume_source_nw(latent_volume)

        # Optionally detach gradients
        if self.args.detach_lat_vol > 0 and self.iteration % self.args.detach_lat_vol == 0:
            latent_volume = latent_volume.detach()

        # Freeze processing network if needed
        if self.args.freeze_proc_nw > 0:
            requires_grad = self.iteration % self.args.freeze_proc_nw != 0
            for param in self.volume_process_nw.parameters():
                param.requires_grad = requires_grad

        return latent_volume

    def _normalize_source_pose(self, latent_volume, data_dict):
        """Transform latent volume from source pose to canonical pose."""
        return self.grid_sample(
            self.grid_sample(latent_volume, data_dict['source_rotation_warp']),
            data_dict['source_xy_warp_resize']
        )

    def _generate_canonical_volume(self, latent_volume, embed_dict):
        """Generate canonical volume representation."""
        if self.args.unet_first and self.args.source_volume_num_blocks > 0:
            return self.volume_source_nw(latent_volume)
        else:
            return self.volume_process_nw(latent_volume, embed_dict)

    def _generate_target_pose(self, canonical_volume, data_dict, background_features=None):
        """Transform canonical volume to target pose and process."""
        # Apply target pose transformation
        aligned_volume = self.grid_sample(
            self.grid_sample(canonical_volume, data_dict['target_uv_warp_resize']),
            data_dict['target_rotation_warp']
        )
        
        # Apply additional volume processing if needed
        if self.args.pred_volume_num_blocks > 0:
            aligned_volume = self.volume_pred_nw(aligned_volume)

        # Process final features
        b, c, d, s = aligned_volume.shape[:4]
        if self.args.use_back and background_features is not None:
            aligned_volume = aligned_volume.view(b, c * d, s, s)
            aligned_volume = self.backgroung_adding_nw(
                torch.cat((aligned_volume, background_features), dim=1)
            )
        elif self.args.volume_rendering:
            aligned_volume, data_dict['pred_tar_img_vol'], data_dict['pred_tar_depth_vol'] = \
                self.volume_renderer_nw(aligned_volume)
        else:
            aligned_volume = aligned_volume.view(b, c * d, s, s)
        
        return aligned_volume

    def _process_neutral_expression(self, data_dict, canonical_volume, target_warp_embed, 
                                background_features, b, c, d, s, epoch, iteration):
        """Process neutral expression generation."""
        context = contextlib.nullcontext() if (epoch == 0 and iteration < 200) else torch.no_grad()
        
        with context:
            canonical_volume_n = canonical_volume.clone()
            
            if self.args.use_back and background_features is not None:
                canonical_volume_n = canonical_volume_n.view(b, c * d, s, s)
                canonical_volume_n = self.backgroung_adding_nw(
                    torch.cat((canonical_volume_n, background_features), dim=1)
                )
            elif self.args.volume_rendering:
                canonical_volume_n, data_dict['pred_tar_img_vol'], data_dict['pred_tar_depth_vol'] = \
                    self.volume_renderer_nw(canonical_volume_n)
            else:
                canonical_volume_n = canonical_volume_n.view(b, c * d, s, s)

            # Generate neutral expression image
            data_dict['pred_neutral_img'], _, _, _ = self.decoder_nw(
                data_dict, target_warp_embed, canonical_volume_n, False, iteration=iteration
            )

            # Extract and align central region
            s_a = data_dict['pred_neutral_img'].shape[-1] // 4
            data_dict['pred_neutral_img_aligned'] = data_dict['pred_neutral_img'][:, :, s_a:3*s_a, s_a:3*s_a]
            
            # Extract expression vector
            data_dict['pred_neutral_expr_vertor'] = self.expression_embedder_nw.net_face(
                data_dict['pred_neutral_img_aligned']
            )[0]

    def _process_expression_mixing(self, data_dict, canonical_volume, mixing_warp_embed,
                                b, c, d, s, epoch, iteration):
        """Process expression mixing and cycle consistency."""
        if not (self.args.use_mix_losses and self.training):
            return self._test_mixing(data_dict, canonical_volume, mixing_warp_embed, b, c, d, s)

        # Generate mixing warping field
        mixing_warp = self._generate_mixing_warp(mixing_warp_embed, b)
        
        # Apply warping transformations
        aligned_mixing = self._apply_mixing_transformations(
            canonical_volume, mixing_warp, data_dict, b, c, d, s
        )
        
        # Generate mixed results
        data_dict = self._generate_mixing_results(
            data_dict, aligned_mixing, mixing_warp_embed, b, epoch, iteration
        )
        
        # Process expression cycle consistency if enabled
        if self.pred_cycle:
            data_dict = self._process_cycle_consistency(
                data_dict, canonical_volume, b, c, d, s, epoch, iteration
            )

        return data_dict

    def _generate_mixing_warp(self, mixing_warp_embed, b):
        """Generate warping field for expression mixing."""
        mixing_uv_warp, _ = self.uv_generator_nw(mixing_warp_embed)
        if self.resize_warp:
            mixing_uv_warp = self.resize_warp_func(mixing_uv_warp)
            
        mixing_theta, self.thetas_pool = get_mixing_theta(
            self.args, 
            self.source_theta, 
            self.target_theta, 
            self.thetas_pool, 
            self.args.random_theta
        )
        
        mixing_warp = self.identity_grid_3d.repeat_interleave(b, dim=0)
        mixing_warp = mixing_warp.bmm(mixing_theta.transpose(1, 2))
        
        return mixing_warp.view(b, *mixing_uv_warp.shape[1:4], 3)

    def _test_mixing(self, data_dict, canonical_volume, mixing_warp_embed, b, c, d, s):
        """Perform mixing during inference/testing."""
        with torch.no_grad():
            mixing_warp = self._generate_mixing_warp(mixing_warp_embed, b)
            aligned_mixing = self._apply_mixing_transformations(
                canonical_volume, mixing_warp, data_dict, b, c, d, s
            )
            
            self.decoder_nw.eval()
            data_dict['pred_mixing_img'], _, _, _ = self.decoder_nw(
                data_dict, mixing_warp_embed, aligned_mixing, False
            )
            self.decoder_nw.train()
            data_dict['pred_mixing_img'] = data_dict['pred_mixing_img'].detach()
        
        return data_dict
    


    def _compute_pose_warping_fields(self, data_dict, b, d, s):
        """Compute 3D warping fields based on estimated head pose."""
        # Estimate head pose
        with torch.no_grad():
            data_dict['source_theta'], source_scale, data_dict['source_rotation'], source_tr = \
                self.head_pose_regressor.forward(data_dict['source_img'], return_srt=True)
            data_dict['target_theta'], target_scale, data_dict['target_rotation'], target_tr = \
                self.head_pose_regressor.forward(data_dict['target_img'], return_srt=True)

        # Initialize 3D grid
        grid = self.identity_grid_3d.repeat_interleave(b, dim=0)

        # Process source warping
        inv_source_theta = data_dict['source_theta'].float().inverse().type(data_dict['source_theta'].type())
        data_dict = self._process_source_warping(data_dict, inv_source_theta, grid, b, d, s)

        # Process target warping
        data_dict = self._process_target_warping(data_dict, grid, inv_source_theta, b, d, s)

        # Handle canonical volume prediction if enabled
        if self.args.predict_target_canon_vol and not (self.epoch == 0 and self.iteration < 0):
            data_dict = self._process_canonical_warping(
                data_dict, source_scale, target_tr, grid, b, d, s
            )

        return data_dict

    def _process_source_warping(self, data_dict, inv_source_theta, grid, b, d, s):
        """Process source image warping and keypoint transformations."""
        # Compute basic rotation warp
        data_dict['source_rotation_warp'] = grid.bmm(
            inv_source_theta[:, :3].transpose(1, 2)
        ).view(-1, d, s, s, 3)

        # Transform keypoints
        data_dict['source_warped_keypoints'] = data_dict['source_keypoints'].bmm(
            inv_source_theta[:, :3, :3]
        )

        # Process normalized keypoints
        data_dict['source_warped_keypoints_n'] = self._process_normalized_keypoints(
            data_dict['source_warped_keypoints']
        )

        # Align keypoints
        data_dict['source_warped_keypoints_n'], transform_matrix_s = align_keypoints_torch(
            data_dict['source_warped_keypoints_n'],
            data_dict['source_warped_keypoints'],
            nose=True
        )

        transform_matrix_s = transform_matrix_s.to(data_dict['source_rotation_warp'].device)

        # Apply final transformations
        new_m = inv_source_theta[:, :3, :3].bmm(transform_matrix_s[:, :3, :3])
        data_dict['source_warped_keypoints_n'] = data_dict['source_keypoints'].bmm(new_m)
        data_dict['source_warped_keypoints_n'] += transform_matrix_s[:, None, :3, 3]

        # Handle aligned warping if enabled
        if self.args.aligned_warp_rot_source:
            data_dict = self._apply_aligned_source_warping(
                data_dict, inv_source_theta, transform_matrix_s, grid, d, s
            )

        return data_dict

    def _process_target_warping(self, data_dict, grid, inv_source_theta, b, d, s):
        """Process target image warping and transformations."""
        if self.args.aligned_warp_rot_target:
            # Compute inverse transformation
            inv_transform_matrix = transform_matrix_s.float().inverse().type(
                data_dict['target_theta'].type()
            )
            new_m_warp_t = inv_transform_matrix.bmm(data_dict['target_theta'])
            
            # Apply transformations
            data_dict['target_rotation_warp'] = grid.bmm(
                new_m_warp_t[:, :3].transpose(1, 2)
            ).view(-1, d, s, s, 3)
            
            # Transform keypoints
            data_dict['target_pre_warped_keypoints'] = data_dict['source_warped_keypoints_n'].bmm(
                inv_transform_matrix[:, :3, :3]
            )
            data_dict['target_warped_keypoints'] = data_dict['target_pre_warped_keypoints'].bmm(
                data_dict['target_theta'][:, :3, :3]
            )
        else:
            data_dict['target_rotation_warp'] = grid.bmm(
                data_dict['target_theta'][:, :3].transpose(1, 2)
            ).view(-1, d, s, s, 3)

        return data_dict

    def _process_normalized_keypoints(self, source_keypoints):
        """Process normalized keypoints with predefined nose positions."""
        normalized_keypoints = source_keypoints.clone()
        normalized_keypoints[:, 27:31] = torch.tensor([
            [-0.0000, -0.2,   0.22],
            [-0.0000, -0.13,  0.26],
            [-0.0000, -0.06,  0.307],
            [-0.0000, -0.008, 0.310]
        ]).to(source_keypoints.device)
        return normalized_keypoints

    def _process_ibug_masks(self, data_dict):
        """Process face masks using IBUG face parsing."""
        if not self.args.use_old_fp:
            try:
                data_dict = self._generate_face_parsing_masks(data_dict)
            except Exception as e:
                print(f"Face parsing failed, falling back to default: {e}")
                data_dict = self._fallback_face_parsing(data_dict)
        else:
            data_dict = self._fallback_face_parsing(data_dict)

        # Add hat masks
        _, _, hat_s, _ = self.face_idt.forward(data_dict['source_img'])
        _, _, hat_t, _ = self.face_idt.forward(data_dict['target_img'])
        
        data_dict['source_mask_face_pars'] += hat_s
        data_dict['target_mask_face_pars'] += hat_t

        return self._finalize_masks(data_dict)

    def _generate_face_parsing_masks(self, data_dict):
        """Generate face parsing masks using neural network."""
        face_mask_source_list = []
        face_mask_target_list = []
        
        for i in range(data_dict['source_img'].shape[0]):
            # Get source masks
            masks_gt, logits_gt, logits_source_soft, faces = self.face_parsing_bug.get_lips(
                data_dict['source_img'][i]
            )
            # Get target masks
            masks_s1, logits_s1, logits_target_soft, _ = self.face_parsing_bug.get_lips(
                data_dict['target_img'][i]
            )

            # Process softmax logits
            logits_source_soft = logits_source_soft.detach()
            logits_target_soft = logits_target_soft.detach()

            # Sum desired face regions
            face_mask_source = logits_source_soft[:, 0:1]  # Using first channel for face
            face_mask_target = logits_target_soft[:, 0:1]

            face_mask_source_list.append(face_mask_source)
            face_mask_target_list.append(face_mask_target)

        return {
            **data_dict,
            'face_mask_source': torch.cat(face_mask_source_list, dim=0),
            'face_mask_target': torch.cat(face_mask_target_list, dim=0)
        }

    def _finalize_masks(self, data_dict):
        """Apply final processing to face masks."""
        threshold = 0.6
        
        # Clone original masks
        data_dict['source_mask_modnet'] = data_dict['source_mask'].clone()
        data_dict['target_mask_modnet'] = data_dict['target_mask'].clone()
        
        # Zero out bottom portions
        data_dict['source_mask_modnet'][:, :, -256:] *= 0
        data_dict['target_mask_modnet'][:, :, -256:] *= 0
        
        # Apply threshold and combine masks
        source_mask = (data_dict['face_mask_source'] + 
                    data_dict['source_mask_modnet'] >= threshold).float()
        target_mask = (data_dict['face_mask_target'] + 
                    data_dict['target_mask_modnet'] >= threshold).float()
        
        # Store processed masks
        data_dict.update({
            'source_mask_face_pars_1': source_mask,
            'target_mask_face_pars_1': target_mask,
            'source_mask': (data_dict['source_mask'] * source_mask).float(),
            'target_mask': (data_dict['target_mask'] * target_mask).float()
        })
        
        return data_dict
    


    
    def predict_embed(self, data_dict):
        n = self.num_source_frames
        b = data_dict['source_img'].shape[0] // n
        t = data_dict['target_img'].shape[0] // b

        # with amp.autocast(enabled=self.autocast):
            # Unsqueeze pose embeds for warping gen
        warp_source_embed = self.pose_unsqueeze_nw(data_dict['source_pose_embed']).view(b * n, -1, self.embed_size,
                                                                                        self.embed_size)
        warp_target_embed = self.pose_unsqueeze_nw(data_dict['target_pose_embed']).view(b * t, -1, self.embed_size,
                                                                                        self.embed_size)

        if self.pred_mixing:
            if self.args.detach_warp_mixing_embed:
                warp_mixing_embed = warp_target_embed.detach()
            else:
                warp_mixing_embed = warp_target_embed.clone()
            warp_mixing_embed = warp_mixing_embed.view(b, t, -1, self.embed_size, self.embed_size).roll(1, dims=0)
            rolled_t_emb = data_dict['target_pose_embed'].clone().roll(1, dims=0)
            warp_mixing_embed = warp_mixing_embed.view(b * t, -1, self.embed_size, self.embed_size)

        pose_embeds = [warp_source_embed, warp_target_embed]
        # idt_embeds = [self.warp_idt_s, self.warp_idt_d]
        num_frames = [n, t]
        if self.pred_mixing:
            pose_embeds += [warp_mixing_embed]
            num_frames += [t]
            # idt_embeds+=[self.warp_idt_d]

        warp_embed_dicts = ({}, {}, {})  # source, target, mixing
        embed_dict = {}

        embd = [data_dict['source_pose_embed'], data_dict['target_pose_embed'], rolled_t_emb]
        # Predict warp embeds
        data_dict['idt_embed'] = data_dict['idt_embed']
        for k, (pose_embed, m) in enumerate(zip(pose_embeds, num_frames)):


            if self.args.cat_em:
                warp_embed_orig = self.warp_embed_head_orig_nw(torch.cat([pose_embed, data_dict['idt_embed'].repeat_interleave(m, dim=0)], dim=1))
                warp_embed_orig_d = self.warp_embed_head_orig_nw(torch.cat([pose_embed.detach(), data_dict['idt_embed'].repeat_interleave(m, dim=0)], dim=1))
            else:
                warp_embed_orig = self.warp_embed_head_orig_nw((pose_embed + data_dict['idt_embed'].repeat_interleave(m, dim=0)) * 0.5)
                warp_embed_orig_d = self.warp_embed_head_orig_nw((pose_embed.detach() + data_dict['idt_embed'].repeat_interleave(m, dim=0)) * 0.5)

            c = warp_embed_orig.shape[1]
            warp_embed_dicts[k]['orig'] = warp_embed_orig.view(b * m, c, self.embed_size ** 2)
            warp_embed_dicts[k]['orig_d'] = warp_embed_orig_d.view(b * m, c, self.embed_size ** 2)
            # warp_embed_dicts[k]['ada_v'] = pose_embed.view(b * m, c, self.embed_size ** 2)
            warp_embed_dicts[k]['ada_v'] = embd[k]
            warp_embed_orig_ = warp_embed_orig.view(b * m * c, self.embed_size ** 2)

            if self.args.gen_use_adaconv:
                for name, layer in self.warp_embed_head_dict.items():
                    warp_embed_dicts[k][name] = layer(warp_embed_orig_).view(b * m, c // 2, -1)

        if self.args.gen_use_adanorm or self.args.gen_use_adaconv:
            # Predict embeds
            embed_orig = self.embed_head_orig(data_dict['idt_embed'])

            c = embed_orig.shape[1]
            embed_dict['orig'] = embed_orig.view(b, c, self.embed_size ** 2)
            embed_orig_ = embed_orig.view(b * c, self.embed_size ** 2)
      

        if self.args.gen_use_adaconv:
            for name, layer in self.embed_head_dict.items():
                embed_dict[name] = layer(embed_orig_).view(b, c // 2, -1)
          

        source_warp_embed_dict, target_warp_embed_dict, mixing_warp_embed_dict = warp_embed_dicts

        return source_warp_embed_dict, target_warp_embed_dict, mixing_warp_embed_dict, embed_dict

    def calc_train_losses(self, data_dict: dict, mode: str = 'gen', epoch=0, ffhq_per_b=0, iteration=0):
        return calc_train_losses(self, data_dict=data_dict, mode=mode, epoch=epoch, ffhq_per_b=ffhq_per_b, iteration=iteration)

    def calc_test_losses(self, data_dict: dict, iteration=0):
        return calc_test_losses(self, data_dict, iteration=iteration)

    def prepare_input_data(self, data_dict):
        return prepare_input_data(self, data_dict)


    def forward(self, 
            data_dict: dict,
            phase: str = 'train',
            optimizer_idx: int = 0,
            visualize: bool = False,
            ffhq_per_b: int = 0,
            iteration: int = 0,
            rank: int = -1,
            epoch: int = 0):
        """
        Unified forward pass for both training and inference.
        """
        # Store state
        self.iteration = iteration
        self.epoch = epoch
        self.visualize = visualize
        self.ffhq_per_b = ffhq_per_b
        
        # In training mode, follow the original training pipeline
        if phase == 'train':
            return self._training_forward(
                data_dict, optimizer_idx, visualize, ffhq_per_b, iteration, rank, epoch
            )
        
        # In inference mode, follow the InferenceWrapper pipeline
        else:
            with torch.no_grad():
                memory_stats()
                if 'source_img' in data_dict and 'source_processed' not in data_dict:
                    data_dict = self._process_source_image(data_dict)
                    data_dict['source_processed'] = True
                    
                if 'target_img' in data_dict:
                    data_dict = self._process_target_features(data_dict)
                memory_stats()
            return None, {}, None, data_dict
                
    def _process_target_features(self, data_dict: dict) -> dict:
        """Generate target features and final image."""
        try:
            # Get model dimensions
            c = self.args.latent_volume_channels
            s = self.args.latent_volume_size
            d = self.args.latent_volume_depth
            logger.info(f"\n=== Processing Target Features ===")
            logger.info(f"Volume dimensions - c:{c} d:{d} s:{s}")



            # Before decoder call
            logger.info(f"Before decoder call:")
            # logger.info(f"- target_latent_feats shape: {target_latent_feats.shape}")
            # logger.info(f"- c*d = {c}*{d} = {c*d}")
            driver_img_crop = data_dict['target_img']
            driver_img_mask = data_dict.get('target_mask', torch.ones_like(driver_img_crop[:,:1]))
            batch_size = driver_img_crop.size(0)
            
            # Get driver pose
            with torch.no_grad():
                data_dict['target_theta'] = self.head_pose_regressor.forward(driver_img_crop)
            
            # Create target transformation
            grid = self.identity_grid_3d.repeat_interleave(batch_size, dim=0)
            data_dict['target_rotation_warp'] = grid.bmm(
                data_dict['target_theta'][:, :3].transpose(1, 2)
            ).view(-1, d, s, s, 3)
            
            # Expand source data to match batch size if needed
            if data_dict['source_img'].size(0) != batch_size:
                # Helper function to repeat tensors correctly
                def repeat_tensor(tensor, num_repeats):
                    if tensor is None:
                        return None
                        
                    # Get original shape size
                    orig_shape = tensor.shape
                    # Calculate repeat dims based on original shape
                    repeat_dims = [num_repeats] + [1] * (len(orig_shape) - 1)
                    return tensor.repeat(*repeat_dims)
                    
                # Repeat all source-related tensors
                for key in list(data_dict.keys()):
                    if any(s in key for s in ['source', 'idt', 'canonical']):
                        if torch.is_tensor(data_dict[key]):
                            data_dict[key] = repeat_tensor(data_dict[key], batch_size)
            
            # Generate target embeddings
            data_dict = self.expression_embedder_nw(data_dict, True, False)
            
            # Get warping fields
            _, target_warp_embed_dict, _, embed_dict = self.predict_embed(data_dict)
            target_uv_warp, _ = self.uv_generator_nw(target_warp_embed_dict)
            
            if self.resize_warp:
                target_uv_warp = self.resize_warp_func(target_uv_warp)
            
            # Generate target volume
            aligned_target_volume = self.grid_sample(
                self.grid_sample(data_dict['canonical_volume'], target_uv_warp),
                data_dict['target_rotation_warp']
            )
            
            # Process final features
            target_latent_feats = aligned_target_volume.view(batch_size, c * d, s, s)
            
            # Generate final image
            data_dict['pred_target_img'], _, _, _ = self.decoder_nw(
                data_dict, target_warp_embed_dict, target_latent_feats,
                False, stage_two=True
            )
        except Exception as e:
            logger.error(f"Error in target features: {str(e)}")
            logger.error(traceback.format_exc())
            return data_dict
    # Add background handling
        if 'source_img_mask' not in data_dict:
            # Generate face mask for source if not present
            _, _, source_mask, _ = self.face_idt.forward(data_dict['source_img'])
            data_dict['source_img_mask'] = source_mask
        
        # Get or generate driver mask
        driver_img_mask = data_dict.get('target_mask', None)
        if driver_img_mask is None:
            _, _, driver_mask, _ = self.face_idt.forward(data_dict['target_img'])
            driver_img_mask = driver_mask
            data_dict['target_mask'] = driver_img_mask

        # Rest of the existing processing code...
        
        # After generating pred_target_img, composite with background
        if 'target_img' in data_dict:
            # Get inverse mask for background
            background_mask = 1 - driver_img_mask
            
            # Composite generated face with original background
            data_dict['pred_target_img'] = (
                data_dict['pred_target_img'] * driver_img_mask + 
                data_dict['target_img'] * background_mask
            )
        
        return data_dict
        
    def _process_target_image(self, data_dict: dict) -> dict:
        """Process target/driver image."""
        c = self.args.latent_volume_channels
        s = self.args.latent_volume_size
        d = self.args.latent_volume_depth
        
        driver_img_crop = data_dict['target_img']
        driver_img_mask = data_dict.get('target_mask', torch.ones_like(driver_img_crop[:,:1]))
        
        # Get driver pose
        with torch.no_grad():
            data_dict['target_theta'] = self.head_pose_regressor.forward(driver_img_crop)
        
        # Create target transformation
        grid = self.identity_grid_3d.repeat_interleave(driver_img_crop.size(0), dim=0)
        data_dict['target_rotation_warp'] = grid.bmm(
            data_dict['target_theta'][:, :3].transpose(1, 2)
        ).view(-1, d, s, s, 3)
        
        # Generate target embeddings
        data_dict = self.expression_embedder_nw(data_dict, True, False)
        
        # Get warping fields
        _, target_warp_embed_dict, _, embed_dict = self.predict_embed(data_dict)
        target_uv_warp, _ = self.uv_generator_nw(target_warp_embed_dict)
        
        if self.resize_warp:
            target_uv_warp = self.resize_warp_func(target_uv_warp)
        
        # Generate aligned target volume
        aligned_target_volume = self.grid_sample(
            self.grid_sample(data_dict['canonical_volume'], target_uv_warp),
            data_dict['target_rotation_warp']
        )
        
        # Process final features
        target_latent_feats = aligned_target_volume.view(
            driver_img_crop.size(0), c * d, s, s
        )
        
        # Generate final image
        data_dict['pred_target_img'], _, _, _ = self.decoder_nw(
            data_dict, target_warp_embed_dict, target_latent_feats,
            False, stage_two=True
        )
        
        return data_dict


    def _inference_forward(self, data_dict: dict) -> Tuple[None, dict, None, dict]:
        """
        Forward pass mimicking InferenceWrapper for inference.
        """
        with torch.no_grad():
            # Process source image if it exists
            source_img = data_dict.get('source_img')
            if source_img is not None:
                data_dict = self._process_source_image(data_dict)
                
            # Process driver image if it exists
            driver_img = data_dict.get('target_img')
            if driver_img is not None:
                data_dict = self._process_driver_image(data_dict)
                
            # Return in training format but with None for unused values
            return None, {}, None, data_dict

    def _process_source_image(self, data_dict: dict) -> dict:
        """Process source identity image."""
        c = self.args.latent_volume_channels
        s = self.args.latent_volume_size
        d = self.args.latent_volume_depth
        
        source_img_crop = data_dict['source_img']
        source_img_mask = data_dict.get('source_mask', torch.ones_like(source_img_crop[:,:1]))
        
        # For expression embedding, we need target_img - use source as target initially
        data_dict.update({
            'target_img': source_img_crop.clone(),
            'target_mask': source_img_mask.clone()
        })
        
        # Generate embeddings
        data_dict['idt_embed'] = self.idt_embedder_nw.forward_image(
            source_img_crop * source_img_mask
        )
        
        # Get source pose
        with torch.no_grad():
            data_dict['source_theta'] = self.head_pose_regressor.forward(source_img_crop)
            data_dict['target_theta'] = data_dict['source_theta'].clone()
        
        # Create transformation grid
        grid = self.identity_grid_3d.repeat_interleave(source_img_crop.size(0), dim=0)
        inv_source_theta = data_dict['source_theta'].float().inverse().type(data_dict['source_theta'].type())
        data_dict['source_rotation_warp'] = grid.bmm(
            inv_source_theta[:, :3].transpose(1, 2)
        ).view(-1, d, s, s, 3)
        
        # Process expression embedding
        data_dict = self.expression_embedder_nw(data_dict, True, False)
        
        # Generate source latents and warping
        source_latents = self.local_encoder_nw(source_img_crop * source_img_mask)
        source_warp_embed_dict, _, _, embed_dict = self.predict_embed(data_dict)
        
        xy_gen_outputs = self.xy_generator_nw(source_warp_embed_dict)
        source_xy_warp = xy_gen_outputs[0]
        
        if self.resize_warp:
            source_xy_warp = self.resize_warp_func(source_xy_warp)
        
        # Process source volume
        source_latent_volume = source_latents.view(1, c, d, s, s)
        if self.args.source_volume_num_blocks > 0:
            source_latent_volume = self.volume_source_nw(source_latent_volume)
        
        # Generate canonical volume
        canonical_volume = self.grid_sample(
            self.grid_sample(source_latent_volume, data_dict['source_rotation_warp']),
            source_xy_warp
        )
        data_dict['canonical_volume'] = self.volume_process_nw(canonical_volume, embed_dict)
        
        return data_dict

    def _process_driver_image(self, data_dict: dict) -> dict:
        """
        Process driver image following InferenceWrapper pipeline.
        """
        try:
            c = self.args.latent_volume_channels
            s = self.args.latent_volume_size
            d = self.args.latent_volume_depth
            logger.info(f"\n=== Processing Driver Image ===")
            logger.info(f"Volume dimensions - c:{c} d:{d} s:{s}")

            driver_img_crop = data_dict['target_img']
            driver_img_mask = data_dict.get('target_mask', torch.ones_like(driver_img_crop[:,:1]))
            
            # Get driver pose
            if hasattr(self, 'head_pose_regressor'):
                with torch.no_grad():
                    data_dict['target_theta'] = self.head_pose_regressor.forward(driver_img_crop)
            
            # Create target transformation
            grid = self.identity_grid_3d.repeat_interleave(driver_img_crop.size(0), dim=0)
            data_dict['target_rotation_warp'] = grid.bmm(
                data_dict['target_theta'][:, :3].transpose(1, 2)
            ).view(-1, d, s, s, 3)
            
            # Generate target embeddings
            data_dict = self.expression_embedder_nw(data_dict, True, False)
            
            # Get warping fields
            _, target_warp_embed_dict, _, embed_dict = self.predict_embed(data_dict)
            target_uv_warp, _ = self.uv_generator_nw(target_warp_embed_dict)
            
            if self.resize_warp:
                target_uv_warp = self.resize_warp_func(target_uv_warp)
            
            # Generate target volume
            aligned_target_volume = self.grid_sample(
                self.grid_sample(data_dict['canonical_volume'], target_uv_warp),
                data_dict['target_rotation_warp']
            )
            
            # Generate final features
            target_latent_feats = aligned_target_volume.view(
                driver_img_crop.size(0), c * d, s, s
            )
            
            # Generate final image
            data_dict['pred_target_img'], _, _, _ = self.decoder_nw(
                data_dict, target_warp_embed_dict, target_latent_feats,
                False, stage_two=True
            )
            
            return data_dict
        except Exception as e:
                logger.error(f"Error in driver image: {str(e)}")
                logger.error(traceback.format_exc())
                return data_dict
    def _training_forward(self, data_dict: dict, optimizer_idx: int, 
                        visualize: bool, ffhq_per_b: int,
                        iteration: int, rank: int, epoch: int):
        """Original training forward pass."""
        mode = self.optimizer_idx_to_mode[optimizer_idx]
        
        # Prepare input data
        data_dict = self.prepare_input_data(data_dict)
        
        # Generator forward pass
        if mode == 'gen':
            data_dict = self.G_forward(data_dict, visualize=visualize, 
                                    iteration=iteration, epoch=epoch)
            loss, losses_dict = self.calc_train_losses(
                data_dict, mode='gen', epoch=epoch,
                ffhq_per_b=ffhq_per_b, iteration=iteration
            )
        
        # Discriminator and StyleGAN modes handled by original code
        elif mode == 'dis':
            loss, losses_dict = self._handle_discriminator_mode(
                data_dict, self.resize_func, epoch, ffhq_per_b
            )
        elif mode == 'dis_stylegan':
            loss, losses_dict = self._handle_stylegan_discriminator_mode(
                data_dict, iteration, epoch
            )
            
        # Generate visualizations if requested
        visuals = None
        if visualize:
            data_dict = self.visualize_data(data_dict)
            visuals = self.get_visuals(data_dict)
            
        return loss, losses_dict, visuals, data_dict

    def _handle_generator_mode(self, data_dict, phase, resize, epoch, iteration, ffhq_per_b):
        """Handle generator forward pass for both training and testing phases."""
        data_dict = self.prepare_input_data(data_dict)
        data_dict = self.G_forward(data_dict, visualize=True, iteration=iteration, epoch=epoch)
        
        if phase == 'train':
            return self._handle_generator_training(data_dict, resize, epoch, ffhq_per_b, iteration)
        elif phase == 'test':
            return self._handle_generator_testing(data_dict, iteration)
        else:
            raise ValueError(f"Invalid phase: {phase}")

    def _handle_generator_training(self, data_dict, resize, epoch, ffhq_per_b, iteration):
        """Handle generator training phase."""
        # Set discriminators to eval mode
        self._set_discriminators_eval_mode()
        
        # Process real and fake images through discriminator
        with torch.no_grad():
            _, data_dict['real_feats_gen'] = self.discriminator_ds(resize(data_dict['target_img']))
            if self.args.use_mix_dis:
                _, data_dict['real_feats_gen_2'] = self.discriminator2_ds(data_dict['pred_target_img'].clone().detach())
        
        # Get fake scores and features
        data_dict['fake_score_gen'], data_dict['fake_feats_gen'] = self.discriminator_ds(
            resize(data_dict['pred_target_img']))
        
        if self.args.use_mix_dis:
            data_dict['fake_score_gen_mix'], _ = self.discriminator2_ds(
                resize(data_dict['pred_mixing_img']))
        
        # Calculate losses
        loss, losses_dict = self.calc_train_losses(
            data_dict=data_dict, mode='gen', epoch=epoch,
            ffhq_per_b=ffhq_per_b, iteration=iteration)
        
        # Handle StyleGAN discriminator if enabled
        if self.use_stylegan_d:
            losses_dict = self._handle_stylegan_generator(data_dict, losses_dict, epoch)
        
        return loss, losses_dict

    def _handle_generator_testing(self, data_dict, iteration):
        """Handle generator testing phase."""
        losses_dict, expl_var, expl_var_test = self.calc_test_losses(data_dict, iteration=iteration)
        
        if expl_var is not None:
            data_dict['expl_var'] = expl_var
            data_dict['expl_var_test'] = expl_var_test
        
        return None, losses_dict

    def _handle_discriminator_mode(self, data_dict, resize, epoch, ffhq_per_b):
        """Handle discriminator forward pass."""
        if self.args.detach_dis_inputs:
            data_dict = self._detach_discriminator_inputs(data_dict)
        
        # Set discriminators to training mode
        self._set_discriminators_train_mode()
        
        # Get real and fake scores
        data_dict['real_score_dis'], _ = self.discriminator_ds(resize(data_dict['target_img']))
        data_dict['fake_score_dis'], _ = self.discriminator_ds(resize(data_dict['pred_target_img'].detach()))
        
        if self.args.use_mix_dis:
            data_dict['real_score_dis_mix'], _ = self.discriminator2_ds(resize(data_dict['pred_target_img'].clone().detach()))
            data_dict['fake_score_dis_mix'], _ = self.discriminator2_ds(resize(data_dict['pred_mixing_img'].clone().detach()))
        
        # Calculate losses
        loss, losses_dict = self.calc_train_losses(
            data_dict=data_dict, mode='dis', ffhq_per_b=ffhq_per_b, epoch=epoch)
        
        return loss, losses_dict

    def _handle_stylegan_discriminator_mode(self, data_dict, iteration, epoch):
        """Handle StyleGAN discriminator forward pass."""
        losses_dict = {}
        self._set_stylegan_discriminator_train_mode()
        
        d_regularize = iteration % self.args.d_reg_every == 0
        if d_regularize:
            data_dict['target_img'].requires_grad_()
            data_dict['target_img'].retain_grad()
        
        # Get predictions
        fake_pred = self.stylegan_discriminator_ds((data_dict['pred_target_img'].detach() - 0.5) * 2)
        real_pred = self.stylegan_discriminator_ds((data_dict['target_img'] - 0.5) * 2)
        losses_dict["d_style"] = d_logistic_loss(real_pred, fake_pred)
        
        if self.args.use_mix_losses and epoch >= self.args.mix_losses_start:
            losses_dict = self._handle_stylegan_mixing_losses(data_dict, losses_dict)
        
        if d_regularize:
            losses_dict = self._handle_stylegan_regularization(data_dict, real_pred, losses_dict)
        
        loss = sum(v for v in losses_dict.values() if torch.is_tensor(v))
        return loss, losses_dict

    def _set_discriminators_eval_mode(self):
        """Set all discriminators to evaluation mode."""
        self.discriminator_ds.eval()
        for p in self.discriminator_ds.parameters():
            p.requires_grad = False
        
        if self.args.use_mix_dis:
            self.discriminator2_ds.eval()
            for p in self.discriminator2_ds.parameters():
                p.requires_grad = False
                
        if self.use_stylegan_d:
            self.stylegan_discriminator_ds.eval()
            for p in self.stylegan_discriminator_ds.parameters():
                p.requires_grad = False

    def _set_discriminators_train_mode(self):
        """Set all discriminators to training mode."""
        self.discriminator_ds.train()
        for p in self.discriminator_ds.parameters():
            p.requires_grad = True
        
        if self.args.use_mix_dis:
            self.discriminator2_ds.train()
            for p in self.discriminator2_ds.parameters():
                p.requires_grad = True

    def _detach_discriminator_inputs(self, data_dict):
        """Detach inputs for discriminator training."""
        data_dict['target_img'] = torch.tensor(data_dict['target_img'].detach().clone().data, requires_grad=True)
        data_dict['pred_target_img'] = torch.tensor(data_dict['pred_target_img'].detach().clone().data, requires_grad=True)
        return data_dict

    def _handle_stylegan_mixing_losses(self, data_dict, losses_dict):
        """Calculate StyleGAN mixing losses."""
        fake_pred_mix = self.stylegan_discriminator_ds((data_dict['pred_mixing_img'].detach() - 0.5) * 2)
        fake_loss_mix = F.softplus(fake_pred_mix)
        losses_dict["d_style"] += fake_loss_mix.mean()
        return losses_dict

    def _handle_stylegan_regularization(self, data_dict, real_pred, losses_dict):
        """Calculate StyleGAN regularization losses."""
        r1_penalty = _calc_r1_penalty(data_dict['target_img'], real_pred, scale_number='all')
        data_dict['target_img'].requires_grad_(False)
        losses_dict["r1"] = r1_penalty * self.args.d_reg_every * self.args.r1
        return losses_dict



    def visualize_data(self, data_dict):
        return visualize_data(self, data_dict)

    def get_visuals(self, data_dict):
        return get_visuals(self, data_dict)

    @staticmethod
    def draw_stickman(args, poses):
        return draw_stickman(args, poses)

    def gen_parameters(self):

        params = itertools.chain(*([getattr(self, net).parameters() for net in self.opt_net_names]), [getattr(self, tensor) for tensor in self.opt_tensor_names])

        for param in params:
            yield param

    def configure_optimizers(self):
        self.optimizer_idx_to_mode = {0: 'gen', 1: 'dis', 2: 'dis_stylegan'}

        opts = {
            'adam': lambda param_groups, lr, beta1, beta2: torch.optim.Adam(
                params=param_groups,
                lr=lr,
                betas=(beta1, beta2),
                # eps=1e-8
            ),
            'adamw': lambda param_groups, lr, beta1, beta2: torch.optim.AdamW(
                params=param_groups,
                lr=lr,
                betas=(beta1, beta2))}

        opt_gen = opts[self.args.gen_opt_type](
            self.gen_parameters(),
            self.args.gen_lr,
            self.args.gen_beta1,
            self.args.gen_beta2)

        if self.args.use_mix_dis:
            opt_dis = opts[self.args.dis_opt_type](
                itertools.chain(self.discriminator_ds.parameters(), self.discriminator2_ds.parameters()),
                self.args.dis_lr,
                self.args.dis_beta1,
                self.args.dis_beta2,
            )
        else:
            opt_dis = opts[self.args.dis_opt_type](
                self.discriminator_ds.parameters(),
                self.args.dis_lr,
                self.args.dis_beta1,
                self.args.dis_beta2,
            )

        if self.use_stylegan_d:
            d_reg_ratio = self.args.d_reg_every / (self.args.d_reg_every + 1)
            opt_dis_style = opts['adam'](
                self.stylegan_discriminator_ds.parameters(),
                self.args.dis_stylegan_lr * d_reg_ratio,
                0 ** d_reg_ratio,
                0.99 ** d_reg_ratio,
            )
            return [opt_gen, opt_dis, opt_dis_style]
        else:
            return [opt_gen, opt_dis]

    def configure_schedulers(self, opts, epochs=None, steps_per_epoch=None):
        shds = {
            'step': lambda optimizer, lr_max, lr_min, max_iters, epochs,
                           steps_per_epoch: torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=max_iters,
                gamma=lr_max / lr_min),

            'cosine': lambda optimizer, lr_max, lr_min, max_iters, epochs,
                             steps_per_epoch: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=max_iters,
                eta_min=lr_min),
            'onecycle': lambda optimizer, lr_max, lr_min, max_iters, epochs,
                               steps_per_epoch: torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=lr_max,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1
            )}

        shd_gen = shds[self.args.gen_shd_type](
            opts[0],
            self.args.gen_lr,
            self.args.gen_shd_lr_min,
            self.args.gen_shd_max_iters,
            epochs,
            steps_per_epoch)

        shd_dis = shds[self.args.dis_shd_type](
            opts[1],
            self.args.dis_lr,
            self.args.dis_shd_lr_min,
            self.args.dis_shd_max_iters,
            epochs,
            steps_per_epoch
        )

        if self.use_stylegan_d:
            shd_dis_stylegan = shds[self.args.dis_shd_type](
                opts[2],
                self.args.dis_stylegan_lr,
                self.args.dis_shd_lr_min,
                self.args.dis_shd_max_iters,
                epochs,
                steps_per_epoch
            )

            return [shd_gen, shd_dis, shd_dis_stylegan], [self.args.gen_shd_max_iters, self.args.dis_shd_max_iters,
                                                          self.args.dis_shd_max_iters]
        else:
            return [shd_gen, shd_dis], [self.args.gen_shd_max_iters, self.args.dis_shd_max_iters]

    def generate_frames_from_motion(self, motion_outputs: Dict[str, torch.Tensor], 
                                   source_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate frames from motion outputs for loss computation.
        
        Args:
            motion_outputs: Dictionary containing motion parameters from VASA model
                - theta: [B, T, 3, 4] or [B, T, 4, 4] pose matrices
                - expression_embed: [B, T, D] expression embeddings
                - scale: [B, T, 1] scale values  
                - rotation: [B, T, 3] rotation values
                - translation: [B, T, 3] translation values
            source_params: Dictionary containing source image parameters
                - source_img: [B, C, H, W] source images
                - idt_embed: [B, D] identity embeddings (optional)
                
        Returns:
            Generated frames tensor [B, T, C, H, W]
        """
        try:
            B, T = motion_outputs['theta'].shape[:2]
            device = motion_outputs['theta'].device
            generated_frames = []
            
            # Get face mask for source images
            source_mask = self.face_idt.forward(source_params['source_img'])[0]
            source_mask = (source_mask > 0.6).float()
            source_masked = source_params['source_img'] * source_mask
            
            # Get identity embedding if not provided
            if 'idt_embed' not in source_params:
                idt_embed = self.idt_embedder_nw(source_masked)
            else:
                idt_embed = source_params['idt_embed']
            
            # Extract source latents and create canonical volume
            source_latents = self.local_encoder_nw(source_masked)
            c = self.args.latent_volume_channels
            d = self.args.latent_volume_depth
            s = self.args.latent_volume_size
            
            source_volume = source_latents.view(B, c, d, s, s)
            if self.args.source_volume_num_blocks > 0:
                source_volume = self.volume_source_nw(source_volume)
            
            # Process each batch item
            for b in range(B):
                batch_frames = []
                
                # Get canonical volume for this batch item
                canonical_volume_b = self.volume_process_nw(source_volume[b:b+1])
                
                for t in range(T):
                    # Get motion parameters for current frame
                    curr_expression = motion_outputs['expression_embed'][b:b+1, t]
                    curr_theta = motion_outputs['theta'][b:b+1, t]
                    
                    # Convert theta to correct format if needed (remove last row if 4x4)
                    if curr_theta.shape[-2] == 4:
                        curr_theta = curr_theta[:, :3, :]
                    
                    # Create data dict for current frame
                    data_dict = {
                        'source_img': source_params['source_img'][b:b+1],
                        'target_img': source_params['source_img'][b:b+1],
                        'source_mask': source_mask[b:b+1],
                        'target_mask': source_mask[b:b+1],
                        'source_theta': curr_theta,
                        'target_theta': curr_theta,
                        'idt_embed': idt_embed[b:b+1],
                        'source_pose_embed': curr_expression,
                        'target_pose_embed': curr_expression
                    }
                    
                    # Generate embeddings
                    source_warp_embed_dict, target_warp_embed_dict, _, embed_dict = self.predict_embed(data_dict)
                    
                    # Generate UV warp
                    target_uv_warp, _ = self.uv_generator_nw(target_warp_embed_dict)
                    
                    # Handle resizing if needed
                    if self.resize_warp:
                        stride = self.warp_resize_stride
                        target_uv_warp = F.avg_pool3d(target_uv_warp.permute(0, 4, 1, 2, 3), 
                                                     kernel_size=stride,
                                                     stride=stride).permute(0, 2, 3, 4, 1)
                    
                    # Create identity grid for rotation
                    grid = self.identity_grid_3d.repeat_interleave(1, dim=0)
                    target_rotation_warp = grid.bmm(curr_theta.transpose(1, 2)).view(-1, d, s, s, 3)
                    
                    # Apply warps to canonical volume
                    aligned_target_volume = self.grid_sample(
                        self.grid_sample(canonical_volume_b, target_uv_warp),
                        target_rotation_warp
                    )
                    
                    target_latent_feats = aligned_target_volume.view(1, c * d, s, s)
                    
                    # Generate frame through decoder
                    frame, _, _, _ = self.decoder_nw(
                        data_dict,
                        embed_dict,
                        target_latent_feats,
                        False,
                        stage_two=True
                    )
                    
                    batch_frames.append(frame)
                
                # Stack frames for this batch item [T, C, H, W]
                batch_frames = torch.cat(batch_frames, dim=0)
                generated_frames.append(batch_frames)
            
            # Stack all batch items [B, T, C, H, W]
            generated_frames = torch.stack(generated_frames, dim=0)
            return generated_frames
            
        except Exception as e:
            logger.error(f"Error generating frames from motion outputs: {str(e)}")
            logger.error(traceback.format_exc())
            logger.error(f"Motion outputs keys: {motion_outputs.keys()}")
            logger.error(f"Source params keys: {source_params.keys()}")
            raise