import torch
from torch import nn
from torch import optim
from torch.cuda import amp
import torch.nn.functional as F
import math
import numpy as np
import itertools
import copy
from torch.cuda import amp
from scipy import linalg
from . import utils
from .utils import ProjectorConv, ProjectorNorm, ProjectorNormLinear, assign_adaptive_conv_params, \
    assign_adaptive_norm_params, Upsample_sg2
import utils.args as args_utils
from dataclasses import dataclass
# from .sg3_generator import Generator
import traceback
from logger import logger

class Decoder(nn.Module):

    @dataclass
    class Config:
        eps : float
        image_size : int
        gen_embed_size : int 
        gen_adaptive_kernel : bool
        gen_adaptive_conv_type: str
        gen_latent_texture_size: int
        in_channels: int
        gen_num_channels: int
        dec_max_channels: int
        gen_use_adanorm: bool
        gen_activation_type: str
        gen_use_adaconv: bool
        dec_channel_mult: float
        dec_num_blocks: int
        dec_up_block_type: str
        dec_pred_seg: bool
        dec_seg_channel_mult: float
        num_gpus: int
        norm_layer_type: str
        bigger: bool = False
        vol_render: bool = False
        im_dec_num_lrs_per_resolution: int = 1
        im_dec_ch_div_factor: float = 2.0
        emb_v_exp: bool = False
        dec_use_sg3_img_dec: bool = False
        no_detach_frec: int = 10
        dec_key_emb: str = 'orig'

    def __init__(self, cfg:Config):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.adaptive_conv_type = self.cfg.gen_adaptive_conv_type
        num_blocks = self.cfg.dec_num_blocks
        num_up_blocks = int(math.log(self.cfg.image_size // self.cfg.gen_latent_texture_size, 2))
        self.in_channels = self.cfg.in_channels
        out_channels = min(int(self.cfg.gen_num_channels * self.cfg.dec_channel_mult * 2**num_up_blocks), self.cfg.dec_max_channels)
        # logger.debug(num_up_blocks, out_channels)
        self.gen_max_channels = self.cfg.dec_max_channels
        self.num_gpus = self.cfg.num_gpus
        self.norm_layer_type = self.cfg.norm_layer_type
        norm_layer_type = self.cfg.norm_layer_type

        if norm_layer_type == 'bn':
            if self.num_gpus > 1:
                norm_layer_type = 'sync_' + norm_layer_type
        if self.cfg.gen_use_adanorm:
            norm_layer_type = 'ada_' + norm_layer_type

        # logger.debug(norm_layer_type)
        if self.cfg.vol_render:
            layers = []
        else:
            layers = [
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    bias=False)
            ]

        for i in range(num_blocks):
            layers += [
                utils.blocks['res'](
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_layer_type=norm_layer_type,
                    activation_type=self.cfg.gen_activation_type,
                    conv_layer_type=('ada_' if self.cfg.gen_use_adaconv else '') + 'conv')]


        self.res_decoder = nn.Sequential(*layers)

        if self.cfg.dec_use_sg3_img_dec:
            self.img_decoder = Generator(
                512,
                512,
                512,
                3,
                num_layers=14,  # Total number of layers, excluding Fourier features and ToRGB. # NOTE Original 6
                num_critical=2,  # Number of critically sampled layers at the end.
                first_cutoff=10.079,  # Cutoff frequency of the first layer (f_{c,0}). # NOTE Original 2.0
                first_stopband=17.959,  # Minimum stopband of the first layer (f_{t,0}). # NOTE Original 2**3.1
                last_stopband_rel=2**0.3,  # Minimum stopband of the last layer, expressed relative to the cutoff.
                margin_size=16,  # Number of additional pixels outside the image.
                num_fp16_res=False,  # Use FP16 for the N highest resolutions.
            )
        else:
            self.img_decoder = ImageDecoder(
                self.cfg.image_size,
                self.cfg.gen_latent_texture_size,
                self.cfg.gen_use_adanorm,
                self.cfg.gen_num_channels,
                self.cfg.dec_up_block_type,
                self.cfg.gen_activation_type,
                self.cfg.gen_use_adaconv,
                self.cfg.dec_pred_seg,
                self.cfg.dec_seg_channel_mult,
                out_channels,
                self.num_gpus,
                norm_layer_type=norm_layer_type,
                bigger=self.cfg.bigger,
                im_dec_num_lrs_per_resolution = self.cfg.im_dec_num_lrs_per_resolution,
                im_dec_ch_div_factor = self.cfg.im_dec_ch_div_factor
                )	


        self.gen_use_adanorm = self.cfg.gen_use_adanorm
        if self.cfg.gen_use_adanorm:
            self.projector = ProjectorNormLinear(net_or_nets=[self.res_decoder, self.img_decoder], eps=self.cfg.eps,
                                           gen_embed_size=self.cfg.gen_embed_size,
                                           gen_max_channels=self.gen_max_channels, emb_v_exp = self.cfg.emb_v_exp, no_detach_frec=self.cfg.no_detach_frec, key_emb=self.cfg.dec_key_emb)
        else:
            self.projector = ProjectorNorm(net_or_nets=[self.res_decoder, self.img_decoder], eps=self.cfg.eps,
                                           gen_embed_size=self.cfg.gen_embed_size,
                                           gen_max_channels=self.gen_max_channels)

        logger.debug(f"{sum(p.numel() for p in self.res_decoder.parameters() if p.requires_grad), sum(p.numel() for p in self.img_decoder.parameters() if p.requires_grad)}")
        if self.cfg.gen_use_adaconv:
            self.projector_conv = ProjectorConv(net_or_nets=[self.res_decoder, self.img_decoder], eps=self.cfg.eps,
                                                gen_adaptive_kernel=self.cfg.gen_adaptive_kernel,
                                                gen_max_channels=self.gen_max_channels)

    def forward(self, data_dict, embed_dict, feat_2d, input_flip_feat=False, annealing_alpha=0.0, embed=None, stage_two=False, iteration=0):
        logger.debug("\n=== Decoder Forward Pass Debug ===")
        logger.debug(f"Initial feat_2d shape: {feat_2d.shape}")
        
        if self.gen_use_adanorm:
            logger.debug("Using AdaNorm")
            params_norm = self.projector(embed_dict, iter=iteration)
            annealing_alpha = 1
        else:
            logger.debug("Using regular norm")
            params_norm = self.projector(embed_dict, iter=iteration)
        
        if input_flip_feat:
            logger.debug("Processing flipped features")
            params_norm_ = []
            for param in params_norm:
                if isinstance(param, tuple):
                    logger.debug(f"Tuple param shapes: {[p.shape for p in param]}")
                    params_norm_.append((torch.cat([p] * 2) for p in param))
                else:
                    logger.debug(f"Single param shape: {param.shape}")
                    params_norm_.append(torch.cat([param] * 2))
        else:
            params_norm_ = params_norm

        assign_adaptive_norm_params([self.res_decoder, self.img_decoder], params_norm_, annealing_alpha)

        if hasattr(self, 'projector_conv'):
            logger.debug("Processing adaptive convolution parameters")
            params_conv = self.projector_conv(embed_dict)

            if input_flip_feat:
                params_conv_ = []
                for param in params_conv:
                    if isinstance(param, tuple):
                        logger.debug(f"Conv tuple param shapes: {[p.shape for p in param]}")
                        params_conv_.append((torch.cat([p] * 2) for p in param))
                    else:
                        logger.debug(f"Conv single param shape: {param.shape}")
                        params_conv_.append(torch.cat([param] * 2))
            else:
                params_conv_ = params_conv

            assign_adaptive_conv_params([self.res_decoder, self.img_decoder], params_conv, self.adaptive_conv_type, annealing_alpha)

        logger.debug(f"\nPre res_decoder feat_2d shape: {feat_2d.shape}")
        feat_2d = self.res_decoder(feat_2d)
        logger.debug(f"Post res_decoder feat_2d shape: {feat_2d.shape}")

        logger.debug("\nCalling img_decoder")
        img, seg, img_f = self.img_decoder(feat_2d, stage_two=stage_two)
        logger.debug(f"img_decoder outputs - img: {img.shape if img is not None else None}, "
            f"seg: {seg.shape if seg is not None else None}, "
            f"img_f: {img_f.shape if img_f is not None else None}")

        if hasattr(self, 'conf_decoder') and self.training and input_flip_feat:
            logger.debug("\nProcessing confidence decoder")
            feat, feat_flip = feat_2d.split(feat_2d.shape[0] // 2)
            logger.debug(f"Split shapes - feat: {feat.shape}, feat_flip: {feat_flip.shape}")

            conf_ms, conf_ms_flip, conf, conf_flip = self.conf_decoder(feat, feat_flip)
            
            logger.debug("Processing confidence measurements")
            for i, (conf_ms_k, conf_ms_flip_k, conf_name) in enumerate(zip(conf_ms, conf_ms_flip, self.conf_ms_names)):
                logger.debug(f"\nConfidence {i} shapes:")
                logger.debug(f"conf_ms_k shapes: {[k.shape for k in conf_ms_k]}")
                logger.debug(f"conf_ms_flip_k shapes: {[k.shape for k in conf_ms_flip_k]}")
                
                data_dict[f'{conf_name}_ms'] = conf_ms_k
                data_dict[f'{conf_name}_flip_ms'] = conf_ms_flip_k
                data_dict[conf_name] = conf_ms_k[0]
                data_dict[f'{conf_name}_flip'] = conf_ms_flip_k[0]

            for i, (conf_k, conf_flip_k, conf_name) in enumerate(zip(conf, conf_flip, self.conf_names)):
                logger.debug(f"\nBase confidence {i} shapes:")
                logger.debug(f"conf_k shape: {conf_k.shape}")
                logger.debug(f"conf_flip_k shape: {conf_flip_k.shape}")
                
                data_dict[f'{conf_name}'] = conf_k
                data_dict[f'{conf_name}_flip'] = conf_flip_k

        logger.debug("\nFinal return values:")
        if stage_two:
            logger.debug("Stage two mode - returning img, seg, feat_2d, img_f")
            return img, seg, feat_2d, img_f
        else:
            logger.debug("Normal mode - returning img, seg, None, None") 
            return img, seg, None, None
                
    

    def forward_simple(
        self,
        volume_frame: torch.Tensor,  # [1, 96, 16, 64, 64]
        idt_embed: torch.Tensor,     # [1, 512, 4, 4]
    ) -> torch.Tensor:
        """
        Simplified forward pass taking volume frame and identity embedding.
        Handles reshaping internally.
        
        Args:
            volume_frame: Single frame of shape [B, C, D, H, W] 
                        where C=96, D=16, H=64, W=64
            idt_embed: Identity embedding of shape [B, 512, 4, 4]
        
        Returns:
            Generated image of shape [B, 3, 512, 512]
        """
        try:
            # logger.debug("\n=== Simple Decoder Forward ===")
            # logger.debug(f"Input shapes:")
            # logger.debug(f"- volume_frame: {volume_frame.shape}")
            # logger.debug(f"- idt_embed: {idt_embed.shape}")

            # Reshape volume to feat_2d format
            B, C, D, H, W = volume_frame.shape
            feat_2d = volume_frame.reshape(B, C * D, H, W)  # [1, 1536, 64, 64]
            # logger.debug(f"Reshaped to feat_2d: {feat_2d.shape}")

            # Create minimal required dicts
            data_dict = {'idt_embed': idt_embed}
            embed_dict = {}
            
            # Get norm parameters
            if self.gen_use_adanorm:
                params_norm = self.projector(embed_dict)
                annealing_alpha = 1
            else:
                params_norm = self.projector(embed_dict)
                
            # Apply parameters
            assign_adaptive_norm_params([self.res_decoder, self.img_decoder], params_norm, 0.0)
            
            # Process through decoders
            feat_2d = self.res_decoder(feat_2d)
            # logger.debug(f"After res_decoder: {feat_2d.shape}")
            
            img, _, img_f = self.img_decoder(feat_2d, stage_two=True)
            # logger.debug(f"Final image shape: {img.shape}")

            return img

        except Exception as e:
            logger.error("\nError in simple decoder forward:")
            logger.error(f"Error: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
class ImageDecoder(nn.Module):
    def __init__(self,
                 image_size,
                 gen_latent_texture_size,
                 gen_use_adanorm,
                 gen_num_channels,
                 dec_up_block_type,
                 gen_activation_type,
                 gen_use_adaconv,
                 dec_pred_seg,
                 dec_seg_channel_mult,
                 shared_in_channels,
                 num_gpus,
                 norm_layer_type,
                 bigger=False,
                 im_dec_num_lrs_per_resolution=1,
                 im_dec_ch_div_factor = 2
                 ):
        super(ImageDecoder, self).__init__()
        num_up_blocks = int(math.log(image_size // gen_latent_texture_size, 2))
        out_channels = shared_in_channels
        self.bigger = bigger
        self.im_dec_num_lrs_per_resolution = im_dec_num_lrs_per_resolution
        self.num_gpus =num_gpus
        if norm_layer_type == 'bn':
            if self.num_gpus > 1:
                norm_layer_type = 'sync_' + norm_layer_type


        layers = []

        if self.bigger:
            num_up_blocks = num_up_blocks - 1

        for i in range(num_up_blocks):
            in_channels = out_channels
            # out_channels = max(out_channels // 2, gen_num_channels)
            out_channels = max(int(out_channels / im_dec_ch_div_factor/32)*32, gen_num_channels)


            if self.bigger:
                out_channels = max(out_channels, 256)
            # out_channels = max(out_channels, gen_num_channels)

            # if out_channels%32!=0:
            #     c_norm_layer_type = 'gn_24'
            # else:
            #     c_norm_layer_type = norm_layer_type
            k=0
            for _ in range(self.im_dec_num_lrs_per_resolution):
                layers += [
                    utils.blocks[dec_up_block_type](
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2 if k==0 else 1,
                        norm_layer_type=norm_layer_type,
                        activation_type=gen_activation_type,
                        conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv',
                        resize_layer_type='nearest' if k==0 else 'none'),
                    
                    # utils.blocks[dec_up_block_type](
                    #     in_channels=out_channels*2,
                    #     out_channels=out_channels,
                    #     norm_layer_type=norm_layer_type,
                    #     activation_type=gen_activation_type,
                    #     conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),
                ]
                in_channels = out_channels
                k+=1

        if self.bigger:
            layers += [
                # utils.blocks[dec_up_block_type](
                #     in_channels=out_channels,
                #     out_channels=out_channels//2,
                #     norm_layer_type=norm_layer_type,
                #     activation_type=gen_activation_type,
                #     conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),



                utils.blocks[dec_up_block_type](
                    in_channels=out_channels,
                    out_channels=out_channels//2,
                    norm_layer_type=norm_layer_type,
                    activation_type=gen_activation_type,
                    conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),

                utils.blocks[dec_up_block_type](
                    in_channels=out_channels//2,
                    out_channels=out_channels//2,
                    stride=2,
                    norm_layer_type=norm_layer_type,
                    activation_type=gen_activation_type,
                    conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv',

                    resize_layer_type='nearest'),
                utils.blocks[dec_up_block_type](
                    in_channels=out_channels // 2,
                    out_channels=out_channels // 4,
                    norm_layer_type=norm_layer_type,
                    activation_type=gen_activation_type,
                    conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),

            ]
            out_channels = out_channels // 4

        self.dec_img_blocks = nn.Sequential(*layers)

        layers = [
            utils.norm_layers[norm_layer_type](out_channels),
            utils.activations[gen_activation_type](inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=3,
                kernel_size=1),
            nn.Sigmoid()]

        self.dec_img_head = nn.Sequential(*layers)

        if dec_pred_seg:
            in_channels = shared_in_channels
            out_channels = int(gen_num_channels * dec_seg_channel_mult * 2**num_up_blocks)

            layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False)]

            for i in range(num_up_blocks):
                in_channels = out_channels
                out_channels = max(out_channels // 2, int(gen_num_channels * dec_seg_channel_mult))
                layers += [
                    utils.blocks[dec_up_block_type](
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2,
                        norm_layer_type=norm_layer_type,
                        activation_type=gen_activation_type,
                        conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv',
                        resize_layer_type='nearest')]


    def forward(self, feat, stage_two=False):
        """
        Forward pass with detailed shape logging
        Args:
            feat: Input features
            stage_two: Whether in stage two of training
        """
        try:
            # logger.debug("\n=== ImageDecoder Forward Pass ===")
            # logger.debug(f"Input feat shape: {feat.shape}")

            # Process through decoder blocks
            img_feat = self.dec_img_blocks(feat)
            # logger.debug(f"After dec_img_blocks shape: {img_feat.shape}")

            # Generate final image
            img = self.dec_img_head(img_feat.float())
            # logger.debug(f"Final output image shape: {img.shape}")

            if stage_two:
                # logger.debug("Returning in stage_two mode")
                return img, None, img_feat
            else:
                # logger.debug("Returning in normal mode")
                return img, None, None

        except Exception as e:
            logger.error("\nError in ImageDecoder forward pass:")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Last known tensor shape - feat: {feat.shape if feat is not None else 'None'}")
            logger.error(f"Last known tensor shape - img_feat: {img_feat.shape if 'img_feat' in locals() else 'None'}")
            logger.error(f"Last known tensor shape - img: {img.shape if 'img' in locals() else 'None'}")
            logger.error(traceback.format_exc())
            raise

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))
    return img

class ImageDecoder_SG2(nn.Module):
    def __init__(self,
                 image_size,
                 gen_latent_texture_size,
                 gen_use_adanorm,
                 gen_num_channels,
                 dec_up_block_type,
                 gen_activation_type,
                 gen_use_adaconv,
                 dec_pred_seg,
                 dec_seg_channel_mult,
                 shared_in_channels,
                 num_gpus,
                 norm_layer_type):
        super(ImageDecoder_SG2, self).__init__()
        self.num_up_blocks = int(math.log(image_size // gen_latent_texture_size, 2))
        out_channels = shared_in_channels

        self.to_rgbs = nn.ModuleList()
        self.num_gpus =num_gpus
        if norm_layer_type == 'bn':
            if self.num_gpus > 1:
                norm_layer_type = 'sync_' + norm_layer_type

        self.upsample = Upsample_sg2(kernel=[1, 3, 3, 1])
        # self.upsample = lambda inputs: F.interpolate(inputs, scale_factor=2, mode='bicubic')
        self.blocks = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        layers = [
            utils.norm_layers[norm_layer_type](out_channels),
            utils.activations[gen_activation_type](inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=3,
                kernel_size=1),
            # nn.Sigmoid()
            # utils.norm_layers[norm_layer_type](out_channels//4),
            # utils.activations[gen_activation_type](inplace=True),
            # nn.Conv2d(
            #     in_channels=out_channels // 4,
            #     out_channels=3,
            #     kernel_size=1),

        ]
        self.to_rgb1 = nn.Sequential(*layers)
        # min_list = [192, 96, 48, 24]  #128, 256, 512, 1024
        # min_list = [256, 128, 64]  # 128, 256, 512, 1024
        for i in range(self.num_up_blocks):
            in_channels = out_channels
            # out_channels = max(out_channels // 2, min_list[i])
            out_channels = max(out_channels // 2, gen_num_channels)
            layers = [
                utils.blocks[dec_up_block_type](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    norm_layer_type=norm_layer_type,
                    activation_type=gen_activation_type,
                    conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv',
                    resize_layer_type='blur'),

                # utils.blocks[dec_up_block_type](
                #     in_channels=out_channels,
                #     out_channels=out_channels,
                #     norm_layer_type=norm_layer_type,
                #     activation_type=gen_activation_type,
                #     conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),
            ]


            self.blocks.append(nn.Sequential(*layers))

            layers = [
                utils.norm_layers[norm_layer_type](out_channels),
                # utils.norm_layers['bn'](out_channels),
                utils.activations[gen_activation_type](inplace=True),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=3,
                    kernel_size=1),
                # nn.Sigmoid()

            ]

            self.to_rgbs.append(nn.Sequential(*layers))



    def forward(self, feat, ):
        images = []
        img = self.to_rgb1(feat.float())
        # logger.debug(img.shape)
        images.append(img)

        for i in range(self.num_up_blocks):
            feat = self.blocks[i](feat).float()
            img = self.to_rgbs[i](feat)
            images.append(img)

        k=1
        for i in images[-2::-1]:
            skip = self.upsample(i)
            for j in range(k-1):
                # logger.debug('s', skip.shape)
                skip = self.upsample(skip)
            img = img + skip
            # logger.debug(i.shape, skip.shape, img.shape)
            k+=1

        img = self.sigmoid(img)
        # img = norm_ip(img, 0, 1)
        # img.clamp_(max=1, min=0)

        return img, None




class ConfDecoder(nn.Module):
    def __init__(self,
                 gen_latent_texture_size,
                 dec_conf_ms_scales,
                 image_size,
                 gen_num_channels,
                 dec_conf_channel_mult,
                 gen_max_channels,
                 gen_activation_type,
                 gen_downsampling_type,
                 shared_in_channels,
                 num_branches_ms,
                 num_branches,
                 num_gpus,
                 norm_layer_type):
        super(ConfDecoder, self).__init__()
        self.num_branches_ms = num_branches_ms
        self.num_branches = num_branches
        self.input_spatial_size = gen_latent_texture_size
        self.num_gpus = num_gpus
        num_down_blocks = int(math.log(gen_latent_texture_size * 2**(dec_conf_ms_scales - 1) // image_size, 2))
        num_up_blocks = int(math.log(image_size // gen_latent_texture_size, 2))
        out_channels = int(gen_num_channels * dec_conf_channel_mult * 2**num_up_blocks)

        self.first_layer = nn.Conv2d(shared_in_channels, out_channels, kernel_size=(1, 1), bias=False)

        shared_in_channels = out_channels

        if norm_layer_type == 'bn':
            if self.num_gpus > 1:
                norm_layer_type = 'sync_' + norm_layer_type
        self.conf_down_blocks = nn.ModuleList()

        for i in range(num_down_blocks):
            in_channels = out_channels
            out_channels = min(in_channels * 2, gen_max_channels)

            self.conf_down_blocks += [nn.ModuleList()]

            for k in range(self.num_branches_ms):
                self.conf_down_blocks[-1] += [
                    utils.blocks['conv'](
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2,
                        norm_layer_type=norm_layer_type,
                        # norm_layer_type='bn',
                        activation_type=gen_activation_type,
                        resize_layer_type=gen_downsampling_type)]

        out_channels = shared_in_channels

        self.conf_up_blocks = nn.ModuleList()

        for i in range(num_up_blocks - 1, -1, -1):
            in_channels = out_channels
            out_channels = min(int(gen_num_channels * dec_conf_channel_mult * 2**i), gen_max_channels)

            self.conf_up_blocks += [nn.ModuleList()]

            for k in range(self.num_branches_ms +  self.num_branches):
                self.conf_up_blocks[-1] += [
                    utils.blocks['conv'](
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2,
                        norm_layer_type=norm_layer_type,
                        # norm_layer_type='bn',
                        activation_type=gen_activation_type,
                        resize_layer_type='nearest')]

        out_channels = shared_in_channels

        self.conf_same_blocks = nn.ModuleList()

        for k in range(self.num_branches_ms):
            self.conf_same_blocks += [utils.blocks['conv'](
                in_channels=out_channels,
                out_channels=out_channels,
                # norm_layer_type='bn' ,
                norm_layer_type=norm_layer_type,
                activation_type=gen_activation_type)]

        self.conf_head_ms = nn.ModuleDict()
        self.conf_head = nn.ModuleList()
        spatial_size = image_size // 2**dec_conf_ms_scales

        for i in range(dec_conf_ms_scales - 1, -1, -1):
            out_channels = min(int(gen_num_channels * dec_conf_channel_mult * 2**i), gen_max_channels)
            spatial_size *= 2

            self.conf_head_ms['%04d' % spatial_size] = nn.ModuleList()

            for k in range(self.num_branches_ms):
                self.conf_head_ms['%04d' % spatial_size] += [
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=2, # one for "straight" texture, one for flip
                        kernel_size=(1, 1))]

            if spatial_size == image_size:
                for k in range(self.num_branches):
                    self.conf_head += [
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=2,
                            kernel_size=(1, 1))]

    def forward(self, feat, feat_flip):

        assert feat.shape[2] == self.input_spatial_size
        inputs = self.first_layer(torch.cat([feat, feat_flip])) # concat in case batch norm is used
        ms_feat = {}

        # Process feat at the same resolution
        spatial_size = self.input_spatial_size
        ms_feat[spatial_size] = []
        for block in self.conf_same_blocks:
            outputs = block(inputs)
            ms_feat[spatial_size] += [outputs.split(feat.shape[0])]

        # Downsample feat
        outputs = [inputs] * self.num_branches_ms
        for blocks in self.conf_down_blocks:
            spatial_size //= 2
            ms_feat[spatial_size] = []

            for k, (block, output) in enumerate(zip(blocks, outputs)):
                output = block(output)
                outputs[k] = output
                ms_feat[spatial_size] += [output.split(feat.shape[0])]

        # Upsample feat
        spatial_size = self.input_spatial_size
        outputs = [inputs] * self.num_branches_ms + [inputs] * self.num_branches
        for blocks in self.conf_up_blocks:
            spatial_size *= 2
            ms_feat[spatial_size] = []

            for k, (block, output) in enumerate(zip(blocks, outputs)):
                output = block(output)
                outputs[k] = output
                ms_feat[spatial_size] += [output.split(feat.shape[0])]

        # Predict confidences
        conf_ms, conf_ms_flip = [[] for k in range(self.num_branches_ms)], [[] for k in range(self.num_branches_ms)]

        for spatial_size in sorted(self.conf_head_ms.keys())[::-1]: # from highest res to lowest res
            for k, ((feat_, feat_flip_), head) in enumerate(zip(ms_feat[int(spatial_size)][:self.num_branches_ms], self.conf_head_ms[spatial_size])):
                conf_ms[k] += [F.softplus(head(feat_.float())[:, :1]).clamp(0.1, 10)]
                conf_ms_flip[k] += [F.softplus(head(feat_flip_.float())[:, 1:]).clamp(0.1, 10)]

        conf, conf_flip = [], []

        spatial_size = max(ms_feat.keys())
        for head, (feat_, feat_flip_) in zip(self.conf_head, ms_feat[spatial_size][self.num_branches_ms:]):
            conf += [F.softplus(head(feat_.float())[:, :1]).clamp(0.1, 10)]
            conf_flip += [F.softplus(head(feat_flip_.float())[:, 1:]).clamp(0.1, 10)]

        return conf_ms, conf_ms_flip, conf, conf_flip


class ImageDecoder_stage2(nn.Module):
    def __init__(self,
                 image_size,
                 gen_latent_texture_size,
                 gen_use_adanorm,
                 gen_num_channels,
                 dec_up_block_type,
                 gen_activation_type,
                 gen_use_adaconv,
                 dec_pred_seg,
                 dec_seg_channel_mult,
                 shared_in_channels,
                 num_gpus,
                 norm_layer_type):
        super(ImageDecoder_stage2, self).__init__()

        num_up_blocks = int(math.log(image_size // gen_latent_texture_size, 2))
        out_channels = shared_in_channels

        self.num_gpus =num_gpus
        if norm_layer_type == 'bn':
            if self.num_gpus > 1:
                norm_layer_type = 'sync_' + norm_layer_type


        layers = []

        for i in range(num_up_blocks):
            in_channels = out_channels
            out_channels = max(out_channels // 2, gen_num_channels)
            # out_channels = max(out_channels, gen_num_channels)

            # if out_channels%32!=0:
            #     c_norm_layer_type = 'gn_24'
            # else:
            #     c_norm_layer_type = norm_layer_type

            layers += [
                utils.blocks[dec_up_block_type](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    norm_layer_type=norm_layer_type,
                    activation_type=gen_activation_type,
                    conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv',
                    resize_layer_type='nearest'),
            ]


        self.dec_img_blocks = nn.Sequential(*layers)

        layers = []
        out_channels_feat = [128, 128, 64, 64, 32]
        for i in range(5):
            in_channels = out_channels
            out_channels = out_channels_feat[i]
            # out_channels = max(out_channels, gen_num_channels)

            # if out_channels%32!=0:
            #     c_norm_layer_type = 'gn_24'
            # else:
            #     c_norm_layer_type = norm_layer_type

            layers += [
                utils.blocks[dec_up_block_type](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    norm_layer_type=norm_layer_type,
                    activation_type=gen_activation_type,
                    conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),

                # utils.blocks[dec_up_block_type](
                #     in_channels=out_channels*2,
                #     out_channels=out_channels,
                #     norm_layer_type=norm_layer_type,
                #     activation_type=gen_activation_type,
                #     conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),
            ]

        self.dec_img_feat_blocks = nn.Sequential(*layers)

        layers = [
            utils.norm_layers[norm_layer_type](out_channels),
            # utils.norm_layers['bn'](out_channels),
            utils.activations[gen_activation_type](inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=3,
                kernel_size=1),
            nn.Sigmoid()]

        self.dec_img_head = nn.Sequential(*layers)


    def forward(self, feat, pred_feat, stage_two=False):
        img_feat = self.dec_img_blocks(feat)
        img_feat = self.dec_img_feat_blocks(torch.cat((img_feat, pred_feat), dim=1))
        img = self.dec_img_head(img_feat.float())


        if stage_two:
            return img, None, img_feat
        else:
            return img, None, None

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))
    return img