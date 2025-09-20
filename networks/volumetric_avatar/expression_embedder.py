from typing import *
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.cuda import amp
from torchvision import models
import torchvision.transforms.functional as FF
import math
import numpy as np
import os
from . import utils
from utils import args as args_utils
from dataclasses import dataclass
import copy



from repos.resnet import ResNet18

class ExpressionEmbed(nn.Module):
    """A neural network model for embedding facial expressions and head poses.
    
    This model processes facial images to extract expression embeddings and compute head pose 
    transformations. It supports both standard ResNet backbones and custom weights, with 
    options for different normalization layers and transformation scales.
    """

    @dataclass
    class Config:
        """Configuration parameters for ExpressionEmbed model.
        
        Attributes:
            lpe_head_backbone (str): Backbone architecture for head pose network
            lpe_face_backbone (str): Backbone architecture for face expression network
            image_size (int): Input image size
            project_dir (str): Project directory path for loading keypoints
            num_gpus (int): Number of GPUs to use
            lpe_output_channels (int): Output channels for landmark prediction
            lpe_output_channels_expression (int): Output channels for expression embedding
            lpe_final_pooling_type (str): Type of final pooling ('avg' or 'transformer')
            lpe_output_size (int): Output size after pooling
            lpe_head_transform_sep_scales (bool): Whether to use separate scales for transforms
            norm_layer_type (str): Type of normalization layer ('bn', 'in', 'gn', or 'bcn')
            use_head_net (bool): Whether to use head pose network
            use_smart_scale (bool): Whether to use dynamic scaling based on pose
            smart_scale_max_scale (float): Maximum scale factor for smart scaling
            smart_scale_max_tol_angle (float): Maximum tolerance angle for smart scaling
            dropout (float): Dropout probability
            custom_w (bool): Whether to use custom pretrained weights
        """
        lpe_head_backbone: str
        lpe_face_backbone: str
        image_size: int
        project_dir: str
        num_gpus: int
        lpe_output_channels: int
        lpe_output_channels_expression: int
        lpe_final_pooling_type: str
        lpe_output_size: int
        lpe_head_transform_sep_scales: bool
        norm_layer_type: str
        use_head_net: bool = False
        use_smart_scale: bool = False
        smart_scale_max_scale: float = 0.75
        smart_scale_max_tol_angle: float = 0.8  # 45 degrees
        dropout: float = 0.0
        custom_w: bool = False

    @classmethod
    def create_default(cls, project_dir: str, num_gpus: int, norm_layer_type: str,
                      image_size: int = 256, output_channels: int = 128,
                      use_smart_scale: bool = False) -> 'Config':
        """Creates a default configuration with commonly used settings.
        
        Args:
            project_dir: Project directory path
            num_gpus: Number of GPUs to use
            norm_layer_type: Type of normalization layer
            image_size: Input image size (default: 256)
            output_channels: Number of output channels (default: 128)
            use_smart_scale: Whether to use smart scaling (default: False)
        
        Returns:
            Config: Default configuration instance
        """
        return cls.Config(
            lpe_head_backbone="resnet18",
            lpe_face_backbone="resnet18",
            image_size=image_size,
            project_dir=project_dir,
            num_gpus=num_gpus,
            lpe_output_channels=output_channels,
            lpe_output_channels_expression=output_channels,
            lpe_final_pooling_type="avg",
            lpe_output_size=4,
            lpe_head_transform_sep_scales=False,
            norm_layer_type=norm_layer_type,
            use_head_net=False,
            use_smart_scale=use_smart_scale,
            smart_scale_max_scale=0.75,
            smart_scale_max_tol_angle=0.8,
            dropout=0.0,
            custom_w=False
        )

    def __init__(self, cfg: Config):
        """Initializes the ExpressionEmbed model.
        
        Args:
            cfg: Configuration object containing model parameters
        """
        super(ExpressionEmbed, self).__init__()
        self.cfg = cfg
        self.num_gpus = self.cfg.num_gpus

        # Initialize head network if required
        if self.cfg.use_head_net:
            self.net_head = ResNetWrapper(
                lpe_output_channels=self.cfg.lpe_output_channels,
                lpe_final_pooling_type=self.cfg.lpe_final_pooling_type,
                lpe_output_size=self.cfg.lpe_output_size,
                image_size=self.cfg.image_size,
                lpe_head_transform_sep_scales=self.cfg.lpe_head_transform_sep_scales,
                backbone=self.cfg.lpe_head_backbone, 
                pred_head_pose=True
            )
        else:
            self.net_head = None

        # Initialize face expression network
        self.net_face = ResNetWrapper(
            lpe_output_channels=self.cfg.lpe_output_channels_expression,
            lpe_final_pooling_type=self.cfg.lpe_final_pooling_type,
            lpe_output_size=self.cfg.lpe_output_size,
            image_size=self.cfg.image_size,
            lpe_head_transform_sep_scales=self.cfg.lpe_head_transform_sep_scales,
            backbone=self.cfg.lpe_face_backbone, 
            pred_expression=True, 
            dropout=self.cfg.dropout, 
            custom_w=cfg.custom_w
        )

        # Initialize transformation grids
        self.grid_size = self.cfg.image_size // 2
        grid = torch.linspace(-1, 1, self.grid_size)
        v, u = torch.meshgrid(grid, grid)
        identity_grid = torch.stack([u, v, torch.ones_like(u)], dim=2).view(1, -1, 3)
        self.register_buffer('identity_grid', identity_grid)

        # Initialize 512x512 grid for higher resolution processing
        grid = torch.linspace(-1, 1, 512)
        v, u = torch.meshgrid(grid, grid)
        identity_grid = torch.stack([u, v, torch.ones_like(u)], dim=2).view(1, -1, 3)
        self.register_buffer('identity_grid_512', identity_grid)

        # Load and process aligned keypoints
        self.lpe_head_transform_sep_scales = self.cfg.lpe_head_transform_sep_scales
        aligned_keypoints = torch.from_numpy(np.load(f'{self.cfg.project_dir}/data/aligned_keypoints_3d.npy'))
        aligned_keypoints = aligned_keypoints / self.cfg.image_size
        aligned_keypoints[:, :2] -= 0.5
        aligned_keypoints *= 2  # map to [-1, 1]
        self.register_buffer('aligned_keypoints', aligned_keypoints[None])
        self.image_size = self.cfg.image_size

        # Setup normalization layers
        if self.cfg.norm_layer_type == 'bn':
            pass
        elif self.cfg.norm_layer_type == 'in':
            self.net_face = utils.replace_bn_to_in(self.net_face, 'ExpressionEmbed_net_face')
        elif self.cfg.norm_layer_type == 'gn':
            self.net_face = utils.replace_bn_to_gn(self.net_face, 'ExpressionEmbed_net_face')
        elif self.cfg.norm_layer_type == 'bcn':
            self.net_face = utils.replace_bn_to_bcn(self.net_face, 'ExpressionEmbed_net_face')
        else:
            raise ValueError('Wrong norm type')

    def forward(self, data_dict: dict, estimate_kp_by_net: bool = False,
                use_seg: bool = False, use_aug: bool = True) -> dict:
        """Forward pass of the model.
        
        Args:
            data_dict: Dictionary containing input data including source and target images
            estimate_kp_by_net: Whether to estimate keypoints using the network
            use_seg: Whether to use segmentation masks
            use_aug: Whether to use augmentation
            
        Returns:
            Updated data dictionary with embeddings and transforms
        """
        # Prepare inputs based on segmentation flag
        if use_seg:
            inputs = torch.cat([
                data_dict['source_img'] * data_dict['source_mask'],
                data_dict['target_img'] * data_dict['target_mask']
            ])
        else:
            inputs = torch.cat([data_dict['source_img'], data_dict['target_img']])

        n = data_dict['source_img'].shape[0]
        t = data_dict['target_img'].shape[0]

        if estimate_kp_by_net:
                    data_dict['pred_source_theta'] = data_dict['source_theta']
                    data_dict['pred_target_theta'] = data_dict['target_theta']
                    theta = torch.cat([data_dict['source_theta'][:, :3, :], data_dict['target_theta'][:, :3, :]])
        else:
            # Estimate transformation parameters using the head network
            scale, rotation, translation = self.net_head(inputs)[0]
            theta = get_similarity_transform_matrix(scale, rotation, translation)

            data_dict['pred_source_theta'] = theta[:n]
            data_dict['pred_target_theta'] = theta[-t:]

        if self.training:
            # Calculate ground truth transformations during training
            if not estimate_kp_by_net:
                data_dict['source_theta'], data_dict['target_theta'] = self.estimate_theta(
                    data_dict['source_keypoints'],
                    data_dict['target_keypoints'],
                    self.num_gpus
                )
                theta = torch.cat([data_dict['source_theta'], data_dict['target_theta']])
            else:
                theta = torch.cat([data_dict['source_theta'][:,:3,:], data_dict['target_theta'][:,:3,:]])

            # Handle augmentation if specified
            if 'source_warp_aug' in data_dict.keys() and use_aug:
                inputs_face = torch.cat([data_dict['source_warp_aug'], data_dict['target_warp_aug']])
            else:
                inputs_face = torch.cat([data_dict['source_img'], data_dict['target_img']])
        else:
            # During inference, use segmentation masks if specified
            if use_seg:
                inputs_face = torch.cat([
                    data_dict['source_img'] * data_dict['source_mask'],
                    data_dict['target_img'] * data_dict['target_mask']
                ])
            else:
                inputs_face = torch.cat([data_dict['source_img'], data_dict['target_img']])

        with torch.no_grad():
            # Prepare transformation matrices for alignment
            eye_vector = torch.zeros(theta.shape[0], 1, 4)
            eye_vector[:, :, 3] = 1
            eye_vector = eye_vector.type(theta.type()).to(theta.device)
            theta_targ = torch.cat([data_dict['target_theta'][:, :3, :], data_dict['target_theta'][:, :3, :]])
            theta_ = torch.cat([theta, eye_vector], dim=1)
            theta_targ_ = torch.cat([theta_targ, eye_vector], dim=1)
            
            # Calculate inverse transformations for 2D alignment
            inv_theta_2d = theta_.float().inverse()[:, :, [0, 1, 3]][:, [0, 1, 3]]
            inv_theta_2d_ = inv_theta_2d
            inv_theta_targ_2d = theta_targ_.float().inverse()[:, :, [0, 1, 3]][:, [0, 1, 3]]

            scale = torch.zeros_like(inv_theta_2d)
            scale_full = torch.zeros_like(inv_theta_2d)
        
            if not self.cfg.use_smart_scale:
                # Apply fixed scaling
                scale[:, [0, 1], [0, 1]] = 0.5
                scale[:, 2, 2] = 1
                scale_full[:, [0, 1, 2], [0, 1, 2]] = 2
                inv_theta_2d = torch.bmm(inv_theta_2d, scale)[:, :2]
                inv_theta_targ_2d = torch.bmm(inv_theta_targ_2d, scale)[:, :2]
                inv_theta_targ_2d_full = torch.bmm(inv_theta_targ_2d, scale_full)[:, :2]
            else:
                # Apply dynamic scaling based on pose angles
                yaw_s, pitch_s, roll_s = data_dict['source_rotation'].split(1, dim=1)
                yaw_t, pitch_t, roll_t = data_dict['target_rotation'].split(1, dim=1)
                all_yaws = torch.cat([yaw_s.view(-1), yaw_t.view(-1)])
                max_scale = self.cfg.smart_scale_max_scale
                max_tol_angle = self.cfg.smart_scale_max_tol_angle
                
                # Calculate scale for each pose
                for s, yaw in enumerate(all_yaws):
                    scale_ = min(0.5 + max(
                        (torch.abs(yaw) - max_tol_angle) * (max_scale - 0.5) / (1.6 - max_tol_angle),
                        0
                    ), max_scale)
                    scale[s, [0, 1], [0, 1]] = scale_
                    if scale_ > 0.55:
                        print(torch.abs(yaw), scale_)
                scale[:, 2, 2] = 1
                inv_theta_2d = torch.bmm(inv_theta_2d, scale)[:, :2]

                # Apply target-specific scaling
                scale_t = copy.deepcopy(scale)
                scale_t[0, [0, 1], [0, 1]] = scale[-1, [0, 1], [0, 1]]
                scale_t[1, [0, 1], [0, 1]] = scale[-1, [0, 1], [0, 1]]
                inv_theta_targ_2d = torch.bmm(inv_theta_targ_2d, scale_t)[:, :2]
                scale_full[:, [0, 1, 2], [0, 1, 2]] = 2
                inv_theta_targ_2d_full = torch.bmm(inv_theta_targ_2d, scale_full)[:, :2]

            # Apply spatial transforms for alignment
            align_warp = self.identity_grid.repeat_interleave(n+t, dim=0)
            align_warp = align_warp.bmm(inv_theta_2d.transpose(1, 2)).view(n+t, self.grid_size, self.grid_size, 2)
            inputs_face_aligned = F.grid_sample(inputs_face.float(), align_warp.float())

            # Store aligned images in data dictionary
            data_dict['source_img_align'], data_dict['target_img_align'] = inputs_face_aligned.split([n, t], dim=0)

            # Compute target warps for visualization/processing
            align_warp_targ = self.identity_grid.repeat_interleave(n + t, dim=0)
            align_warp_targ = align_warp_targ.bmm(inv_theta_targ_2d.transpose(1, 2)).view(n + t, self.grid_size, self.grid_size, 2)
            data_dict['align_warp'] = align_warp_targ

            # Compute full resolution warps
            align_warp_targ = self.identity_grid_512.repeat_interleave(n + t, dim=0)
            data_dict['align_warp_full'] = align_warp_targ.bmm(inv_theta_targ_2d_full.transpose(1, 2)).view(n + t, 512, 512, 2)[0:2]

        # Extract pose embeddings from aligned faces
        pose_embed = self.net_face(inputs_face_aligned)[0]
        data_dict['source_pose_embed'] = pose_embed[:n]
        data_dict['target_pose_embed'] = pose_embed[-t:]

        return data_dict

    def estimate_theta(self, source_keypoints: torch.Tensor, target_keypoints: torch.Tensor,
                      use_gpu: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimates optimal transformation parameters to align keypoints.
        
        Uses optimization to find the best similarity transform that aligns the input
        keypoints with pre-computed aligned keypoints.
        
        Args:
            source_keypoints: Keypoints from source images [B, K, 2/3]
            target_keypoints: Keypoints from target images [B, K, 2/3]
            use_gpu: Whether to perform computation on GPU
            
        Returns:
            Tuple containing:
                - source_theta: Transformation matrices for source images
                - target_theta: Transformation matrices for target images
        """
        keypoints = torch.cat([source_keypoints, target_keypoints], dim=0)
        keypoints = torch.cat([keypoints, torch.ones(keypoints.shape[0], keypoints.shape[1], 1).to(keypoints.device)], dim=2)

        m = keypoints.shape[0]

        # Initialize transformation parameters
        if self.lpe_head_transform_sep_scales:
            # scale_x, scale_y, scale_z, yaw, pitch, roll, dx, dy, dz
            param = torch.FloatTensor([1, 1, 1, 0, 0, 0, 0, 0, 0])
            param = param[None].repeat_interleave(m, dim=0)
        else:
            param = torch.FloatTensor([1, 0, 0, 0, 0, 0, 0])  # scale, yaw, pitch, roll, dx, dy, dz
            param = param[None].repeat_interleave(m, dim=0)

        if use_gpu:
            param = param.cuda()

        # Split parameters into components
        if self.lpe_head_transform_sep_scales:
            scale, rotation, translation = param.split([3, 3, 3], dim=1)
        else:
            scale, rotation, translation = param.split([1, 3, 3], dim=1)

        # Setup optimization
        params = [scale, rotation, translation]
        params = [p.clone().requires_grad_() for p in params]
        opt = optim.LBFGS(params)

        def closure():
            """Optimization closure for LBFGS."""
            opt.zero_grad()
            transform_matrix = get_similarity_transform_matrix(*params)
            pred_aligned_keypoints = keypoints @ transform_matrix.transpose(1, 2)
            loss = ((pred_aligned_keypoints - self.aligned_keypoints)**2).mean()
            loss.backward()
            return loss

        # Optimize transformation parameters
        for _ in range(5):
            opt.step(closure)

        # Get final transformation matrices
        theta = get_similarity_transform_matrix(*params).detach().float()
        source_theta, target_theta = theta.split(m//2)

        return source_theta, target_theta

    def forward_image(self, image: torch.Tensor, normalize: bool = False,
                     delta_yaw: Optional[float] = None,
                     delta_pitch: Optional[float] = None) -> Tuple[torch.Tensor, ...]:
        """Processes a single image to extract pose embeddings and transformations.
        
        Args:
            image: Input image tensor
            normalize: Whether to normalize rotations and translations
            delta_yaw: Optional adjustment to yaw angle
            delta_pitch: Optional adjustment to pitch angle
            
        Returns:
            Tuple containing:
                - pose_embed: Expression embedding
                - scale: Scale parameters
                - rotation: Rotation parameters
                - translation: Translation parameters
                - pred_rotation: Original predicted rotation
                - image_align: Aligned image
        """
        scale, rotation, translation = self.net_head(image)[0]
        pred_rotation = rotation.clone()

        if normalize:
            rotation[:, [0, 1]] = 0.0  # zero rotations
            translation = torch.zeros_like(translation)  # and translations

        if delta_yaw is not None:
            rotation[:, 0] = rotation[:, 0].clamp(-math.pi/2, math.pi) + delta_yaw

        if delta_pitch is not None:
            rotation[:, 1] = rotation[:, 1].clamp(-math.pi/2, math.pi) + delta_pitch

        theta = get_similarity_transform_matrix(scale, rotation, translation)

        # Align input images using theta
        eye_vector = torch.zeros(theta.shape[0], 1, 4)
        eye_vector[:, :, 3] = 1
        eye_vector = eye_vector.type(theta.type()).to(theta.device)

        theta_ = torch.cat([theta, eye_vector], dim=1)
        inv_theta_2d = theta_.float().inverse()[:, :, [0, 1, 3]][:, [0, 1, 3]]

        # Apply zoom-in transform
        scale_t = torch.zeros_like(inv_theta_2d)
        scale_t[:, [0, 1], [0, 1]] = 0.5
        scale_t[:, 2, 2] = 1

        inv_theta_2d = torch.bmm(inv_theta_2d, scale_t)[:, :2]

        # Apply spatial transform
        align_warp = self.identity_grid.repeat_interleave(image.shape[0], dim=0)
        align_warp = align_warp.bmm(inv_theta_2d.transpose(1, 2)).view(image.shape[0], self.grid_size, self.grid_size, 2)

        image_align = F.grid_sample(image.float(), align_warp.float())

        # Extract pose embedding
        pose_embed = self.net_face(image_align)[0]

        return pose_embed, scale, rotation, translation, pred_rotation, image_align

class ResNetWrapper(nn.Module):
    """[Previous documentation retained...]"""
    
    def __init__(self,
                 lpe_output_channels,
                 lpe_final_pooling_type,
                 lpe_output_size,
                 image_size,
                 backbone,
                 lpe_head_transform_sep_scales,
                 pred_expression=False,
                 pred_head_pose=False,
                 dropout=0.0,
                 custom_w=False):
        """Initializes the ResNet wrapper with custom heads.
        
        Args:
            lpe_output_channels: Number of output channels for embeddings
            lpe_final_pooling_type: Type of pooling ('avg' or 'transformer')
            lpe_output_size: Size of output after pooling
            image_size: Size of input images
            backbone: ResNet backbone architecture name
            lpe_head_transform_sep_scales: Whether to use separate scales for transforms
            pred_expression: Whether to predict expressions
            pred_head_pose: Whether to predict head pose
            dropout: Dropout probability
            custom_w: Whether to use custom pretrained weights
        """
        super(ResNetWrapper, self).__init__()
        self.pred_expression = pred_expression
        self.pred_head_pose = pred_head_pose
        
        self.lpe_output_channels = lpe_output_channels
        self.lpe_final_pooling_type = lpe_final_pooling_type
        self.lpe_output_size = lpe_output_size
        self.image_size = image_size
        self.backbone = backbone
        self.lpe_head_transform_sep_scales = lpe_head_transform_sep_scales
        self.pred_expression = pred_expression
        self.custom_w = custom_w
        self.dropout = dropout
        
        # Initialize backbone network
        self.net = getattr(models, backbone)(pretrained=True)
        expansion = 1 if (backbone == 'resnet18' or backbone == 'resnet34') else 4
        num_outputs = lpe_output_channels

        # Load custom weights if specified
        if self.custom_w:
            self.net = ResNet18()
            checkpoint = torch.load('repos/official/Resnet18/checkpoints/best_checkpoint.tar')
            self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print('custom weights expression')
            
        # Replace final fully connected layer
        self.net.fc = nn.Conv2d(
            in_channels=512 * expansion,
            out_channels=num_outputs,
            kernel_size=1, 
            bias=False)
        
        self.drop = nn.Dropout(p=dropout)

        # Setup expression prediction head
        if self.pred_expression:
            if lpe_final_pooling_type == 'avg':
                # Average pooling head
                self.pose_avgpool = nn.AdaptiveAvgPool2d(lpe_output_size)
                self.pose_head = nn.Linear(num_outputs * lpe_output_size**2, num_outputs, bias=False)
            elif lpe_final_pooling_type == 'transformer':
                # Transformer-based head
                num_inputs = (image_size // 2**5)**2
                self.pose_head = nn.Sequential(
                    utils.TransformerHead(num_inputs, num_outputs, depth=3, heads=8,
                                        dim_head=64, mlp_dim=1024, dropout=0.1, emb_dropout=0.1),
                    nn.LayerNorm(num_outputs),
                    nn.Linear(num_outputs, num_outputs, bias=False))

        # Setup head pose prediction head
        if self.pred_head_pose:
            self.theta_avgpool = nn.AdaptiveAvgPool2d(1)
            num_params = 9 if lpe_head_transform_sep_scales else 7
            self.param_head = nn.Linear(num_outputs, num_params)
            
            # Initialize parameter head with zeros and appropriate bias
            self.param_head.weight.data.zero_()
            if lpe_head_transform_sep_scales:
                # scale_x, scale_y, scale_z, yaw, pitch, roll, dx, dy, dz
                self.param_head.bias.data.copy_(torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float))
            else:
                # scale, yaw, pitch, roll, dx, dy, dz
                self.param_head.bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.float))

        # Register normalization parameters
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std',  torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None])

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation of the forward pass through the ResNet backbone.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed feature tensor
        """
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x) if not self.custom_w else x

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.fc(x)
        x = self.drop(x)

        return x

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x: Input image tensor
            
        Returns:
            List of outputs depending on configuration:
                - Expression embeddings if pred_expression is True
                - Pose parameters if pred_head_pose is True
        """
        # Preprocess input
        if self.custom_w:
            x = FF.rgb_to_grayscale(x)
        else:
            x = (x - self.mean) / self.std
            
        x = self._forward_impl(x)
        outputs = []

        # Generate expression embeddings if requested
        if self.pred_expression:
            if hasattr(self, 'pose_avgpool'):
                pose_embed = self.pose_avgpool(x)
                pose_embed = torch.flatten(pose_embed, 1)
            else:
                pose_embed = x

            pose_embed = self.pose_head(pose_embed)
            outputs.append(pose_embed)
        
        # Generate pose parameters if requested
        if self.pred_head_pose:
            param = self.theta_avgpool(x)
            param = torch.flatten(param, 1)
            param = self.param_head(param)

            if param.shape[1] == 7:
                scale, rotation, translation = param.split([1, 3, 3], dim=1)
            elif param.shape[1] == 9:
                scale, rotation, translation = param.split([3, 3, 3], dim=1)
            else:
                raise ValueError("Invalid parameter dimension")

            outputs.append((scale, rotation, translation))

        return outputs


def get_similarity_transform_matrix(scale: torch.Tensor,
                                 rotation: torch.Tensor,
                                 translation: torch.Tensor) -> torch.Tensor:
    """Computes a similarity transformation matrix from scale, rotation, and translation parameters.
    
    Args:
        scale: Scale parameters [batch_size, 1 or 3]
        rotation: Euler angles (yaw, pitch, roll) [batch_size, 3]
        translation: Translation parameters [batch_size, 3]
        
    Returns:
        Transformation matrices [batch_size, 3, 4]
    """
    # Initialize identity matrix for each sample in batch
    eye_matrix = torch.eye(4)[None].repeat_interleave(scale.shape[0], dim=0)
    eye_matrix = eye_matrix.type(scale.type()).to(scale.device)

    # Construct scale transform
    S = eye_matrix.clone()
    if scale.shape[1] == 3:  # Separate scales for each dimension
        S[:, 0, 0] = scale[:, 0]
        S[:, 1, 1] = scale[:, 1]
        S[:, 2, 2] = scale[:, 2]
    else:  # Single uniform scale
        S[:, 0, 0] = scale[:, 0]
        S[:, 1, 1] = scale[:, 0]
        S[:, 2, 2] = scale[:, 0]

    # Construct rotation transform
    R = eye_matrix.clone()
    rotation = rotation.clamp(-math.pi/2, math.pi)  # Clamp rotation angles
    yaw, pitch, roll = rotation.split(1, dim=1)
    yaw, pitch, roll = yaw[:, 0], pitch[:, 0], roll[:, 0]  # squeeze angles

    # Compute trigonometric values
    yaw_cos, yaw_sin = yaw.cos(), yaw.sin()
    pitch_cos, pitch_sin = pitch.cos(), pitch.sin()
    roll_cos, roll_sin = roll.cos(), roll.sin()

    # Fill rotation matrix using Euler angles
    R[:, 0, 0] = yaw_cos * pitch_cos
    R[:, 0, 1] = yaw_cos * pitch_sin * roll_sin - yaw_sin * roll_cos
    R[:, 0, 2] = yaw_cos * pitch_sin * roll_cos + yaw_sin * roll_sin

    R[:, 1, 0] = yaw_sin * pitch_cos
    R[:, 1, 1] = yaw_sin * pitch_sin * roll_sin + yaw_cos * roll_cos
    R[:, 1, 2] = yaw_sin * pitch_sin * roll_cos - yaw_cos * roll_sin

    R[:, 2, 0] = -pitch_sin
    R[:, 2, 1] = pitch_cos * roll_sin
    R[:, 2, 2] = pitch_cos * roll_cos

    # Construct translation transform
    T = eye_matrix.clone()
    T[:, 0, 3] = translation[:, 0]
    T[:, 1, 3] = translation[:, 1]
    T[:, 2, 3] = translation[:, 2]

    # Combine transforms: Scale -> Rotate -> Translate
    theta = (S @ R @ T)[:, :3]

    return theta

"""
Key Components and Their Interactions:

1. ExpressionEmbed:
   - Main class that orchestrates the entire pipeline
   - Handles image alignment and transformation
   - Manages both expression and pose prediction networks

2. ResNetWrapper:
   - Adapts ResNet for expression and pose tasks
   - Supports both standard and custom pretrained weights
   - Provides flexible output heads for different tasks

3. Transformation Pipeline:
   - Estimates and applies similarity transforms
   - Supports smart scaling based on pose angles
   - Handles both training and inference modes

4. Key Features:
   - Expression embedding generation
   - Head pose estimation
   - Image alignment and warping
   - Support for segmentation masks
   - Augmentation handling
   - Custom weight loading

Usage Notes:
1. The model expects preprocessed face images
2. Supports both single and separate scale parameters
3. Can handle both RGB and grayscale inputs
4. Provides flexibility in normalization approaches
5. Supports various pooling types for feature aggregation
"""