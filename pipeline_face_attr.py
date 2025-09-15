import argparse
import os
import cv2
import torch
import numpy as np
from torch import nn
from glob import glob
from PIL import Image
import h5py

from torchvision.transforms import transforms
from torch.nn import functional as F
from tqdm import trange, tqdm
from torchvision.transforms import ToTensor, ToPILImage

from networks.volumetric_avatar import FaceParsing
from repos.MODNet.src.models.modnet import MODNet
from ibug.face_detection import RetinaFacePredictor

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
    print("Warning: apex not found, running without mixed precision")
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
# from bilayer_model.external.Graphonomy.wrapper import SegmentationWrapper
import contextlib
none_context = contextlib.nullcontext()
from typing import *
from logger import logger
from PIL import Image
from vis_helper import save_expression_embed


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


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


class InferenceWrapper(nn.Module):
    def __init__(self, experiment_name, which_epoch='latest', model_file_name='', use_gpu=True, num_gpus=1,
                 fixed_bounding_box=False, project_dir='./', folder= 'mp_logs', model_ = 'va',
                 torch_home='', debug=False, print_model=False, print_params=True, args_overwrite={}, state_dict=None, pose_momentum=0.5, rank=0, args_path=None):
        super(InferenceWrapper, self).__init__()
        self.use_gpu = use_gpu
        self.debug = debug
        self.num_gpus = num_gpus

        self.modnet_pass = 'repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'

        # Get a config for the network
        args_path = pathlib.Path(project_dir) / folder / experiment_name / 'args.txt' if args_path is None else args_path

        self.args = args_utils.parse_args(args_path)
        # Add args from args_overwrite dict that overwrite the default ones
        self.args.project_dir = project_dir
        if args_overwrite is not None:
            for k, v in args_overwrite.items():
                setattr(self.args, k, v)

        if torch_home:
            os.environ['TORCH_HOME'] = torch_home

        if self.num_gpus > 0:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.manual_seed_all(self.args.random_seed)

        self.check_grads = self.args.check_grads_of_every_loss
        self.print_model = print_model
        # Set distributed training options
        if self.num_gpus <= 1:
            self.rank = 0

        elif self.num_gpus > 1 and self.num_gpus <= 8:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.rank)

        elif self.num_gpus > 8:
            raise

        #print(self.args)
        # Initialize model

        self.model = importlib.import_module(f'models.stage_1.volumetric_avatar.{model_}').Model(self.args, training=False)

        if rank == 0 and print_params:
            for n, p in self.model.net_param_dict.items():
                print(f'Number of perameters in {n}: {p}')

        if self.use_gpu:
            self.model.cuda()

        if self.rank == 0 and self.print_model:
            print(self.model)
            ms = torch_summarize(self.model)

        # Load pre-trained weights
        self.model_checkpoint = pathlib.Path(project_dir) / folder / experiment_name / 'checkpoints' / model_file_name
        if self.args.model_checkpoint:
            if self.rank == 0:
                pass
            self.model_dict = torch.load(self.model_checkpoint, map_location='cpu') if state_dict==None else state_dict
            self.model.load_state_dict(self.model_dict, strict=False)

        # Initialize distributed training
        if self.num_gpus > 1:
            self.model = apex.parallel.convert_syncbn_model(self.model)
            self.model = apex.parallel.DistributedDataParallel(self.model)

        self.model.eval()

        self.modnet = MODNet(backbone_pretrained=False)

        if self.num_gpus > 0:
            self.modnet = nn.DataParallel(self.modnet).cuda()

        if self.use_gpu:
            self.modnet = self.modnet.cuda()

        self.modnet.load_state_dict(torch.load(self.modnet_pass, map_location='cpu'))
        self.modnet.eval()

        # Face detection is required as pre-processing
        device = 'cuda' if self.use_gpu else 'cpu'
        self.device = device
        face_detector = 'sfd'
        face_detector_module = __import__('face_alignment.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=False)

        # Face tracking and bounding box smoothing parameters
        self.fixed_bounding_box = False
        self.momentum = 0.01

        self.prev_targets_to_init_recv = {'source_delta_thetas': None, 'source_thetas': None, 'target_delta_thetas': None,
                                            'source_uvs': None, 'source_deltas': None, 'target_pose_embed': None,
                                           'target_volume_deltas': None, 'target_volume': None}
        self.prev_targets = {'source_delta_thetas': None, 'source_thetas': None, 'target_delta_thetas': None,
                              'source_uvs': None, 'source_deltas': None, 'target_pose_embed': None,
                             'target_volume_deltas': None, 'target_volume': None}

        self.prev_targets_final = {'source_delta_thetas': None, 'source_thetas': None, 'target_delta_thetas': None,
                              'source_uvs': None, 'source_deltas': None, 'target_pose_embed': None,
                             'target_volume_deltas': None, 'target_volume': None}

        self.mix_old = True
        self.smooth_theta = False

        self.tracking_params = {}

        self.delta_yaw, self.delta_pitch = None, None

        # Load mediapipe detector
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5)
        self.mtcnn = MTCNN(image_size=224, device='cuda')
        self.delta_yaw, self.delta_pitch = None, None
        self.pose_momentum = pose_momentum

        self.custome_target_pose_embed = None
        self.custome_target_theta_embed = None


        # From paper3_1 pipeline
        self.source_img_with_torso = None
        self.source_img_with_bg = None
        self.source_torso_only = None
        self.source_bg_only = None
        self.source_crop_bbox = {}
        self.crop_center = None
        self.crop_size = None
        self.identity_frame = None

        self.tracking_is_on = True
        #self.smooth_pose_theta = SmoothPoseTheta()

        self.center = None
        self.size = None
        self.theta = None
        self.delta_yaw = None
        self.delta_pitch = None
        self.pose_momentum = pose_momentum
        self.mix = False

        self.target_theta = False

        if rank == 0 and print_model:
            print(torch_summarize(self.model))

    def preprocess_source(self, source_image):
        """Preprocess and set source image for face attributes driven generation."""
        log_processing_step("Preprocessing Source Image")

        # Store identity frame for later use
        self.identity_frame = source_image
        self.s_img = source_image

        # Convert to tensor if needed
        if isinstance(source_image, Image.Image):
            source_tensor = to_tensor(source_image).unsqueeze(0).cuda()
        else:
            source_tensor = source_image

        log_tensor_state("Source tensor", source_tensor)

        return source_tensor

    def forward_with_face_attrs(self, face_attrs_dict, frame_idx=0):
        """
        Drive the source image using face attributes instead of a driver video.
        We have identity, volumetric data, and expressions - just need to combine them.
        """
        log_processing_step(f"Forward with Face Attributes - Frame {frame_idx}")

        with torch.no_grad():
            # Use the pre-computed embeddings from face attributes
            # These should already be in the right format from the cache

            # Get expression embedding directly from cache
            if 'expression_embed' in face_attrs_dict:
                expression_embed = face_attrs_dict['expression_embed']
                if not isinstance(expression_embed, torch.Tensor):
                    expression_embed = torch.tensor(expression_embed).cuda()
                if expression_embed.dim() == 1:
                    expression_embed = expression_embed.unsqueeze(0)

                # Set as custom target theta embed for expression control
                self.custome_target_theta_embed = expression_embed

            # Get theta (head pose) from cache
            if 'theta' in face_attrs_dict:
                theta = face_attrs_dict['theta']
                if not isinstance(theta, torch.Tensor):
                    theta = torch.tensor(theta).cuda()
                if theta.dim() == 2:
                    theta = theta.unsqueeze(0)

                # Set as custom target pose
                self.custome_target_pose_embed = theta

            # For first frame, process source image
            # For subsequent frames, use mix mode to maintain continuity
            source_img = self.identity_frame if frame_idx == 0 else None

            # Call the original forward method
            # This will use the volumetric avatar with our cached expressions
            result = self.forward(
                source_image=source_img,
                driver_image=None,  # No driver needed - we have the attributes
                custome_target_pose_embed=self.custome_target_pose_embed,
                custome_target_theta_embed=self.custome_target_theta_embed,
                target_theta=True,
                mix=True,
                mix_old=True if frame_idx > 0 else False,
                frame_idx=frame_idx,
                modnet_mask=False
            )

            return result

    def forward(self, source_image=None, driver_image=None, source_mask=None, source_mask_add=0,
         driver_mask=None, crop=True, reset_tracking=False, smooth_pose=False,
         hard_normalize=False, soft_normalize=False, delta_yaw=None, delta_pitch=None, cloth=False,
         thetas_pass='', theta_n=0, target_theta=True, mix=False, mix_old=True,
         c_source_latent_volume=None, c_target_latent_volume=None, custome_target_pose_embed=None,
         custome_target_theta_embed=None, no_grad_infer=True, modnet_mask=False,frame_idx=0):

        log_processing_step("Forward Pass Initialization")
        log_tensor_state("Input source image", source_image)
        log_tensor_state("Input driver image", driver_image)

        self.no_grad_infer = no_grad_infer
        self.target_theta = target_theta

        with torch.no_grad():
            if reset_tracking:
                log_processing_step("Resetting Tracking Parameters")
                self.center = None
                self.size = None
                self.theta = None
                self.delta_yaw = None
                self.delta_pitch = None

            self.mix = mix
            self.mix_old = mix_old
            if delta_yaw is not None:
                self.delta_yaw = delta_yaw
            if delta_pitch is not None:
                self.delta_pitch = delta_pitch

            if source_image is not None:
                log_processing_step("Processing Source Image")
                self.s_img = source_image

                source = to_tensor(source_image).unsqueeze(0)
                source = source.cuda()

                # Rest of the forward method implementation...
                # (This is a simplified version - you'd need to copy the full implementation)

            # Handle custom embeddings if provided (for face_attr driving)
            if custome_target_pose_embed is not None or custome_target_theta_embed is not None:
                log_processing_step("Using Custom Face Attribute Embeddings")
                # Process with custom embeddings instead of driver image
                # This allows driving by face attributes

        # Return generated image (for now just return the identity frame as placeholder)
        # In a real implementation, this would generate the actual frame
        if self.identity_frame is not None:
            return [[self.identity_frame]]
        else:
            # Return a dummy image
            dummy_img = Image.new('RGB', (512, 512), color='black')
            return [[dummy_img]]


def load_face_attrs_from_h5(h5_path, window_idx=0):
    """
    Load face attributes from H5 cache file.

    Args:
        h5_path: Path to H5 cache file
        window_idx: Window index to load

    Returns:
        List of face attribute dictionaries for all frames in the window
    """
    face_attrs_list = []

    with h5py.File(h5_path, 'r') as f:
        window_key = f'window_{window_idx}'
        if window_key not in f:
            raise ValueError(f"Window {window_idx} not found in cache")

        window_group = f[window_key]

        # Load attributes for all frames
        # Assuming 50 frames per window
        num_frames = 50

        # Load tensors that apply to all frames
        gaze = torch.from_numpy(window_group['gaze'][()]).cuda()
        emotion = torch.from_numpy(window_group['emotion'][()]).cuda()
        head_distance = torch.from_numpy(window_group['head_distance'][()]).cuda()

        # Load per-frame blink states
        if 'blink_eye_left' in window_group:
            blink_left = torch.from_numpy(window_group['blink_eye_left'][()]).cuda()
            blink_right = torch.from_numpy(window_group['blink_eye_right'][()]).cuda()
        else:
            blink_left = torch.zeros(num_frames).cuda()
            blink_right = torch.zeros(num_frames).cuda()

        # Create face_attrs dict for each frame
        for frame_idx in range(num_frames):
            face_attrs = {
                'gaze': gaze[frame_idx] if gaze.dim() > 1 else gaze,
                'emotion': emotion[frame_idx] if emotion.dim() > 1 else emotion,
                'head_distance': head_distance[frame_idx] if head_distance.dim() > 0 else head_distance,
                'blink_eye_left': blink_left[frame_idx] if blink_left.dim() > 0 else blink_left,
                'blink_eye_right': blink_right[frame_idx] if blink_right.dim() > 0 else blink_right,
            }

            # Add audio features if available
            if 'audio_features' in window_group:
                audio = torch.from_numpy(window_group['audio_features'][()]).cuda()
                face_attrs['audio_features'] = audio[frame_idx] if audio.dim() > 1 else audio

            face_attrs_list.append(face_attrs)

    return face_attrs_list


def drive_image_with_face_attrs(source_image, face_attrs_h5_path, window_idx=0, max_frames=None):
    """
    Drive source image using face attributes from H5 cache.

    Args:
        source_image: PIL Image of source face
        face_attrs_h5_path: Path to H5 file containing face attributes
        window_idx: Window index to use from cache
        max_frames: Maximum number of frames to generate

    Returns:
        List of generated images
    """
    # Load face attributes from cache
    face_attrs_list = load_face_attrs_from_h5(face_attrs_h5_path, window_idx)

    if max_frames is not None:
        face_attrs_list = face_attrs_list[:max_frames]

    # Preprocess source image
    inferer.preprocess_source(source_image)

    # Generate frames driven by face attributes
    generated_frames = []

    for frame_idx, face_attrs in enumerate(tqdm(face_attrs_list, desc="Generating frames")):
        # Generate frame using face attributes
        result = inferer.forward_with_face_attrs(face_attrs, frame_idx=frame_idx)
        generated_frames.append(result[0][0])

    return generated_frames


def make_video_from_frames(frames, save_path, fps=25.0):
    """Save list of PIL images as video."""
    if not frames:
        print("No frames to save")
        return

    # Get dimensions from first frame
    first_frame = frames[0]
    if isinstance(first_frame, Image.Image):
        width, height = first_frame.size
    else:
        height, width = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for frame in frames:
        if isinstance(frame, Image.Image):
            frame_np = np.array(frame)
        else:
            frame_np = frame

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved to {save_path}")


# Initialize models (same as original pipeline2.py)
project_dir = os.path.dirname(os.path.abspath(__file__))
args_overwrite = {'l1_vol_rgb':0}
face_idt = FaceParsing(None, 'cuda')

lama = torch.jit.load('repos/jit_lama.pt').cuda()

modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet).cuda()
modnet.load_state_dict(torch.load('repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'))
modnet.eval()

threshold = 0.8
device = 'cuda'
face_detector = RetinaFacePredictor(threshold=threshold, device=device,
                                   model=(RetinaFacePredictor.get_model('mobilenet0.25')))

inferer = InferenceWrapper(experiment_name = 'Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1',
                          model_file_name = '328_model.pth',
                          project_dir = project_dir, folder = 'logs', state_dict = None,
                          args_overwrite=args_overwrite, pose_momentum = 0.1,
                          print_model=False, print_params = True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_image_path', type=str, default='data/IMG_1.png',
                       help='Path to source image')
    parser.add_argument('--face_attrs_h5', type=str,
                       default='cache_single_bucket/all_windows_cache.h5',
                       help='Path to H5 file containing face attributes')
    parser.add_argument('--window_idx', type=int, default=0,
                       help='Window index to use from cache')
    parser.add_argument('--saved_to_path', type=str, default='data/face_attr_result.mp4',
                       help='Path to save result video')
    parser.add_argument('--fps', type=float, default=25.0, help='FPS of output video')
    parser.add_argument('--max_frames', type=int, default=50,
                       help='Maximum number of frames to process')

    args = parser.parse_args()

    # Load source image
    source_img = to_512(Image.open(args.source_image_path))

    # Generate video driven by face attributes
    generated_frames = drive_image_with_face_attrs(
        source_img,
        args.face_attrs_h5,
        window_idx=args.window_idx,
        max_frames=args.max_frames
    )

    # Save as video
    make_video_from_frames(generated_frames, args.saved_to_path, fps=args.fps)