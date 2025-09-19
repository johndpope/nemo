"""
Background extraction and compositing utilities for VASA-1 pipelines.
Separated from pipeline2.py and pipeline4.py for better code organization.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple, Union


class BackgroundProcessor:
    """Handles background extraction and compositing for face animation pipelines."""

    def __init__(self, lama_model_path: str = 'repos/jit_lama.pt', device: str = 'cuda'):
        """
        Initialize the background processor.

        Args:
            lama_model_path: Path to the JIT compiled LAMA model
            device: Device to run the model on
        """
        self.device = device
        self.lama = torch.jit.load(lama_model_path).to(device)
        self.modnet = None  # Will be set from pipeline
        self.face_idt = None  # Will be set from pipeline if using face parsing

    def set_models(self, modnet=None, face_idt=None):
        """
        Set external models used for masking.

        Args:
            modnet: MODNet model for matting
            face_idt: Face parsing model
        """
        self.modnet = modnet
        self.face_idt = face_idt

    def to_tensor(self, img: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        if isinstance(img, torch.Tensor):
            return img
        return transforms.ToTensor()(img)

    def to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        return transforms.ToPILImage()(tensor.cpu())

    def get_modnet_mask(self, img: torch.Tensor) -> torch.Tensor:
        """
        Get mask using MODNet.

        Args:
            img: Input image tensor [B, C, H, W]

        Returns:
            Mask tensor [B, 1, H, W]
        """
        if self.modnet is None:
            raise ValueError("MODNet model not set. Call set_models() first.")

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
        im = torch.nn.functional.interpolate(im, size=(im_rh, im_rw), mode='area')

        _, _, matte = self.modnet(im, True)

        # Resize back to original size
        matte = torch.nn.functional.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte.repeat(1, 1, 1, 1)  # Add channel dimension if needed

        return matte

    def get_mask(self, source_img_crop: torch.Tensor) -> torch.Tensor:
        """
        Get basic mask using MODNet.

        Args:
            source_img_crop: Input image tensor

        Returns:
            Mask tensor clamped to [0, 1]
        """
        source_img_mask = self.get_modnet_mask(source_img_crop)
        source_img_mask = source_img_mask.clamp_(max=1, min=0)
        return source_img_mask

    @torch.no_grad()
    def get_mask_fp(self, source_img_crop: torch.Tensor) -> torch.Tensor:
        """
        Get mask using face parsing model combined with MODNet.

        Args:
            source_img_crop: Input image tensor

        Returns:
            Combined face mask
        """
        if self.face_idt is None:
            raise ValueError("Face parsing model not set. Call set_models() first.")

        face_mask_source, _, _, cloth_s = self.face_idt.forward(source_img_crop)
        threshold = 0.6
        face_mask_source = (face_mask_source > threshold).float()
        source_mask_modnet = self.get_mask(source_img_crop)
        face_mask_source = (face_mask_source * source_mask_modnet).float()
        return face_mask_source

    def extract_background(self,
                          source_img: Union[Image.Image, torch.Tensor],
                          use_modnet: bool = True,
                          dilation_kernel_size: int = 21,
                          dilation_iterations: int = 2) -> Tuple[torch.Tensor, Image.Image]:
        """
        Extract background from source image using LAMA inpainting.

        Args:
            source_img: Source image (PIL Image or tensor)
            use_modnet: Whether to use MODNet for masking (vs face parsing)
            dilation_kernel_size: Size of dilation kernel for mask expansion
            dilation_iterations: Number of dilation iterations

        Returns:
            Tuple of (background_tensor, background_image)
        """
        # Convert to tensor if needed
        if isinstance(source_img, Image.Image):
            gt_img_t = self.to_tensor(source_img)[:3].unsqueeze(dim=0).to(self.device)
        else:
            gt_img_t = source_img

        # Get mask
        if use_modnet:
            m = self.get_mask(gt_img_t)
        else:
            m = self.get_mask_fp(gt_img_t)

        # Create dilated mask for better inpainting
        kernel_back = np.ones((dilation_kernel_size, dilation_kernel_size), 'uint8')
        mask = (m >= 0.8).float()
        mask = mask[0].permute(1, 2, 0)
        dilate_mask = cv2.dilate(mask.cpu().numpy(), kernel_back, iterations=dilation_iterations)
        dilate_mask = torch.FloatTensor(dilate_mask).unsqueeze(0).unsqueeze(0).to(self.device)

        # Inpaint background using LAMA
        background = self.lama(gt_img_t, dilate_mask)
        bg_img = self.to_image(background[0])

        # Resize to 512x512 for consistency
        bg = self.to_tensor(bg_img.resize((512, 512), Image.BICUBIC))

        return bg, bg_img

    def composite_with_background(self,
                                 foreground_img: Union[Image.Image, torch.Tensor],
                                 background: torch.Tensor,
                                 use_modnet: bool = True,
                                 mask_threshold: float = 0.3,
                                 mask_power: float = 8.0) -> Image.Image:
        """
        Composite foreground image with background using alpha blending.

        Args:
            foreground_img: Foreground image (generated face)
            background: Background tensor
            use_modnet: Whether to use MODNet for masking
            mask_threshold: Threshold for mask binarization
            mask_power: Power to apply to mask for sharper edges

        Returns:
            Composited PIL Image
        """
        # Convert foreground to tensor if needed
        if isinstance(foreground_img, Image.Image):
            pred_img_t = self.to_tensor(foreground_img)[:3].unsqueeze(0).to(self.device)
        else:
            pred_img_t = foreground_img

        # Get mask for foreground
        if use_modnet:
            source_img_mask = self.get_modnet_mask(pred_img_t)
        else:
            source_img_mask = self.get_mask_fp(pred_img_t)

        # Process mask for better compositing
        mask_processed = torch.where(
            source_img_mask > mask_threshold,
            source_img_mask,
            source_img_mask * 0
        ) ** mask_power

        # Composite: foreground * mask + background * (1 - mask)
        out_composite = mask_processed.cpu() * pred_img_t.cpu() + (1 - mask_processed.cpu()) * background.cpu()

        return self.to_image(out_composite[0])

    def process_frame_with_background(self,
                                     frame: Union[Image.Image, torch.Tensor],
                                     background: Optional[torch.Tensor] = None,
                                     source_for_bg: Optional[Union[Image.Image, torch.Tensor]] = None,
                                     use_modnet: bool = True) -> Image.Image:
        """
        Process a single frame with background extraction or compositing.

        Args:
            frame: Frame to process
            background: Pre-extracted background (if None, will extract from source_for_bg)
            source_for_bg: Source image for background extraction (if background is None)
            use_modnet: Whether to use MODNet for masking

        Returns:
            Processed frame with background
        """
        # Extract background if not provided
        if background is None:
            if source_for_bg is None:
                raise ValueError("Either background or source_for_bg must be provided")
            background, _ = self.extract_background(source_for_bg, use_modnet)

        # Composite frame with background
        return self.composite_with_background(frame, background, use_modnet)


# Convenience functions for backward compatibility
_global_processor = None

def init_background_processor(lama_path: str = 'repos/jit_lama.pt',
                            modnet=None,
                            face_idt=None,
                            device: str = 'cuda') -> BackgroundProcessor:
    """Initialize global background processor."""
    global _global_processor
    _global_processor = BackgroundProcessor(lama_path, device)
    if modnet is not None or face_idt is not None:
        _global_processor.set_models(modnet, face_idt)
    return _global_processor

def get_bg(s_img: Union[Image.Image, torch.Tensor],
          mdnt: bool = True) -> Tuple[torch.Tensor, Image.Image]:
    """
    Extract background from image (backward compatible function).

    Args:
        s_img: Source image
        mdnt: Use MODNet if True, face parsing if False

    Returns:
        Tuple of (background_tensor, background_PIL)
    """
    if _global_processor is None:
        raise RuntimeError("Background processor not initialized. Call init_background_processor() first.")
    return _global_processor.extract_background(s_img, use_modnet=mdnt)

def connect_img_and_bg(img: Union[Image.Image, torch.Tensor],
                      bg: torch.Tensor,
                      mdnt: bool = True) -> Image.Image:
    """
    Composite image with background (backward compatible function).

    Args:
        img: Foreground image
        bg: Background tensor
        mdnt: Use MODNet if True, face parsing if False

    Returns:
        Composited PIL Image
    """
    if _global_processor is None:
        raise RuntimeError("Background processor not initialized. Call init_background_processor() first.")
    return _global_processor.composite_with_background(img, bg, use_modnet=mdnt)