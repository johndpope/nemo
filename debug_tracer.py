#!/usr/bin/env python3
"""
DebugTracer: A comprehensive debugging and tracing module for pipeline debugging.
Extracted from pipeline3.py for reuse across different pipelines.
"""

import json
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List, Union
from logger import logger


class DebugTracer:
    """Helper class for debug tracing and saving."""

    def __init__(self, output_dir: str = "debug_output", enabled: bool = True):
        """
        Initialize the DebugTracer.

        Args:
            output_dir: Directory to save debug output
            enabled: Whether debugging is enabled
        """
        self.enabled = enabled
        if not self.enabled:
            return

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.step_counter = 0
        self.trace_data = []

        # Create subdirectories
        self.img_dir = self.output_dir / "images"
        self.img_dir.mkdir(exist_ok=True)
        self.tensor_dir = self.output_dir / "tensors"
        self.tensor_dir.mkdir(exist_ok=True)
        self.json_dir = self.output_dir / "json"
        self.json_dir.mkdir(exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _extract_info(self, value: Any) -> Any:
        """
        Extract type and shape information from various objects.

        Args:
            value: The value to extract information from

        Returns:
            Extracted information in a serializable format
        """
        if isinstance(value, torch.Tensor):
            stats = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device)
            }

            # Only compute statistics for floating point tensors
            if value.dtype in [torch.float16, torch.float32, torch.float64]:
                if value.numel() > 0:
                    stats["min"] = float(value.min().item())
                    stats["max"] = float(value.max().item())
                    stats["mean"] = float(value.mean().item())
                    stats["std"] = float(value.std().item())
            elif value.numel() > 0:
                # For integer tensors
                stats["min"] = int(value.min().item())
                stats["max"] = int(value.max().item())

            return stats
        elif isinstance(value, dict):
            # Recursively extract info from dictionary values
            result = {}
            for k, v in value.items():
                if hasattr(v, 'shape'):
                    result[k] = {'shape': list(v.shape), 'dtype': str(v.dtype)}
                elif isinstance(v, (torch.Tensor, dict, list)):
                    result[k] = self._extract_info(v)
                elif isinstance(v, (str, int, float, bool, type(None))):
                    result[k] = v
                else:
                    result[k] = {'type': type(v).__name__}
            return result
        elif isinstance(value, list):
            # For lists, show the actual values if they're shape-like (small numeric lists)
            if len(value) <= 10 and all(isinstance(x, (int, float)) for x in value):
                return value
            else:
                # For larger lists or lists with complex objects, provide more info
                return {
                    'type': 'list',
                    'length': len(value),
                    'sample': value[:3] if len(value) > 3 else value
                }
        elif isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif hasattr(value, 'shape'):
            # For numpy arrays or other objects with shape
            return {'shape': list(value.shape), 'dtype': str(value.dtype)}
        else:
            return {'type': type(value).__name__}

    def log_step(self, method_name: str, stage: str = "entry", **kwargs) -> int:
        """
        Log a step with metadata.

        Args:
            method_name: Name of the method being traced
            stage: Stage of execution (entry, exit, error, etc.)
            **kwargs: Additional data to log

        Returns:
            Step number
        """
        if not self.enabled:
            return 0

        self.step_counter += 1

        step_data = {
            "step": self.step_counter,
            "method": method_name,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "data": {}
        }

        # Process all kwargs using the extraction helper
        for key, value in kwargs.items():
            step_data["data"][key] = self._extract_info(value)

        self.trace_data.append(step_data)

        # Save JSON for this step
        json_path = self.json_dir / f"step_{self.step_counter:04d}_{method_name}_{stage}.json"
        with open(json_path, 'w') as f:
            json.dump(step_data, f, indent=2)

        logger.debug(f"[TRACE {self.step_counter:04d}] {method_name}.{stage}: {list(kwargs.keys())}")

        return self.step_counter

    def save_image(self, tensor: Union[torch.Tensor, np.ndarray, Image.Image],
                   name: str, step: Optional[int] = None) -> None:
        """
        Save tensor or image as PNG file.

        Args:
            tensor: Tensor, numpy array, or PIL Image to save
            name: Name for the saved file
            step: Step number (uses current counter if None)
        """
        if not self.enabled:
            return

        if step is None:
            step = self.step_counter

        if tensor is None:
            return

        # Handle PIL Image
        if isinstance(tensor, Image.Image):
            tensor.save(self.img_dir / f"step_{step:04d}_{name}.png")
            logger.debug(f"  [IMG] Saved {name} -> step_{step:04d}_{name}.png")
            return

        # Handle numpy array
        if isinstance(tensor, np.ndarray):
            # Convert to tensor for consistent processing
            tensor = torch.from_numpy(tensor)

        # Handle different tensor formats
        if len(tensor.shape) == 4:  # NCHW
            tensor = tensor[0]

        # Handle different channel arrangements
        if len(tensor.shape) == 3:
            if tensor.shape[0] in [1, 3]:  # CHW
                img = tensor.detach().cpu().permute(1, 2, 0).numpy()
            else:
                # Not a standard image format, skip
                return
        elif len(tensor.shape) == 2:  # HW
            img = tensor.detach().cpu().numpy()
        else:
            # Not an image format, skip
            return

        # Normalize to [0, 1]
        if img.min() < -0.1:  # Likely [-1, 1] range
            img = (img + 1) / 2
        elif img.max() > 1.1:  # Likely [0, 255] range
            img = img / 255.0

        img = np.clip(img, 0, 1)

        # Convert to uint8
        img = (img * 255).astype(np.uint8)

        # Handle grayscale images with extra dimension
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze(2)

        # Save
        if len(img.shape) == 2:  # Grayscale
            Image.fromarray(img, mode='L').save(
                self.img_dir / f"step_{step:04d}_{name}.png"
            )
        elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB
            Image.fromarray(img).save(
                self.img_dir / f"step_{step:04d}_{name}.png"
            )
        elif len(img.shape) == 3 and img.shape[2] == 1:  # Grayscale with extra dim
            Image.fromarray(img.squeeze(2), mode='L').save(
                self.img_dir / f"step_{step:04d}_{name}.png"
            )

        logger.debug(f"  [IMG] Saved {name} -> step_{step:04d}_{name}.png")

    def save_tensor(self, tensor, name: str, step: Optional[int] = None) -> None:
        """
        Save tensor or dict of tensors to file.

        Args:
            tensor: Tensor or dict of tensors to save
            name: Name for the saved file
            step: Step number (uses current counter if None)
        """
        if not self.enabled:
            return

        if step is None:
            step = self.step_counter

        if tensor is None:
            return

        # Handle dict of tensors
        if isinstance(tensor, dict):
            import json
            # Save each tensor in the dict with a prefixed name
            for key, val in tensor.items():
                if isinstance(val, torch.Tensor):
                    torch.save(val.cpu(), self.tensor_dir / f"step_{step:04d}_{name}_{key}.pt")
                    logger.debug(f"  [TENSOR] Saved {name}.{key} -> step_{step:04d}_{name}_{key}.pt")
            # Also save the dict structure for reference
            dict_structure = {k: str(v.shape) if isinstance(v, torch.Tensor) else str(type(v)) for k, v in tensor.items()}
            with open(self.tensor_dir / f"step_{step:04d}_{name}_structure.json", 'w') as f:
                json.dump(dict_structure, f, indent=2)
        elif isinstance(tensor, torch.Tensor):
            torch.save(tensor.cpu(), self.tensor_dir / f"step_{step:04d}_{name}.pt")
            logger.debug(f"  [TENSOR] Saved {name} -> step_{step:04d}_{name}.pt")
        else:
            logger.warning(f"  [TENSOR] Warning: {name} is not a tensor or dict of tensors, skipping")

    def save_final_trace(self) -> None:
        """Save complete trace to JSON."""
        if not self.enabled:
            return

        trace_path = self.output_dir / f"trace_{self.session_id}.json"
        with open(trace_path, 'w') as f:
            json.dump(self.trace_data, f, indent=2)
        logger.info(f"\n[TRACE] Complete trace saved to {trace_path}")
        logger.info(f"[TRACE] Total steps: {self.step_counter}")

    def save_video_frames(self, frames: List[Union[np.ndarray, Image.Image]],
                         prefix: str = "frame") -> None:
        """
        Save a list of video frames.

        Args:
            frames: List of frames (numpy arrays or PIL Images)
            prefix: Prefix for frame filenames
        """
        if not self.enabled:
            return

        for i, frame in enumerate(frames):
            self.save_image(frame, f"{prefix}_{i:04d}")