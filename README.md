# EMOPortraits (nemo fork)

Fork of the official implementation of **EMOPortraits: Emotion-enhanced Multimodal One-shot Head Avatars**.

This fork includes fixes for hardcoded paths and improved configurability for use as a submodule.

![EMOPortraits Example](./data/EP_v.gif)

## Overview

EMOPortraits introduces a novel approach for generating realistic and expressive one-shot head avatars driven by multimodal inputs, including extreme and asymmetric emotions.

## Volumetric Avatar Warping System

The core of EMOPortraits is its volumetric avatar model that uses a sophisticated warping system to separate identity from expression. This is implemented in `models/stage_1/volumetric_avatar/va.py`.

### Key Components

#### Warp Generators
The model defines two warp generators for bidirectional transformations:

```python
# From va.py - Define networks for warping to (xy) and from (uv) canonical volume cube
self.xy_generator_nw = volumetric_avatar.WarpGenerator(self.va_config.warp_generator_cfg)
self.uv_generator_nw = volumetric_avatar.WarpGenerator(self.va_config.warp_generator_cfg)
```

#### XY Warps (Source → Canonical)
- **Generator**: `self.xy_generator_nw`
- **Purpose**: Removes expression from source to create canonical volume
- **Coordinate Space**: 3D volumetric space (16×64×64 grid)
- **Method**: `_generate_warping_fields()` in va.py
- **Effect**: Normalizes any facial expression back to neutral

#### UV Warps (Canonical → Target)
- **Generator**: `self.uv_generator_nw`
- **Purpose**: Applies target expression to canonical volume
- **Coordinate Space**: Surface/texture space (0-1 normalized)
- **Method**: `_generate_warping_fields()` in va.py
- **Effect**: Deforms neutral face to create desired expression

### Warping Pipeline

The `_generate_warping_fields()` method orchestrates the warping process:

1. **Extract embeddings** from source and target images
2. **Generate XY warps** to normalize source expression
3. **Generate UV warps** to apply target expression
4. **Apply warps sequentially** to transform the volumetric representation

```python
# Simplified warping flow from va.py
source_warp_embed → xy_generator_nw → XY warp field → Remove expression
target_warp_embed → uv_generator_nw → UV warp field → Apply expression
```

### Integration with VASA

When used with VASA for motion generation:
- XY warps are extracted to normalize identity frames to canonical space
- UV warps are predicted by VASA to apply dynamic expressions
- The combination enables expression transfer while preserving identity

## Key Improvements in this Fork

- **Fixed hardcoded paths** - All paths are now relative or configurable
- **Improved module imports** - Better path resolution for submodule usage
- **Auto-detection of project directories** - Works from any location
- **Configurable model paths** - Via config files instead of hardcoded

---

## Installation

### 1. Environment Setup

```shell
conda create -n emo python=3.12
conda activate emo
pip install -r requirements.txt

# Install l2cs from johndpope fork (fixes device handling)
pip install git+https://github.com/johndpope/l2cs-net.git

# Install soundfile for audio processing
pip install soundfile

./bootstrap.sh # fetch all the required models from gdrive
```

## Usage

### As a Submodule (Recommended)

This repository is designed to work as a git submodule. See [VASA-1-hack](https://github.com/johndpope/VASA-1-hack) for an example integration.

### Standalone

```shell
python main.py
```

---

## Recent Changes

- Added Apache 2.0 License (cherry-picked from upstream)
- Fixed path resolution in `face_parcing.py`, `perceptual.py`, and `va.py`
- Made volumetric avatar model accept config parameter
- Commented out non-essential textual_image import

---

## Acknowledgements

We extend our gratitude to all contributors and participants who made this project possible. Special thanks to:
- The original EMOPortraits team at neeek2303/EMOPortraits
- The developers of the datasets and tools that were instrumental in this research

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Original work by the EMOPortraits team. Modifications for improved modularity and path handling by this fork's contributors.