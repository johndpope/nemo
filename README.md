# EMOPortraits (nemo fork)

Fork of the official implementation of **EMOPortraits: Emotion-enhanced Multimodal One-shot Head Avatars**.

This fork includes fixes for hardcoded paths and improved configurability for use as a submodule.

![EMOPortraits Example](./data/EP_v.gif)

## Overview

EMOPortraits introduces a novel approach for generating realistic and expressive one-shot head avatars driven by multimodal inputs, including extreme and asymmetric emotions.

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