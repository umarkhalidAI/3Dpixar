# Video Editing

This script processes and edits video files using an Ip2p model with cross-frame attention. It supports various customizable parameters for creative transformations, including guidance scales, diffusion steps, and batch processing.

## Features

- Process multiple video files from a specified directory.
- Apply customizable transformations guided by textual prompts.
- Configure parameters such as guidance scales and diffusion steps.
- Process video frames in batches for improved efficiency.
- Optionally use CLIP-based predictions for LangSAM.

## Running the Code

**Option 1 (Recommended)** – Use an explicit label to guide editing:

````bash
python Video_Editing.py --directory "/path/to/videos" \
    --guidance_scale 8.0 \
    --image_guidance_scale 2.0 \
    --diffusion_steps 100 \
    --prompt "Turn sky cloudy" \
    --frames_per_batch 16 \
    --label "sky"

**Option 2 (Recommended)** – Automatically identify labels using CLIP-based predictions::
```bash
python Video_Editing.py --directory "/path/to/videos" \
    --guidance_scale 8.0 \
    --image_guidance_scale 2.0 \
    --diffusion_steps 100 \
    --prompt "Turn sky cloudy" \
    --frames_per_batch 16 \
    --clip_label

**Option 3** – No segmentation-based guidance:
```bash
python Video_Editing.py --directory "/path/to/videos" \
    --guidance_scale 8.0 \
    --image_guidance_scale 2.0 \
    --diffusion_steps 100 \
    --prompt "Turn sky cloudy" \
    --frames_per_batch 16

````

# User-Defined Masking vs. Automatic Identification

**Option 1:** The user explicitly defines the label for the object they wish to edit, providing direct control over the target region.

**Option 2:** The script automatically determines the target object label through a grounding mask extraction stage. This process uses CLIP-Score Filtering to extract key noun phrases from the prompt and then leverages [Grounded-SAM](https://github.com/IDEA-Research/GroundingDINO) to create the final segmentation mask.

**Option 3:** No masking is used, allowing the entire frame to be modified according to the prompt without localized segmentation.

# Installation

We provide an `environment.yml` file with most dependencies. Follow these steps:

**Create and activate the environment:**

```bash
conda env create -f environment.yml
conda activate <your_environment_name>
```


You will need to download the NLP model separately with:

```
python -m spacy download en_core_web_sm

```

Lastly, to install GroundingDINO:

```
export CUDA_HOME=/path/to/your/cuda/installation

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
python setup.py install

mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../../
```


