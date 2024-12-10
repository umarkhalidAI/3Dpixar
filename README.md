# Image Editing

This script processes and edits video files using various image editing models

## Features

- Process multiple video files from a specified directory.
- Apply customizable transformations guided by textual prompts.
- Configure parameters such as guidance scales and diffusion steps.
- Process video frames in batches for improved efficiency.
- Optionally use CLIP-based predictions for LangSAM.

## Running the Code

**Option 1 (Recommended)** – Use an explicit label to guide editing:

```
python Image_Editing_folder.py \
    --directory "./../../output_clips/02206.Axon_Body_4_Video_2023-11-27_1542_D01A1513G_1590_5146/" \
    --prompt "turn jacket green"\
     --force_512 
     --label "jacket"

```

**Option 2** – Automatically identify labels using CLIP-based predictions::

```
python Image_Editing_auto_mask.py --video_path "/path/to/videos" \
    --guidance_scale 8.0 \
    --image_guidance_scale 2.0 \
    --diffusion_steps 100 \
    --prompt "Turn sky cloudy" \
    
```



# User-Defined Masking vs. Automatic Identification

**Option 1:** The user explicitly defines the label for the object they wish to edit, providing direct control over the target region.

**Option 2:** The script automatically determines the target object label through a grounding mask extraction stage. This process uses CLIP-Score Filtering to extract key noun phrases from the prompt and then leverages [Grounded-SAM](https://github.com/IDEA-Research/GroundingDINO) to create the final segmentation mask.



# Installation

We provide an `requirement.txt` file with most dependencies. Follow these steps:

**Create and activate the environment:**



## Following is Optional only if you want to use Automatically Identify the Mask-Label##

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
