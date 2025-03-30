"""
LTX-Video Image-to-Video Generator (subprocess approach)

This script converts a static image into a video animation using the LTX-Video model from Lightricks.
It uses subprocess to call the inference.py script with appropriate parameters and automatically 
uses the most recent image from the INPUT_DIR folder.

Features:
- Automatically selects the most recent image from INPUT_DIR
- Adjusts image dimensions to be compatible with the model (divisible by 32)
- Scales large images down to 720p max resolution
- Supports random or fixed seeds for reproducibility
- Simple subprocess approach with minimal dependencies
- Saves output with timestamp and seed value in filename

Requirements:
- PyTorch with CUDA support
- Sufficient GPU memory for generation

One-liner example:
    python USER_DIR/02_image_to_video.py
"""

# Copy-paste ready command (direct inference.py equivalent):
# python inference.py --ckpt_path MODEL_DIR/ltx-video-2b-v0.9.5.safetensors --prompt "A beautiful cinematic animation based on the input image" --height 720 --width 1280 --num_frames 241 --seed 42 --output_path OUTPUT_DIR --device cuda --frame_rate 24 --conditioning_media_paths INPUT_DIR/example.jpg --conditioning_start_frames 0 --prompt_enhancement_words_threshold 500 --num_inference_steps 40 --guidance_scale 3.0 --negative_prompt "Low resolution, inconsistent motion, visual artifacts, jitter, blur, distortion, unrealistic or misaligned frames, poor composition, dull lighting."

# ============ USER CONFIGURABLE PARAMETERS ============
# Set these values before running the script

# Video dimensions - set to None to auto-detect from input image
# If only one is set, the other will be calculated maintaining aspect ratio
WIDTH = None 
HEIGHT = None

# Maximum dimensions if auto-detecting (720p)
MAX_WIDTH = 1280
MAX_HEIGHT = 720

# Generation parameters
FPS = 24                     # Frames per second
NUM_FRAMES = 241             # 10 seconds at 24 FPS (multiple of 8 + 1)
NUM_INFERENCE_STEPS = 40     # Number of denoising steps

# Set to None for random seed each time
SEED = None

# Prompt parameters
PROMPT = "A beautiful cinematic animation based on the input image. The motion is smooth and fluid, enhancing the original scene with dynamic elements."
NEGATIVE_PROMPT = "Low resolution, inconsistent motion, visual artifacts, jitter, blur, distortion, unrealistic or misaligned frames, poor composition, dull lighting."
GUIDANCE_SCALE = 3.0         # 1.0-5.0, higher values follow prompt more closely
# =====================================================

import subprocess
import os
import glob
from pathlib import Path
from datetime import datetime
from PIL import Image
import random

# One-liner for terminal:
# python USER_DIR/02_image_to_video.py
#
# Direct inference.py command equivalent (replace [LATEST_IMAGE] with your image filename):
# python inference.py --ckpt_path MODEL_DIR/ltx-video-2b-v0.9.5.safetensors --prompt "A beautiful cinematic animation based on the input image" --num_frames 241 --seed 42 --output_path OUTPUT_DIR --device cuda --frame_rate 24 --conditioning_media_paths INPUT_DIR/[LATEST_IMAGE] --conditioning_start_frames 0 --prompt_enhancement_words_threshold 500 --num_inference_steps 40 --guidance_scale 3.0

# Set up paths - Using paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from USER_DIR to project root
MODEL_DIR = os.path.join(PROJECT_ROOT, "MODEL_DIR")
INPUT_DIR = os.path.join(PROJECT_ROOT, "INPUT_DIR")  # Directory to look for input images
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "OUTPUT_DIR")
MODEL_FILE = os.path.join(MODEL_DIR, "ltx-video-2b-v0.9.5.safetensors")
INFERENCE_SCRIPT = os.path.join(PROJECT_ROOT, "inference.py")

print(f"Using directories:")
print(f"  MODEL_DIR: {MODEL_DIR}")
print(f"  INPUT_DIR: {INPUT_DIR}")
print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  MODEL_FILE: {MODEL_FILE}")
print(f"  INFERENCE_SCRIPT: {INFERENCE_SCRIPT}")

# Create input directory if it doesn't exist
Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)

# Find the latest image in INPUT_DIR
def find_latest_image(directory):
    """
    Find the most recently modified image file in the specified directory.
    
    Args:
        directory (str): Directory to search for images
        
    Returns:
        str: Path to the most recent image, or None if no images found
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))
    
    if not image_files:
        return None
    
    return max(image_files, key=os.path.getmtime)

# Get the latest image
IMAGE_PATH = find_latest_image(INPUT_DIR)

# Verify model file exists
if not os.path.exists(MODEL_FILE):
    print(f"Error: Model file not found at {MODEL_FILE}")
    available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")]
    if available_models:
        print(f"Available models: {available_models}")
        MODEL_FILE = os.path.join(MODEL_DIR, available_models[0])
        print(f"Using alternative model: {MODEL_FILE}")
    else:
        print("No safetensors models found!")
        exit(1)

# Check if input image exists
if not IMAGE_PATH:
    print(f"Warning: No image found in {INPUT_DIR}")
    print(f"Please place at least one image file (jpg, jpeg, png, bmp, webp) in the {INPUT_DIR} folder.")
    # List all files in INPUT_DIR for debugging
    print(f"Files found in {INPUT_DIR}:")
    for file in os.listdir(INPUT_DIR):
        print(f"  {file}")
    exit(1)

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Get dimensions from input image and adjust to be divisible by 32
def get_adjusted_dimensions(image_path, target_width=None, target_height=None, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """
    Adjust image dimensions to be divisible by 32 and scale down if needed.
    
    The LTX-Video model requires dimensions to be divisible by 32.
    This function also scales down large images to max resolution while maintaining aspect ratio.
    
    Args:
        image_path (str): Path to the input image
        target_width (int, optional): Specific width to use (will maintain aspect ratio if only one dimension specified)
        target_height (int, optional): Specific height to use (will maintain aspect ratio if only one dimension specified)
        max_width (int): Maximum width if auto-detecting
        max_height (int): Maximum height if auto-detecting
        
    Returns:
        tuple: (width, height) adjusted to be divisible by 32
    """
    with Image.open(image_path) as img:
        orig_width, orig_height = img.size
    
    # If specific dimensions are provided
    if target_width is not None or target_height is not None:
        # Calculate based on aspect ratio if only one dimension is provided
        if target_width is None:
            # Height was provided, calculate width based on aspect ratio
            width = int((orig_width / orig_height) * target_height)
            height = target_height
        elif target_height is None:
            # Width was provided, calculate height based on aspect ratio
            height = int((orig_height / orig_width) * target_width) 
            width = target_width
        else:
            # Both dimensions provided
            width = target_width
            height = target_height
    else:
        # Use original dimensions
        width = orig_width
        height = orig_height
    
    # Adjust to be divisible by 32 while maintaining aspect ratio
    width = ((width - 1) // 32 + 1) * 32
    height = ((height - 1) // 32 + 1) * 32
    
    # Scale down to max resolution if needed, maintaining aspect ratio
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        width = ((int(width * scale) - 1) // 32 + 1) * 32
        height = ((int(height * scale) - 1) // 32 + 1) * 32
    
    return width, height

# Process seed value
if SEED is None:
    seed = random.randint(0, 2147483647)
    print(f"Using random seed: {seed}")
else:
    seed = SEED
    print(f"Using fixed seed: {seed}")

# Get dimensions from input image
width, height = get_adjusted_dimensions(IMAGE_PATH, WIDTH, HEIGHT)

# Construct the command for inference.py
cmd = [
    "python", INFERENCE_SCRIPT,
    "--ckpt_path", MODEL_FILE,
    "--prompt", PROMPT,
    "--negative_prompt", NEGATIVE_PROMPT,
    "--height", str(height),
    "--width", str(width),
    "--num_frames", str(NUM_FRAMES),
    "--seed", str(seed),
    "--output_path", OUTPUT_DIR,
    "--device", "cuda",
    "--frame_rate", str(FPS),
    "--conditioning_media_paths", IMAGE_PATH,
    "--conditioning_start_frames", "0",
    "--prompt_enhancement_words_threshold", "500",  # Enable prompt enhancement
    "--num_inference_steps", str(NUM_INFERENCE_STEPS),
    "--guidance_scale", str(GUIDANCE_SCALE)
]

# Get image filename for output
image_filename = os.path.basename(IMAGE_PATH)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_filename = f"{os.path.splitext(image_filename)[0]}_{timestamp}_seed-{seed}.mp4"

# Print generation information
print("Running image-to-video generation...")
print(f"Using model: {MODEL_FILE}")
print(f"Input image: {IMAGE_PATH} (latest image in {INPUT_DIR})")
print(f"Original dimensions: {Image.open(IMAGE_PATH).size}")
print(f"Adjusted dimensions for generation: {width}x{height} (must be divisible by 32)")
print(f"Generating: {NUM_FRAMES} frames at {FPS} FPS ({NUM_FRAMES/FPS:.1f} seconds)")
print(f"Prompt: '{PROMPT}'")
print(f"Guidance scale: {GUIDANCE_SCALE}")
print(f"Prompt enhancement: Enabled")
print(f"Inference steps: {NUM_INFERENCE_STEPS}")
print(f"Output will be saved to: {os.path.join(OUTPUT_DIR, output_filename)}")

# Run the command
subprocess.run(cmd)

print("Generation completed! Check the outputs folder for your video.") 