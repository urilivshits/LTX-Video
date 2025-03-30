#!/usr/bin/env python
"""
LTX-Video Image-to-Video Generator with Custom Prompt Enhancement

This script converts a static image into a video animation using the LTX-Video model from Lightricks.
It uses the Diffusers API for direct integration instead of calling inference.py
and enhances the prompt using OpenAI's GPT models for better results.

Features:
- Custom prompt enhancement using OpenAI API (GPT-4o-mini by default)
- Analyzes input image content and enhances user prompts
- HD resolution with automatic adjustment based on input image
- Customizable generation parameters
- Supports random or fixed seeds for reproducibility
- Direct Diffusers API integration for better performance
- Saves output with timestamp and seed value in filename

Requirements:
- PyTorch with CUDA support
- diffusers library
- OpenAI API key (set in .env file or environment)
- Sufficient GPU memory for generation (at least 12GB recommended)

One-liner example:
    python USER_DIR/03_image_to_video_diffusers.py
"""

# ============ USER CONFIGURABLE PARAMETERS ============
# Set these values before running the script

# Video dimensions - set to None to auto-detect from input image
# If only one is set, the other will be calculated maintaining aspect ratio
WIDTH = 320 
HEIGHT = None

# Maximum dimensions if auto-detecting (720p)
MAX_WIDTH = 1280
MAX_HEIGHT = 720

# Generation parameters
FPS = 24                     # Frames per second
NUM_FRAMES = 121             # 5 seconds at 24 FPS (multiple of 8 + 1)
NUM_INFERENCE_STEPS = 20     # Number of denoising steps

# Set to None for random seed each time
SEED = None

# Prompt parameters
POSITIVE_PROMPT = """A person standing naturally with occasional subtle movements and expressions. The movements include gentle breathing, slight shifts in posture, and natural facial micro-expressions. The lighting is even and flattering, highlighting the person's features while maintaining a realistic appearance."""
NEGATIVE_PROMPT = "Low resolution, inconsistent motion, visual artifacts, jitter, blur, distortion, unnatural movement, poor composition, unbalanced or dull lighting, unrealistic anatomy, awkward proportions, incoherent shading or motion."
GUIDANCE_SCALE = 5         # 1.0-5.0, higher values follow prompt more closely

# OpenAI model for prompt enhancement
OPENAI_MODEL = "gpt-4o-mini"  # Can be changed to other OpenAI models
MAX_PROMPT_CHARS = 400       # Maximum characters for enhanced prompt
# =====================================================

import os
import torch
import random
import glob
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import animation
from dotenv import load_dotenv

# Import our custom prompt improvement module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path
from USER_DIR.prompt_improvement import analyze_and_improve_prompt

# Load environment variables (for OpenAI API key)
load_dotenv()

# For diffusers support
try:
    # Try importing directly (newer versions of diffusers)
    from diffusers import LTXImageToVideoPipeline
    print("Using standard LTXImageToVideoPipeline import")
except ImportError:
    # If that fails, ask user to update
    print("ERROR: Could not import LTXImageToVideoPipeline. Please update your diffusers library:")
    print("pip install --upgrade diffusers")
    exit(1)

# Set up argument parser
parser = argparse.ArgumentParser(description='LTX-Video Image-to-Video Generator with Custom Prompt Enhancement')
parser.add_argument('--width', type=int, default=WIDTH, help='Video width (divisible by 32)')
parser.add_argument('--height', type=int, default=HEIGHT, help='Video height (divisible by 32)')
parser.add_argument('--num_frames', type=int, default=NUM_FRAMES, help='Number of frames (multiple of 8 + 1)')
parser.add_argument('--fps', type=int, default=FPS, help='Frames per second')
parser.add_argument('--seed', type=int, default=SEED, help='Random seed (None for random)')
parser.add_argument('--prompt', type=str, default=POSITIVE_PROMPT, help='Positive prompt (will be enhanced)')
parser.add_argument('--negative_prompt', type=str, default=NEGATIVE_PROMPT, help='Negative prompt')
parser.add_argument('--guidance_scale', type=float, default=GUIDANCE_SCALE, help='Guidance scale (1.0-5.0)')
parser.add_argument('--inference_steps', type=int, default=NUM_INFERENCE_STEPS, help='Number of inference steps')
parser.add_argument('--openai_model', type=str, default=OPENAI_MODEL, help='OpenAI model for prompt enhancement')
parser.add_argument('--max_prompt_chars', type=int, default=MAX_PROMPT_CHARS, help='Max characters for enhanced prompt')
parser.add_argument('--conditioning_frame_start_idx', type=int, default=0, help='First frame index for conditioning')
parser.add_argument('--conditioning_frame_end_idx', type=int, default=0, help='Last frame index for conditioning')
parser.add_argument('--no_prompt_enhancement', action='store_true', help='Disable OpenAI prompt enhancement')
args = parser.parse_args()

# Update parameters with command line arguments
WIDTH = args.width
HEIGHT = args.height
NUM_FRAMES = args.num_frames
FPS = args.fps
SEED = args.seed
POSITIVE_PROMPT = args.prompt
NEGATIVE_PROMPT = args.negative_prompt
GUIDANCE_SCALE = args.guidance_scale
NUM_INFERENCE_STEPS = args.inference_steps
OPENAI_MODEL = args.openai_model
MAX_PROMPT_CHARS = args.max_prompt_chars
USE_PROMPT_ENHANCEMENT = not args.no_prompt_enhancement

# Set up paths using absolute paths derived from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from USER_DIR to project root
MODEL_DIR = os.path.join(PROJECT_ROOT, "MODEL_DIR")
INPUT_DIR = os.path.join(PROJECT_ROOT, "INPUT_DIR")  # Directory to look for input images
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "OUTPUT_DIR")
MODEL_FILE = os.path.join(MODEL_DIR, "ltx-video-2b-v0.9.5.safetensors")

print("Using directories:")
print(f"  MODEL_DIR: {MODEL_DIR}")
print(f"  INPUT_DIR: {INPUT_DIR}")
print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  MODEL_FILE: {MODEL_FILE}")

# Create directories if they don't exist
Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Helper function to adjust dimensions to be divisible by 32
def adjust_dimensions(width, height):
    if width is not None:
        width = ((width - 1) // 32 + 1) * 32
    if height is not None:
        height = ((height - 1) // 32 + 1) * 32
    return width, height

# Function to calculate aspect ratio preserving dimensions
def get_adjusted_dimensions(image_path, target_width=None, target_height=None):
    # Open image and get its dimensions
    image = Image.open(image_path)
    img_width, img_height = image.size
    
    # If both dimensions are provided, just ensure they're divisible by 32
    if target_width is not None and target_height is not None:
        return adjust_dimensions(target_width, target_height)
    
    # If only one dimension is provided, calculate the other preserving aspect ratio
    if target_width is not None:
        target_width = min(target_width, MAX_WIDTH)
        target_height = int(target_width * img_height / img_width)
        target_height = min(target_height, MAX_HEIGHT)
    elif target_height is not None:
        target_height = min(target_height, MAX_HEIGHT)
        target_width = int(target_height * img_width / img_height)
        target_width = min(target_width, MAX_WIDTH)
    else:
        # If neither is provided, use the original dimensions capped at maximum
        target_width = min(img_width, MAX_WIDTH)
        target_height = min(img_height, MAX_HEIGHT)
        
        # If image is too large, scale down preserving aspect ratio
        if img_width > MAX_WIDTH:
            target_width = MAX_WIDTH
            target_height = int(MAX_WIDTH * img_height / img_width)
        if target_height > MAX_HEIGHT:
            target_height = MAX_HEIGHT
            target_width = int(MAX_HEIGHT * img_width / img_height)
    
    # Ensure dimensions are divisible by 32
    return adjust_dimensions(target_width, target_height)

# Find the most recently added image in the INPUT_DIR
image_list = glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*.png")) + glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))
if not image_list:
    print(f"No images found in {INPUT_DIR}. Please add an image file.")
    exit(1)

# Sort by modification time (most recent first)
image_list.sort(key=os.path.getmtime, reverse=True)
IMAGE_PATH = image_list[0]
print(f"Using most recent image: {IMAGE_PATH}")

# Process seed value
if SEED is None:
    seed = random.randint(0, 2147483647)
    print(f"Using random seed: {seed}")
else:
    seed = SEED
    print(f"Using fixed seed: {seed}")

# Get dimensions from input image
width, height = get_adjusted_dimensions(IMAGE_PATH, WIDTH, HEIGHT)

# Print generation information
print("Running image-to-video generation using Diffusers API with prompt enhancement...")
print(f"Input image: {IMAGE_PATH} (latest image in {INPUT_DIR})")
print(f"Original dimensions: {Image.open(IMAGE_PATH).size}")
print(f"Adjusted dimensions for generation: {width}x{height} (must be divisible by 32)")
print(f"Generating: {NUM_FRAMES} frames at {FPS} FPS ({NUM_FRAMES/FPS:.1f} seconds)")
print(f"Original prompt: '{POSITIVE_PROMPT}'")

# Enhance the prompt using OpenAI if enabled
if USE_PROMPT_ENHANCEMENT:
    try:
        print(f"Enhancing prompt using OpenAI {OPENAI_MODEL}...")
        result = analyze_and_improve_prompt(
            IMAGE_PATH,
            POSITIVE_PROMPT,
            model=OPENAI_MODEL,
            max_chars=MAX_PROMPT_CHARS
        )
        
        enhanced_prompt = result.get("enhanced_prompt", POSITIVE_PROMPT)
        print(f"Enhanced prompt: '{enhanced_prompt}'")
        print(f"Enhanced prompt length: {len(enhanced_prompt)} characters")
        
        # Use the enhanced prompt
        POSITIVE_PROMPT = enhanced_prompt
    except Exception as e:
        print(f"Error enhancing prompt: {e}")
        print("Continuing with original prompt...")
else:
    print("Prompt enhancement disabled. Using original prompt.")

# Generate a descriptive filename based on the prompt and parameters
image_filename = os.path.basename(IMAGE_PATH)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_filename = f"{os.path.splitext(image_filename)[0]}_diffusers_enhanced_{timestamp}_seed-{seed}.mp4"
output_path = os.path.join(OUTPUT_DIR, output_filename)
print(f"Output will be saved to: {output_path}")

# Set up the LTX Video Diffusion Pipeline
print("Loading LTX-Video model (this might take a minute)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    print("WARNING: CUDA not available, falling back to CPU. Generation will be extremely slow!")

# Load the input image
print(f"Loading input image: {IMAGE_PATH}")
image = Image.open(IMAGE_PATH).convert("RGB")

# Check for bfloat16 support
use_bfloat16 = False
if device == "cuda":
    if torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        use_bfloat16 = True
        print("Using bfloat16 precision (recommended by Lightricks)")
    else:
        print("bfloat16 not supported on this GPU, falling back to float16")

# Initialize the pipeline with recommended precision
pipeline = LTXImageToVideoPipeline.from_pretrained(
    MODEL_DIR,  # Use model directory instead of specific file
    torch_dtype=torch.bfloat16 if use_bfloat16 else (torch.float16 if device == "cuda" else torch.float32),
    local_files_only=True,  # Force use of local files only
)
pipeline = pipeline.to(device)

# Set the safety checker to None to bypass content filtering if available
if hasattr(pipeline, "safety_checker"):
    pipeline.safety_checker = None

# Get the generator for deterministic results
if seed is not None:
    generator = torch.Generator(device=device).manual_seed(seed)
else:
    generator = None

try:
    # Generate the video
    print("Starting generation (this may take several minutes)...")
    print(f"Guidance scale: {GUIDANCE_SCALE}")
    print(f"Inference steps: {NUM_INFERENCE_STEPS}")
    
    video_frames = pipeline(
        image=image,
        prompt=POSITIVE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).frames[0]
    
    print(f"Generated {len(video_frames)} frames. Converting to MP4...")
    
    # Save the frames as a video using matplotlib animation
    fig = plt.figure(figsize=(width/100, height/100))
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # Create frames for animation
    images = [[plt.imshow(frame, animated=True)] for frame in video_frames]
    
    # Create animation
    ani = animation.ArtistAnimation(fig, images, interval=1000/FPS, blit=True)
    
    # Save as MP4
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=FPS, metadata=dict(artist='LTX-Video'), bitrate=5000)
    
    ani.save(output_path, writer=writer)
    plt.close()
    
    print(f"Video saved to {output_path}")
    print("Generation completed! Check the outputs folder for your video.")
    
except Exception as e:
    print(f"Error during generation: {e}") 