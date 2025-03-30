"""
LTX-Video Text-to-Video Generator (diffusers API approach)

This script generates videos from text prompts using the LTX-Video model from Lightricks.
It uses the Diffusers API for direct integration instead of calling inference.py
and provides defaults optimized for quality results.

Features:
- HD resolution (1280x720)
- 20 seconds duration at 24 FPS
- Customizable prompt and generation parameters
- Supports random or fixed seeds for reproducibility
- Direct Diffusers API integration for better performance
- Saves output with timestamp and seed value in filename

Requirements:
- PyTorch with CUDA support
- diffusers library
- Sufficient GPU memory for generation (at least 12GB recommended)

One-liner example:
    python USER_DIR/01_text_to_video_diffusers.py
"""

# Copy-paste ready command:
# python USER_DIR/01_text_to_video_diffusers.py --prompt "A beautiful cinematic scene with detailed textures and dramatic lighting" --width 1280 --height 720 --num_frames 241 --seed 42 --guidance_scale 3.0 --inference_steps 40 --fps 24

# ============ USER CONFIGURABLE PARAMETERS ============
# Set these values before running the script

# Video dimensions - both must be divisible by 32
WIDTH = 1280                 # Width in pixels
HEIGHT = 720                 # Height in pixels

# Generation parameters
FPS = 24                     # Frames per second
NUM_FRAMES = 241             # 10 seconds at 24 FPS (multiple of 8 + 1)
NUM_INFERENCE_STEPS = 40     # Number of denoising steps

# Set to None for random seed each time
SEED = None

# Prompt parameters
PROMPT = "A beautiful cinematic scene with detailed textures and dramatic lighting. The motion is smooth and fluid, with a dynamic camera that enhances the visual storytelling."
NEGATIVE_PROMPT = "Low resolution, inconsistent motion, visual artifacts, jitter, blur, distortion, unrealistic or misaligned frames, poor composition, dull lighting."
GUIDANCE_SCALE = 3.0         # 1.0-5.0, higher values follow prompt more closely
# =====================================================

import os
import torch
import inspect
import random
from pathlib import Path
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import animation

# For diffusers support
try:
    # Try importing directly (newer versions of diffusers)
    from diffusers import LTXTextToVideoPipeline
    print("Using standard LTXTextToVideoPipeline import")
except ImportError:
    # If that fails, ask user to update
    print("ERROR: Could not import LTXTextToVideoPipeline. Please update your diffusers library:")
    print("pip install --upgrade diffusers")
    exit(1)

# One-liner for terminal:
# python USER_DIR/01_text_to_video_diffusers.py
#
# This uses the Diffusers API directly (no inference.py call)

# Set up paths using absolute paths derived from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from USER_DIR to project root
MODEL_DIR = os.path.join(PROJECT_ROOT, "MODEL_DIR")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "OUTPUT_DIR")
MODEL_FILE = os.path.join(MODEL_DIR, "ltx-video-2b-v0.9.5.safetensors")

print(f"Using directories:")
print(f"  MODEL_DIR: {MODEL_DIR}")
print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  MODEL_FILE: {MODEL_FILE}")

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

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

# Process seed value
if SEED is None:
    seed = random.randint(0, 2147483647)
    print(f"Using random seed: {seed}")
else:
    seed = SEED
    print(f"Using fixed seed: {seed}")

# Check that dimensions are divisible by 32 (required by the model)
width = WIDTH
height = HEIGHT
if width % 32 != 0 or height % 32 != 0:
    print(f"Warning: Dimensions must be divisible by 32. Adjusting...")
    width = ((width - 1) // 32 + 1) * 32
    height = ((height - 1) // 32 + 1) * 32
    print(f"Adjusted dimensions: {width}x{height}")

# Print generation information
print("Running text-to-video generation using Diffusers API...")
print(f"Dimensions: {width}x{height}")
print(f"Generating: {NUM_FRAMES} frames at {FPS} FPS ({NUM_FRAMES/FPS:.1f} seconds)")
print(f"Prompt: '{PROMPT}'")
print(f"Guidance scale: {GUIDANCE_SCALE}")
print(f"Prompt enhancement: Enabled")
print(f"Inference steps: {NUM_INFERENCE_STEPS}")

# Generate a descriptive filename based on the prompt and parameters
prompt_first_words = PROMPT.split()[:3]
prompt_slug = "_".join(prompt_first_words).lower()
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_filename = f"text2video_diffusers_{prompt_slug}_{timestamp}_seed-{seed}.mp4"
output_path = os.path.join(OUTPUT_DIR, output_filename)
print(f"Output will be saved to: {output_path}")

# Set up the LTX Video Diffusion Pipeline
print("Loading LTX-Video model (this might take a minute)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    print("WARNING: CUDA not available, falling back to CPU. Generation will be extremely slow!")

# Check for bfloat16 support
use_bfloat16 = False
if device == "cuda":
    if torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        use_bfloat16 = True
        print("Using bfloat16 precision (recommended by Lightricks)")
    else:
        print("bfloat16 not supported on this GPU, falling back to float16")

# Initialize the pipeline with recommended precision
pipeline = LTXTextToVideoPipeline.from_pretrained(
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
    # Generate the video frames
    print("Starting generation (this may take several minutes)...")
    video_frames = pipeline(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        width=width,
        height=height,
        num_frames=NUM_FRAMES,
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
    # Print more detailed exception information
    import traceback
    traceback.print_exc()
    print("\nTip: If you're seeing CUDA out of memory errors, try reducing the resolution or number of frames.") 