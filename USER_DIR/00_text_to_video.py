"""
LTX-Video Text-to-Video Generator (subprocess approach)

This script generates videos from text prompts using the LTX-Video model from Lightricks.
It uses subprocess to call the inference.py script with appropriate parameters and
provides defaults optimized for quality results.

Features:
- HD resolution (1280x720)
- 20 seconds duration at 24 FPS
- Customizable prompt and generation parameters
- Supports random or fixed seeds for reproducibility
- Simple subprocess approach with minimal dependencies
- Saves output with timestamp and seed value in filename

Requirements:
- PyTorch with CUDA support
- Sufficient GPU memory for generation (at least 12GB recommended)

One-liner example:
    python USER_DIR/00_text_to_video.py
"""

# Copy-paste ready command (direct inference.py equivalent):
# python inference.py --ckpt_path MODEL_DIR/ltx-video-2b-v0.9.5.safetensors --prompt "A beautiful cinematic scene with detailed textures and dramatic lighting" --height 720 --width 1280 --num_frames 241 --seed 42 --output_path OUTPUT_DIR --device cuda --frame_rate 24 --prompt_enhancement_words_threshold 500 --num_inference_steps 40 --guidance_scale 3.0 --negative_prompt "Low resolution, inconsistent motion, visual artifacts, jitter, blur, distortion, unrealistic or misaligned frames, poor composition, dull lighting."

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

import subprocess
import os
import glob
from pathlib import Path
from datetime import datetime
import random

# One-liner for terminal:
# python USER_DIR/00_text_to_video.py
#
# Direct inference.py command equivalent:
# python inference.py --ckpt_path MODEL_DIR/ltx-video-2b-v0.9.5.safetensors --prompt "An elegantly nude woman..." --height 720 --width 1280 --num_frames 481 --seed 42 --output_path OUTPUT_DIR --device cuda --frame_rate 24 --prompt_enhancement_words_threshold 500 --num_inference_steps 40 --guidance_scale 5.0

# Set up paths using absolute paths derived from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from USER_DIR to project root
MODEL_DIR = os.path.join(PROJECT_ROOT, "MODEL_DIR")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "OUTPUT_DIR")
MODEL_FILE = os.path.join(MODEL_DIR, "ltx-video-2b-v0.9.5.safetensors")
INFERENCE_SCRIPT = os.path.join(PROJECT_ROOT, "inference.py")

print(f"Using directories:")
print(f"  MODEL_DIR: {MODEL_DIR}")
print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  MODEL_FILE: {MODEL_FILE}")
print(f"  INFERENCE_SCRIPT: {INFERENCE_SCRIPT}")

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
    "--prompt_enhancement_words_threshold", "500",  # Enable prompt enhancement
    "--num_inference_steps", str(NUM_INFERENCE_STEPS),
    "--guidance_scale", str(GUIDANCE_SCALE)
]

# Generate a descriptive filename based on the prompt and parameters
prompt_first_words = PROMPT.split()[:3]
prompt_slug = "_".join(prompt_first_words).lower()
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_filename = f"text2video_{prompt_slug}_{timestamp}_seed-{seed}.mp4"

# Print generation information
print("Running text-to-video generation...")
print(f"Using model: {MODEL_FILE}")
print(f"Dimensions: {width}x{height}")
print(f"Generating: {NUM_FRAMES} frames at {FPS} FPS ({NUM_FRAMES/FPS:.1f} seconds)")
print(f"Prompt: '{PROMPT}'")
print(f"Guidance scale: {GUIDANCE_SCALE}")
print(f"Prompt enhancement: Enabled")
print(f"Inference steps: {NUM_INFERENCE_STEPS}")
print(f"Output will be saved to: {os.path.join(OUTPUT_DIR, output_filename)}")

# Run the command
subprocess.run(cmd)

print("Generation completed! Check the outputs folder for your video.") 