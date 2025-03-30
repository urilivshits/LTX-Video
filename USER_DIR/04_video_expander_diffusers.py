"""
LTX-Video Expander (diffusers API approach)

This script creates longer videos by chaining multiple generations together using the LTX-Video model.
It first generates a video from an input image, then uses the last frame of that video as input
for the next generation, and repeats this process to create an extended video.

Features:
- Supports multiple expansion iterations to create longer videos
- Maintains consistency between segments by using the last frame of each segment
- Saves both individual segments and the combined expanded video
- Customizable prompt and generation parameters
- Direct Diffusers API integration

Requirements:
- PyTorch with CUDA support
- diffusers library
- Sufficient GPU memory for generation (at least 12GB recommended)

One-liner example:
    python USER_DIR/04_video_expander_diffusers.py --num_expansions 3
"""

# Copy-paste ready command:
# python USER_DIR/04_video_expander_diffusers.py --prompt "A beautiful cinematic animation" --width 576 --height 320 --num_frames 81 --num_expansions 3 --seed 42 --guidance_scale 3.0 --inference_steps 30 --fps 24

# ============ USER CONFIGURABLE PARAMETERS ============
# Set these values before running the script

# Video dimensions - set to None to auto-detect from input image
# If only one is set, the other will be calculated maintaining aspect ratio
WIDTH = 320                   # Width in pixels (must be divisible by 32, None for auto-detection from input image)
HEIGHT = None                  # Height in pixels (must be divisible by 32, None for auto-detection from input image)

# Maximum dimensions if auto-detecting (720p)
MAX_WIDTH = 1280              # Maximum width if auto-detecting dimensions (to prevent memory issues)
MAX_HEIGHT = 720              # Maximum height if auto-detecting dimensions (to prevent memory issues)

# Generation parameters
FPS = 24                      # Frames per second for output video (standard film is 24, higher values create smoother motion)
NUM_FRAMES = 121              # Number of frames per segment (must be multiple of 8 + 1, e.g. 9, 17, 25, etc.)
NUM_INFERENCE_STEPS = 20      # Number of denoising steps (higher values = better quality but slower; 20-40 is typical range)
NUM_EXPANSIONS = 3            # Number of expansions to perform (total segments = 1 + NUM_EXPANSIONS)

# Subject consistency parameters
REGRESSION_STRENGTH = 0.5    # Blend with original image (0.0-1.0, 0.0=disabled, 0.35=recommended). Higher values keep the subject more similar to the original but may reduce motion variety.

USE_MIDDLE_FRAME = False       # Use middle frame for next segment instead of last frame. Middle frames typically have better subject quality than end frames which often degrade.

SEGMENT_PROMPTS = None        # Comma-separated prompts for each segment, e.g.: "A woman walking,The woman sits down,The woman stands up". This allows custom prompts for each segment to guide the narrative progression.

GUIDANCE_SCALE = 4          # Controls how closely the model follows your prompt (1.0-5.0). 3.5-4.0 recommended for better detail, lower values are more creative.

# OpenAI prompt enhancement settings
USE_PROMPT_ENHANCEMENT = True # Enable OpenAI prompt enhancement for initial image
OPENAI_MODEL = "gpt-4o-mini"  # Can be changed to other OpenAI models
MAX_PROMPT_CHARS = 400        # Maximum characters for enhanced prompt
ENHANCE_ALL_SEGMENTS = True  # Whether to enhance prompts for all segments or just the first one

# Set to None for random seed each time
SEED = None             # Fixed seed ensures reproducible results, None generates a random seed each time. Using the same seed with identical parameters will generate the same video.

# Prompt parameters
POSITIVE_PROMPT = ""

NEGATIVE_PROMPT = "Low resolution, inconsistent motion, visual artifacts, jitter, blur, distortion, unnatural movement, poor composition, unbalanced or dull lighting, unrealistic anatomy, awkward proportions, incoherent shading or motion."

# =====================================================

import os
import torch
import random
import glob
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import tempfile
import cv2
import hashlib
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
parser = argparse.ArgumentParser(description='LTX-Video Expander')
parser.add_argument('--width', type=int, default=WIDTH, help='Video width (divisible by 32)')
parser.add_argument('--height', type=int, default=HEIGHT, help='Video height (divisible by 32)')
parser.add_argument('--num_frames', type=int, default=NUM_FRAMES, help='Number of frames per segment (multiple of 8 + 1)')
parser.add_argument('--num_expansions', type=int, default=NUM_EXPANSIONS, help='Number of expansion iterations')
parser.add_argument('--fps', type=int, default=FPS, help='Frames per second')
parser.add_argument('--seed', type=int, default=SEED, help='Random seed (None for random)')
parser.add_argument('--prompt', type=str, default=POSITIVE_PROMPT, help='Positive prompt')
parser.add_argument('--negative_prompt', type=str, default=NEGATIVE_PROMPT, help='Negative prompt')
parser.add_argument('--guidance_scale', type=float, default=GUIDANCE_SCALE, help='Guidance scale (1.0-5.0)')
parser.add_argument('--inference_steps', type=int, default=NUM_INFERENCE_STEPS, help='Number of inference steps')
parser.add_argument('--regression_strength', type=float, default=REGRESSION_STRENGTH, help='How much to blend the last frame with the original image (0.0-1.0)')
parser.add_argument('--use_middle_frame', action='store_true', default=USE_MIDDLE_FRAME, help='Use middle frame instead of last frame for next segment')
parser.add_argument('--segment_prompts', type=str, default=SEGMENT_PROMPTS, help='Comma-separated prompts for each segment (overrides main prompt)')
parser.add_argument('--openai_model', type=str, default=OPENAI_MODEL, help='OpenAI model for prompt enhancement')
parser.add_argument('--max_prompt_chars', type=int, default=MAX_PROMPT_CHARS, help='Max characters for enhanced prompt')
parser.add_argument('--no_prompt_enhancement', action='store_true', help='Disable OpenAI prompt enhancement')
parser.add_argument('--enhance_all_segments', action='store_true', default=ENHANCE_ALL_SEGMENTS, help='Enhance prompts for all segments, not just the first')
args = parser.parse_args()

# Update parameters from arguments
WIDTH = args.width
HEIGHT = args.height
NUM_FRAMES = args.num_frames
NUM_EXPANSIONS = args.num_expansions
FPS = args.fps
SEED = args.seed
POSITIVE_PROMPT = args.prompt
NEGATIVE_PROMPT = args.negative_prompt
GUIDANCE_SCALE = args.guidance_scale
NUM_INFERENCE_STEPS = args.inference_steps
REGRESSION_STRENGTH = args.regression_strength
USE_MIDDLE_FRAME = args.use_middle_frame
SEGMENT_PROMPTS = args.segment_prompts.split(',') if args.segment_prompts else None
OPENAI_MODEL = args.openai_model
MAX_PROMPT_CHARS = args.max_prompt_chars
USE_PROMPT_ENHANCEMENT = not args.no_prompt_enhancement
ENHANCE_ALL_SEGMENTS = args.enhance_all_segments

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

# Get dimensions from input image and adjust to be divisible by 32
def get_adjusted_dimensions(image_path=None, img=None, target_width=None, target_height=None, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """
    Adjust image dimensions to be divisible by 32 and scale down if needed.
    
    The LTX-Video model requires dimensions to be divisible by 32.
    This function also scales down large images to max resolution while maintaining aspect ratio.
    
    Args:
        image_path (str, optional): Path to the input image
        img (PIL.Image, optional): Image object if already loaded
        target_width (int, optional): Specific width to use (will maintain aspect ratio if only one dimension specified)
        target_height (int, optional): Specific height to use (will maintain aspect ratio if only one dimension specified)
        max_width (int): Maximum width if auto-detecting
        max_height (int): Maximum height if auto-detecting
        
    Returns:
        tuple: (width, height) adjusted to be divisible by 32
    """
    if img is None and image_path is not None:
        with Image.open(image_path) as img:
            orig_width, orig_height = img.size
    elif img is not None:
        orig_width, orig_height = img.size
    else:
        raise ValueError("Either image_path or img must be provided")
    
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
pipeline = LTXImageToVideoPipeline.from_pretrained(
    MODEL_DIR,  # Use model directory instead of specific file
    torch_dtype=torch.bfloat16 if use_bfloat16 else (torch.float16 if device == "cuda" else torch.float32),
    local_files_only=True,  # Force use of local files only
)
pipeline = pipeline.to(device)

# Set the safety checker to None to bypass content filtering if available
if hasattr(pipeline, "safety_checker"):
    pipeline.safety_checker = None

def save_video_frames(frames, output_path, fps=24, add_fade=False, fade_frames=8):
    """
    Save a list of frames as a video file.
    
    Args:
        frames (list): List of numpy arrays or PIL Images representing video frames
        output_path (str): Path to save the video
        fps (int): Frames per second
        add_fade (bool): Whether to add a fade-out effect to the end
        fade_frames (int): Number of frames for fade effect
    """
    # Apply fade out if requested
    if add_fade and len(frames) > fade_frames:
        for i in range(fade_frames):
            alpha = 1.0 - (i / fade_frames)
            frame_idx = len(frames) - fade_frames + i
            if isinstance(frames[frame_idx], np.ndarray):
                frames[frame_idx] = (frames[frame_idx] * alpha).astype(np.uint8)
            else:
                # For PIL Images, convert to array, apply fade, and convert back
                frame_array = np.array(frames[frame_idx])
                frame_array = (frame_array * alpha).astype(np.uint8)
                frames[frame_idx] = Image.fromarray(frame_array)
    
    # Create a temporary directory for the frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save all frames as images
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            if isinstance(frame, np.ndarray):
                Image.fromarray(frame).save(frame_path)
            else:
                # If it's already a PIL Image, save directly
                frame.save(frame_path)
            frame_paths.append(frame_path)
        
        # Use OpenCV to create video
        first_frame = cv2.imread(frame_paths[0])
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            video.write(frame)
        
        video.release()
    
    print(f"Video saved to {output_path}")

def save_frame_as_image(frame, output_path):
    """
    Save a single frame as an image.
    
    Args:
        frame (numpy array or PIL Image): The frame to save
        output_path (str): Path to save the image
    """
    if isinstance(frame, np.ndarray):
        Image.fromarray(frame).save(output_path)
    else:
        # If it's already a PIL Image, save directly
        frame.save(output_path)
    print(f"Frame saved to {output_path}")

def concatenate_videos(video_paths, output_path, fps=24, crossfade_frames=8):
    """
    Concatenate multiple videos into a single video with crossfade transitions.
    
    Args:
        video_paths (list): List of paths to the videos to concatenate
        output_path (str): Path to save the concatenated video
        fps (int): Frames per second
        crossfade_frames (int): Number of frames for crossfade transitions
    """
    all_frames = []
    
    # Read all videos
    for i, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Skip empty videos
        if not frames:
            print(f"Warning: Video {video_path} is empty, skipping")
            continue
        
        # For videos after the first one, create a crossfade with the previous video
        if i > 0 and crossfade_frames > 0 and len(all_frames) >= crossfade_frames and len(frames) >= crossfade_frames:
            # Replace the last few frames with crossfade
            for j in range(crossfade_frames):
                alpha = j / crossfade_frames
                crossfade_frame = cv2.addWeighted(
                    all_frames[-(crossfade_frames-j)], 1-alpha, 
                    frames[j], alpha, 0
                )
                all_frames[-(crossfade_frames-j)] = crossfade_frame
            
            # Add the rest of the frames from the current video
            all_frames.extend(frames[crossfade_frames:])
        else:
            # For the first video, just add all frames
            all_frames.extend(frames)
    
    # If no frames were added, return
    if not all_frames:
        print("Error: No frames to concatenate")
        return
    
    # Save the concatenated video
    height, width, _ = all_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in all_frames:
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    
    video.release()
    print(f"Concatenated video saved to {output_path}")

def blend_images(img1, img2, alpha):
    """
    Blend two images together using the specified alpha value.
    
    Args:
        img1 (PIL.Image): First image
        img2 (PIL.Image): Second image
        alpha (float): Blend factor (0.0 = 100% img1, 1.0 = 100% img2)
        
    Returns:
        PIL.Image: Blended image
    """
    # Ensure both images are the same size
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)
        
    # Convert to numpy arrays
    arr1 = np.array(img1).astype(float)
    arr2 = np.array(img2).astype(float)
    
    # Blend arrays
    blended_arr = arr1 * (1 - alpha) + arr2 * alpha
    
    # Convert back to PIL Image
    return Image.fromarray(blended_arr.astype(np.uint8))

# Get dimensions from input image
width, height = get_adjusted_dimensions(IMAGE_PATH, target_width=WIDTH, target_height=HEIGHT)

# Verify frame count is valid (multiple of 8 + 1)
if (NUM_FRAMES - 1) % 8 != 0:
    original_num_frames = NUM_FRAMES
    NUM_FRAMES = ((NUM_FRAMES - 1) // 8) * 8 + 1
    print(f"Warning: Adjusted frame count from {original_num_frames} to {NUM_FRAMES} (must be multiple of 8 + 1)")

# Print generation information
print("\n=== LTX-Video Expander Configuration ===")
print(f"Initial input image: {IMAGE_PATH}")
print(f"Video dimensions: {width}x{height}")
print(f"Segment length: {NUM_FRAMES} frames ({NUM_FRAMES/FPS:.1f} seconds)")
print(f"Number of expansions: {NUM_EXPANSIONS}")
print(f"Total video length: {(NUM_EXPANSIONS + 1) * NUM_FRAMES/FPS:.1f} seconds")
print(f"Prompt: '{POSITIVE_PROMPT}'")
print(f"Guidance scale: {GUIDANCE_SCALE}")
print(f"Inference steps: {NUM_INFERENCE_STEPS}")
print(f"Frame rate: {FPS} FPS")
print(f"Seed: {seed}")
print(f"Regression strength: {REGRESSION_STRENGTH}")
print(f"Using middle frame: {USE_MIDDLE_FRAME}")
print(f"Prompt enhancement: {'Enabled' if USE_PROMPT_ENHANCEMENT else 'Disabled'}")
if USE_PROMPT_ENHANCEMENT:
    print(f"OpenAI model: {OPENAI_MODEL}")
    print(f"Enhance all segments: {ENHANCE_ALL_SEGMENTS}")
if SEGMENT_PROMPTS:
    print("Using different prompts for each segment")
print("=========================================\n")

# Generate timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
base_filename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
segment_dir = os.path.join(OUTPUT_DIR, f"{base_filename}_expansion_{timestamp}")
os.makedirs(segment_dir, exist_ok=True)

# List to store paths to all segment videos
segment_video_paths = []

try:
    # Create base generator for first segment
    base_generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    
    # Load the initial input image
    original_image = Image.open(IMAGE_PATH).convert("RGB")
    current_image = original_image
    current_image_path = IMAGE_PATH
    
    # Generate multiple video segments
    for segment_idx in range(NUM_EXPANSIONS + 1):
        print(f"\n=== Generating segment {segment_idx+1}/{NUM_EXPANSIONS+1} ===")
        
        # Set up segment-specific seed
        if segment_idx == 0:
            # First segment uses the base seed
            segment_seed = seed
            generator = base_generator
            print(f"Using base seed for first segment: {segment_seed}")
        else:
            # Subsequent segments derive seed from base seed and last frame hash
            img_array = np.array(current_image)
            # Get hash of the image and convert to integer
            frame_hash = int(hashlib.md5(img_array.tobytes()).hexdigest(), 16) % 10**8
            # Combine base seed with frame hash using XOR
            segment_seed = seed ^ frame_hash
            generator = torch.Generator(device=device).manual_seed(segment_seed)
            print(f"Using derived seed for segment {segment_idx+1}: {segment_seed} (base seed: {seed}, frame hash: {frame_hash})")
        
        # Get segment-specific prompt if provided
        if SEGMENT_PROMPTS and segment_idx < len(SEGMENT_PROMPTS):
            segment_prompt = SEGMENT_PROMPTS[segment_idx]
            print(f"Using segment-specific prompt: '{segment_prompt}'")
        else:
            segment_prompt = POSITIVE_PROMPT
        
        # Enhance the prompt using OpenAI if enabled
        # Only enhance first segment by default, or all segments if ENHANCE_ALL_SEGMENTS is True
        if USE_PROMPT_ENHANCEMENT and (segment_idx == 0 or ENHANCE_ALL_SEGMENTS):
            try:
                print(f"Enhancing prompt for segment {segment_idx+1} using OpenAI {OPENAI_MODEL}...")
                result = analyze_and_improve_prompt(
                    current_image_path,
                    segment_prompt,
                    model=OPENAI_MODEL,
                    max_chars=MAX_PROMPT_CHARS
                )
                
                enhanced_prompt = result.get("enhanced_prompt", segment_prompt)
                print(f"Enhanced prompt: '{enhanced_prompt}'")
                print(f"Enhanced prompt length: {len(enhanced_prompt)} characters")
                
                # Use the enhanced prompt
                segment_prompt = enhanced_prompt
                
                # If this is the first segment and there are no segment-specific prompts,
                # update POSITIVE_PROMPT for logging purposes and for any segments that don't have
                # their own specified prompt
                if segment_idx == 0 and not SEGMENT_PROMPTS:
                    POSITIVE_PROMPT = enhanced_prompt
            except Exception as e:
                print(f"Error enhancing prompt: {e}")
                print("Continuing with original prompt...")
        
        if segment_idx > 0:
            print(f"Using {'middle' if USE_MIDDLE_FRAME else 'last'} frame from previous segment as input")
        else:
            print(f"Using initial image: {IMAGE_PATH}")
            
        # Log the final prompt being used for this segment
        print(f"\nFINAL PROMPT FOR SEGMENT {segment_idx+1}: '{segment_prompt}'")
        print(f"Prompt length: {len(segment_prompt)} characters")
        
        # Generate the video segment
        print(f"Starting generation with {NUM_INFERENCE_STEPS} inference steps...")
        video_frames = pipeline(
            image=current_image,
            prompt=segment_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=width,
            height=height,
            num_frames=NUM_FRAMES,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        ).frames[0]
        
        print(f"Generated {len(video_frames)} frames for segment {segment_idx+1}")
        
        # Save the video segment
        segment_filename = f"segment_{segment_idx+1:02d}_seed-{segment_seed}.mp4"
        segment_path = os.path.join(segment_dir, segment_filename)
        save_video_frames(video_frames, segment_path, fps=FPS)
        segment_video_paths.append(segment_path)
        
        # Save the frame to use as input for the next segment
        if segment_idx < NUM_EXPANSIONS:
            # Choose which frame to use as the basis for the next segment
            if USE_MIDDLE_FRAME:
                selected_frame_idx = len(video_frames) // 2
                frame_type = "middle"
            else:
                selected_frame_idx = -1  # Last frame
                frame_type = "last"
            
            next_input_frame = video_frames[selected_frame_idx]
            
            # Apply regression toward original image if specified
            if REGRESSION_STRENGTH > 0:
                print(f"Applying {REGRESSION_STRENGTH:.2f} regression toward original image")
                if isinstance(next_input_frame, np.ndarray):
                    next_input_frame_img = Image.fromarray(next_input_frame)
                else:
                    next_input_frame_img = next_input_frame
                
                # Blend the frame with the original image based on regression strength
                next_input_frame = blend_images(next_input_frame_img, original_image, REGRESSION_STRENGTH)
            
            # Save the frame
            next_frame_filename = f"{frame_type}_frame_segment_{segment_idx+1:02d}.png"
            next_frame_path = os.path.join(segment_dir, next_frame_filename)
            save_frame_as_image(next_input_frame, next_frame_path)
            
            # Update current image for next iteration
            if isinstance(next_input_frame, np.ndarray):
                current_image = Image.fromarray(next_input_frame)
            else:
                current_image = next_input_frame
            current_image_path = next_frame_path
    
    # Generate the combined video
    if len(segment_video_paths) > 1:
        print("\n=== Creating combined video with all segments ===")
        combined_filename = f"{base_filename}_combined_{timestamp}_seed-{seed}.mp4"
        combined_path = os.path.join(OUTPUT_DIR, combined_filename)
        concatenate_videos(segment_video_paths, combined_path, fps=FPS)
        
        print("\nAll processing complete!")
        print(f"Individual segments saved in: {segment_dir}")
        print(f"Combined video saved to: {combined_path}")
    else:
        print("\nAll processing complete!")
        print(f"Video saved to: {segment_video_paths[0]}")
        
except Exception as e:
    print(f"Error during generation: {e}")
    import traceback
    traceback.print_exc()
    print("\nTip: If you're seeing CUDA out of memory errors, try reducing the resolution or number of frames.") 