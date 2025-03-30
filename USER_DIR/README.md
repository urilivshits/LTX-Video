# Custom LTX-Video Workflows

This directory contains customized workflows and utilities built upon the LTX-Video model from Lightricks.

## Available Scripts

- **Image-to-Video with Prompt Enhancement**: (`03_image_to_video_diffusers.py`)
  - Generates video from a single image using the diffusers library.
  - Optionally enhances the user prompt using OpenAI's API (GPT-4o-mini by default) by analyzing the input image content.
  - Finds the latest image in `INPUT_DIR` automatically.
  - Saves output to `OUTPUT_DIR`.

- **Video Expander with Advanced Options**: (`04_video_expander_diffusers.py`)
  - Creates longer videos by chaining multiple generation segments.
  - Uses the middle or last frame of the previous segment as input for the next.
  - Optional regression towards the original image to maintain subject consistency.
  - Supports segment-specific prompts for narrative control.
  - Optionally enhances prompts using OpenAI API for the first segment or all segments.
  - Combines segments with crossfade transitions.
  - Saves individual segments and the final combined video.

- **Prompt Improvement Utility**: (`prompt_improvement.py`)
  - Standalone module providing functions to analyze images and enhance prompts using OpenAI.
  - Supports different OpenAI models and enforces character limits with word-aware truncation.

- **Basic Text-to-Video**: (`01_text_to_video_diffusers.py`)
  - Simple text-to-video generation using the diffusers library.

- **Legacy Text-to-Video**: (`00_text_to_video.py`)
  - Older version using `subprocess` to call `inference.py`. Retained for reference.

- **Legacy Image-to-Video**: (`02_image_to_video.py`)
  - Older version using `subprocess` to call `inference.py`. Retained for reference.

- **Dependency Checker**: (`check_dependencies.py`)
  - Utility to verify necessary libraries are installed.

- **Prompt Examples**: (`prompt_examples.json`)
  - Collection of example prompts (not directly used by scripts).

## Setup

1.  **Environment**: Ensure you have the required Python environment set up as described in the main project `README.md`.
2.  **Dependencies**: Install all dependencies from the project root directory:
    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure you have the latest version of `diffusers` for best compatibility: `pip install --upgrade diffusers`)*
3.  **Model**: Download the LTX-Video model into the `MODEL_DIR` folder (see main project `README.md`).
4.  **OpenAI API Key**: If using prompt enhancement features, create a `.env` file in the project root directory (one level above `USER_DIR`) and add your key:
    ```
    OPENAI_API_KEY=<your_actual_api_key>
    ```

## Running the Scripts

All scripts should be run from the **project root directory**, not from inside `USER_DIR`. This ensures that relative paths to `MODEL_DIR`, `INPUT_DIR`, `OUTPUT_DIR`, and `.env` are resolved correctly.

```bash
# Correct way (from project root):
python USER_DIR/03_image_to_video_diffusers.py --prompt "A person blinking naturally"

# Incorrect way (will likely cause path errors):
# cd USER_DIR
# python 03_image_to_video_diffusers.py --prompt "A person blinking naturally"
```

## Usage Examples

### Image-to-Video (with optional enhancement - `03_...`)

Place your input image in the `INPUT_DIR` folder at the project root. The script uses the most recently modified image.

```bash
# Basic usage (enhancement enabled by default)
python USER_DIR/03_image_to_video_diffusers.py --prompt "A person blinking naturally"

# Disable prompt enhancement
python USER_DIR/03_image_to_video_diffusers.py --prompt "A person blinking naturally" --no_prompt_enhancement

# Use a different OpenAI model
python USER_DIR/03_image_to_video_diffusers.py --prompt "A person blinking naturally" --openai_model gpt-4-turbo
```

### Video Expander (`04_...`)

Place your initial input image in the `INPUT_DIR` folder at the project root.

```bash
# Basic expansion (enhances first prompt by default)
python USER_DIR/04_video_expander_diffusers.py --prompt "A person looking around" --num_expansions 2

# Enhance prompts for all segments
python USER_DIR/04_video_expander_diffusers.py --prompt "A person looking around" --num_expansions 2 --enhance_all_segments

# Use segment-specific prompts (disables automatic enhancement unless --enhance_all_segments is used)
python USER_DIR/04_video_expander_diffusers.py --segment_prompts "Segment 1 prompt,Segment 2 prompt,Segment 3 prompt" --num_expansions 2

# Adjust subject consistency
python USER_DIR/04_video_expander_diffusers.py --prompt "A person looking around" --num_expansions 2 --regression_strength 0.4 --use_middle_frame
```

### Basic Text-to-Video (`01_...`)

```bash
python USER_DIR/01_text_to_video_diffusers.py --prompt "A cinematic scene description"
```

### Command Line Arguments

Refer to the `--help` option for each script for a full list of arguments:

```bash
python USER_DIR/01_text_to_video_diffusers.py --help
python USER_DIR/03_image_to_video_diffusers.py --help
python USER_DIR/04_video_expander_diffusers.py --help
# etc. for other scripts if needed
```

## Key Parameters (Common across scripts)

- `--prompt`: Initial text prompt.
- `--width`, `--height`: Video dimensions (must be divisible by 32). If `None` for image-based scripts, auto-detected from image.
- `--num_frames`: Number of frames to generate (recommend multiple of 8 + 1).
- `--fps`: Output video frames per second.
- `--seed`: Integer for reproducible results, `None` for random.
- `--guidance_scale`: How strictly to follow the prompt (e.g., 1.0-5.0).
- `--inference_steps`: Number of denoising steps (e.g., 20-40).
- `--negative_prompt`: Describe concepts to avoid.

### Prompt Enhancement Parameters (`03_...` & `04_...`)

- `--openai_model`: OpenAI model for enhancement (e.g., `gpt-4o-mini`, `gpt-4-turbo`).
- `--no_prompt_enhancement`: Disable OpenAI enhancement.
- `--max_prompt_chars`: Max characters for enhanced prompts (default 400).

### Video Expander Parameters (`04_...` only)

- `--enhance_all_segments`: Enhance prompts for every segment, not just the first.
- `--segment_prompts`: Comma-separated prompts for each segment.
- `--num_expansions`: Number of times to extend the video (total segments = 1 + num_expansions).
- `--regression_strength`: Blend subsequent frames with the original image (0.0-1.0) for consistency.
- `--use_middle_frame`: Use the middle frame (vs. last frame) of the previous segment as input for the next.

## Input Images (for `03_...` & `04_...`)

1. Place images in the `INPUT_DIR` folder (at the project root).
2. The scripts will automatically use the **most recently modified** image in that folder.
3. Supported formats: JPG, JPEG, PNG, BMP, WEBP.
4. Images are automatically resized/adjusted for model compatibility (divisible by 32, within max dimensions).

## Troubleshooting

- **Import Errors**: Ensure `diffusers` and other dependencies in `requirements.txt` are installed and up-to-date (`pip install --upgrade diffusers`).
- **Path Errors**: Always run scripts from the **project root directory**, one level above `USER_DIR`.
- **CUDA/Memory Errors**: Ensure you have a compatible GPU and sufficient VRAM. Try reducing `--width`, `--height`, or `--num_frames`. The scripts will use CPU if CUDA isn't found, but it will be very slow.
- **OpenAI Errors**: Ensure your API key is correct in the `.env` file at the project root and that the specified `--openai_model` is valid. 