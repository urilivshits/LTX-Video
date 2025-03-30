#!/usr/bin/env python
"""
LTX-Video Prompt Improvement Feature

This script provides a custom prompt improvement function using OpenAI's GPT models
for the LTX-Video image-to-video workflow. It analyzes the input image and enhances 
the user prompt to create better image-to-video generations.

Features:
- Uses OpenAI API (defaults to gpt-4o-mini but can be swapped to other models)
- Analyzes input images to describe visual content
- Enhances user prompts based on image analysis
- Outputs improved prompts in JSON format
- Limits output to 400 characters
"""

import os
import base64
import json
import argparse
from typing import Dict, Any, Optional, Tuple
from io import BytesIO
from PIL import Image
import requests
from dotenv import load_dotenv

# Load environment variables from .env file (for API key)
load_dotenv()

# OpenAI API configuration - never hardcode the key in the source code
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # API key should only be loaded from environment variables
DEFAULT_MODEL = "gpt-4o-mini"  # Can be changed to any OpenAI model


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image to base64 encoding for API transmission.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to base64 encoding.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string of the image
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def analyze_and_improve_prompt(
    image_path: str, 
    user_prompt: str, 
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_chars: int = 400
) -> Dict[str, str]:
    """Analyze an image and improve a user prompt using OpenAI API.
    
    Args:
        image_path: Path to the input image
        user_prompt: Original user prompt to enhance
        model: OpenAI model to use
        api_key: OpenAI API key (will use env var if None)
        max_chars: Maximum characters for the enhanced prompt
        
    Returns:
        Dictionary with the enhanced prompt and other metadata
    """
    # Use provided API key or fall back to environment variable
    api_key = api_key or OPENAI_API_KEY
    
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass as parameter.")
    
    # Prepare the image
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": f"""You are a specialist in analyzing images and creating detailed prompts for image-to-video generation models.
                Follow these instructions carefully:
                
                1. Analyze the provided image and describe precisely what's in it
                2. For people: describe pose, facial expressions, clothing, and appearance details
                3. For scenes: describe key visual elements, lighting, atmosphere, and composition
                4. Combine your image analysis with the user's prompt to create an enhanced prompt
                5. The enhanced prompt should be comprehensive yet concise, focusing on motion possibilities
                6. Your output MUST be a valid JSON object containing ONLY the "enhanced_prompt" field
                7. The enhanced prompt MUST NOT exceed {max_chars} characters in length
                8. The enhanced prompt should incorporate both what's visible in the image and the user's intentions
                
                Return ONLY a JSON object like: {{"enhanced_prompt": "your enhanced prompt here"}}
                """
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Here is my original prompt: '{user_prompt}'. Please analyze the image and create an enhanced prompt."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    # Make the API request
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    # Process the response
    if response.status_code != 200:
        raise Exception(f"OpenAI API request failed with status code {response.status_code}: {response.text}")
    
    try:
        response_data = response.json()
        result = json.loads(response_data["choices"][0]["message"]["content"])
        
        # Ensure the prompt doesn't exceed max_chars
        if len(result.get("enhanced_prompt", "")) > max_chars:
            enhanced_prompt = result.get("enhanced_prompt", "")
            # Truncate at the last complete word within the character limit
            if " " in enhanced_prompt[:max_chars]:
                last_space = enhanced_prompt[:max_chars].rstrip().rfind(" ")
                result["enhanced_prompt"] = enhanced_prompt[:last_space]
            else:
                # If no spaces found, just truncate at max_chars
                result["enhanced_prompt"] = enhanced_prompt[:max_chars]
        
        return result
    except Exception as e:
        raise Exception(f"Failed to parse OpenAI response: {e}")


def analyze_and_improve_prompt_from_image_object(
    image: Image.Image, 
    user_prompt: str, 
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_chars: int = 400
) -> Dict[str, str]:
    """Analyze a PIL Image object and improve a user prompt using OpenAI API.
    
    Args:
        image: PIL Image object
        user_prompt: Original user prompt to enhance
        model: OpenAI model to use
        api_key: OpenAI API key (will use env var if None)
        max_chars: Maximum characters for the enhanced prompt
        
    Returns:
        Dictionary with the enhanced prompt and other metadata
    """
    # Use provided API key or fall back to environment variable
    api_key = api_key or OPENAI_API_KEY
    
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass as parameter.")
    
    # Prepare the image
    base64_image = image_to_base64(image)
    
    # Prepare the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": f"""You are a specialist in analyzing images and creating detailed prompts for image-to-video generation models.
                Follow these instructions carefully:
                
                1. Analyze the provided image and describe precisely what's in it
                2. For people: describe pose, facial expressions, clothing, and appearance details
                3. For scenes: describe key visual elements, lighting, atmosphere, and composition
                4. Combine your image analysis with the user's prompt to create an enhanced prompt
                5. The enhanced prompt should be comprehensive yet concise, focusing on motion possibilities
                6. Your output MUST be a valid JSON object containing ONLY the "enhanced_prompt" field
                7. The enhanced prompt MUST NOT exceed {max_chars} characters in length
                8. The enhanced prompt should incorporate both what's visible in the image and the user's intentions
                
                Return ONLY a JSON object like: {{"enhanced_prompt": "your enhanced prompt here"}}
                """
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Here is my original prompt: '{user_prompt}'. Please analyze the image and create an enhanced prompt."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    # Make the API request
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    # Process the response
    if response.status_code != 200:
        raise Exception(f"OpenAI API request failed with status code {response.status_code}: {response.text}")
    
    try:
        response_data = response.json()
        result = json.loads(response_data["choices"][0]["message"]["content"])
        
        # Ensure the prompt doesn't exceed max_chars
        if len(result.get("enhanced_prompt", "")) > max_chars:
            enhanced_prompt = result.get("enhanced_prompt", "")
            # Truncate at the last complete word within the character limit
            if " " in enhanced_prompt[:max_chars]:
                last_space = enhanced_prompt[:max_chars].rstrip().rfind(" ")
                result["enhanced_prompt"] = enhanced_prompt[:last_space]
            else:
                # If no spaces found, just truncate at max_chars
                result["enhanced_prompt"] = enhanced_prompt[:max_chars]
        
        return result
    except Exception as e:
        raise Exception(f"Failed to parse OpenAI response: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance prompts for image-to-video generation using OpenAI")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--prompt", required=True, help="Original user prompt to enhance")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-chars", type=int, default=400, help="Maximum characters for the enhanced prompt")
    args = parser.parse_args()
    
    try:
        result = analyze_and_improve_prompt(
            args.image, 
            args.prompt, 
            model=args.model, 
            max_chars=args.max_chars
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}") 