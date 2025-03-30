import sys
import importlib
import subprocess
from pathlib import Path

def check_package(package_name):
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "Unknown")
        print(f"✓ {package_name} (version: {version})")
        return True
    except ImportError:
        print(f"✗ {package_name} is not installed")
        return False

def main():
    print(f"Python version: {sys.version.split()[0]}")
    
    # Essential dependencies
    required_packages = [
        "torch",
        "diffusers",
        "safetensors",
        "transformers",
        "huggingface_hub",
        "PIL",
        "imageio"
    ]
    
    # Check each package
    print("\nChecking required packages:")
    missing_packages = []
    for package in required_packages:
        if not check_package(package):
            missing_packages.append(package)
    
    # Check CUDA availability
    if check_package("torch"):
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Installation instructions for missing packages
    if missing_packages:
        print("\nMissing packages. Install them with:")
        print("pip install " + " ".join(missing_packages))
        print("\nFor PyTorch with CUDA support, use:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("\nAll required packages are installed!")
        print("\nYou're ready to run the example scripts:")
        print("- 00_text_to_video.py")
        print("- 01_text_to_video_diffusers.py")
        print("- 02_image_to_video.py")
        print("- 03_image_to_video_diffusers.py")

if __name__ == "__main__":
    main() 