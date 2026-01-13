"""
Download ImageNet from Hugging Face for GFCS Experiments
=========================================================
This script downloads ImageNet-1k validation set from Hugging Face
and prepares it for use with the experiment configs.

Requirements:
    1. Hugging Face account
    2. Accept terms at: https://huggingface.co/datasets/imagenet-1k
    3. Get token from: https://huggingface.co/settings/tokens

Usage:
    python setup_imagenet_huggingface.py --token YOUR_HF_TOKEN
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def install_dependencies():
    """Install required packages."""
    print("="*80)
    print("INSTALLING DEPENDENCIES")
    print("="*80)
    import subprocess
    subprocess.run("pip install -q datasets huggingface_hub pillow tqdm", shell=True)
    print("✓ Dependencies installed\n")

def download_imagenet(hf_token):
    """Download ImageNet from Hugging Face."""
    print("="*80)
    print("DOWNLOADING IMAGENET FROM HUGGING FACE")
    print("="*80)
    
    from huggingface_hub import login
    from datasets import load_dataset
    
    # Login
    print("Logging in to Hugging Face...")
    login(token=hf_token)
    print("✓ Logged in\n")
    
    # Download validation set
    print("Downloading ImageNet-1k validation set...")
    print("This will download ~6.7 GB of data (50,000 images)")
    print("This may take 10-20 minutes...\n")
    
    dataset = load_dataset(
        "imagenet-1k", 
        split="validation",
        cache_dir="./data/imagenet_cache",
        trust_remote_code=True
    )
    
    print(f"\n✓ Downloaded {len(dataset)} validation images!")
    return dataset

def prepare_dataset_structure(dataset):
    """
    Save dataset in ImageFolder structure for PyTorch.
    Structure: ./data/imagenet/val/{class_id}/image.jpg
    """
    print("\n" + "="*80)
    print("ORGANIZING DATASET INTO IMAGEFOLDER STRUCTURE")
    print("="*80)
    
    output_dir = Path("./data/imagenet/val")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving images to: {output_dir}")
    print("This may take 10-15 minutes...\n")
    
    # Save each image in its class folder
    for idx, sample in enumerate(tqdm(dataset, desc="Processing images")):
        img = sample['image']
        label = sample['label']
        
        # Create class directory
        class_dir = output_dir / str(label)
        class_dir.mkdir(exist_ok=True)
        
        # Save image
        img_path = class_dir / f"img_{idx:05d}.JPEG"
        
        # Convert to RGB if needed (some images are grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img.save(img_path, 'JPEG', quality=95)
    
    print(f"\n✓ Saved all images to {output_dir}")
    return output_dir

def verify_dataset():
    """Verify the dataset is properly set up."""
    print("\n" + "="*80)
    print("VERIFYING DATASET")
    print("="*80)
    
    import glob
    
    val_dir = Path("./data/imagenet/val")
    
    if not val_dir.exists():
        print(f"❌ ERROR: {val_dir} does not exist!")
        return False
    
    # Count classes
    class_dirs = [d for d in val_dir.iterdir() if d.is_dir()]
    print(f"✓ Found {len(class_dirs)} classes")
    
    # Count images
    images = list(val_dir.glob("**/*.JPEG"))
    print(f"✓ Found {len(images)} images")
    
    if len(images) >= 2000:
        print(f"✅ Sufficient images for experiment!")
        print(f"\n{'='*80}")
        print("SETUP COMPLETE - Ready to run experiments!")
        print('='*80)
        print("\nNext step:")
        print("!python run_experiment_from_config.py exp_001 --config_dir ./configs --device cuda")
        return True
    else:
        print(f"⚠️ Only {len(images)} images found (expected ~50,000)")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download ImageNet from Hugging Face')
    parser.add_argument('--token', type=str, help='Hugging Face API token')
    parser.add_argument('--skip-save', action='store_true', 
                       help='Skip saving to disk (keeps in HF cache only)')
    args = parser.parse_args()
    
    print("="*80)
    print("IMAGENET SETUP VIA HUGGING FACE")
    print("="*80)
    print()
    
    # Get token
    hf_token = args.token
    if not hf_token:
        print("ERROR: Hugging Face token required!")
        print("\nTo get your token:")
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("2. Click 'New token' and create a read token")
        print("3. Copy the token")
        print("\nThen run:")
        print("python setup_imagenet_huggingface.py --token YOUR_TOKEN_HERE")
        sys.exit(1)
    
    # Install dependencies
    install_dependencies()
    
    # Download dataset
    dataset = download_imagenet(hf_token)
    
    if not args.skip_save:
        # Prepare ImageFolder structure
        prepare_dataset_structure(dataset)
        
        # Verify
        verify_dataset()
    else:
        print("\n✓ Dataset downloaded to cache")
        print("Note: You'll need to modify run_experiment_from_config.py")
        print("      to load directly from HF cache instead of ImageFolder")

if __name__ == "__main__":
    main()
