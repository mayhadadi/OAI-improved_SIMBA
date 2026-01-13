"""
Setup ImageNet Validation Dataset for Google Colab
===================================================
This script downloads and sets up the ImageNet validation dataset for use with GFCS experiments.

Usage:
    In Google Colab:
    1. Upload your kaggle.json credentials
    2. Run this script: !python setup_imagenet_colab.py
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and print status."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        return False
    print(f"✓ {description} completed successfully")
    return True

def setup_kaggle_credentials():
    """Setup Kaggle API credentials."""
    print("\n" + "="*80)
    print("SETTING UP KAGGLE CREDENTIALS")
    print("="*80)
    
    # Check if kaggle.json exists in current directory
    if not os.path.exists('kaggle.json'):
        print("\n⚠️  kaggle.json not found in current directory!")
        print("\nTo download ImageNet from Kaggle, you need to:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section and click 'Create New API Token'")
        print("3. This will download kaggle.json")
        print("4. Upload kaggle.json to Colab using the file upload button")
        print("5. Run this script again")
        return False
    
    # Create .kaggle directory
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    
    # Copy kaggle.json to ~/.kaggle/
    import shutil
    shutil.copy('kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
    
    # Set permissions
    os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
    
    print("✓ Kaggle credentials configured")
    return True

def install_kaggle():
    """Install Kaggle CLI."""
    if not run_command("pip install -q kaggle", "Installing Kaggle CLI"):
        return False
    return True

def download_imagenet():
    """Download ImageNet validation dataset from Kaggle."""
    print("\n" + "="*80)
    print("DOWNLOADING IMAGENET VALIDATION DATASET")
    print("="*80)
    print("This will download ~6.3 GB of data. This may take several minutes...")
    
    # Create data directory
    os.makedirs('./data', exist_ok=True)
    
    # Download using Kaggle API
    # Note: Using ImageNet Object Localization Challenge which includes validation set
    cmd = "kaggle competitions download -c imagenet-object-localization-challenge -p ./data/"
    if not run_command(cmd, "Downloading ImageNet from Kaggle"):
        return False
    
    return True

def extract_imagenet():
    """Extract ImageNet validation dataset."""
    print("\n" + "="*80)
    print("EXTRACTING IMAGENET VALIDATION DATASET")
    print("="*80)
    
    # Check if zip file exists
    zip_path = './data/imagenet-object-localization-challenge.zip'
    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found!")
        return False
    
    # Extract validation set
    cmd = f"unzip -q {zip_path} ILSVRC/Data/CLS-LOC/val/* -d ./data/"
    if not run_command(cmd, "Extracting validation images"):
        # Try alternative extraction
        cmd = f"unzip -q {zip_path} -d ./data/"
        if not run_command(cmd, "Extracting all ImageNet data"):
            return False
    
    return True

def organize_val_images():
    """Organize validation images into class folders."""
    print("\n" + "="*80)
    print("ORGANIZING VALIDATION IMAGES INTO CLASS FOLDERS")
    print("="*80)
    
    # The validation set needs to be organized into subdirectories
    # PyTorch's ImageFolder expects: data/imagenet/val/class_name/image.jpg
    
    val_dir = './data/ILSVRC/Data/CLS-LOC/val'
    if not os.path.exists(val_dir):
        print(f"ERROR: {val_dir} not found!")
        print("Looking for alternative paths...")
        # Try to find val directory
        for root, dirs, files in os.walk('./data'):
            if 'val' in dirs:
                potential_val = os.path.join(root, 'val')
                if any(f.endswith(('.JPEG', '.jpg', '.png')) for f in os.listdir(potential_val)):
                    val_dir = potential_val
                    print(f"Found validation directory at: {val_dir}")
                    break
    
    # Download and use validation labels
    print("Downloading validation ground truth labels...")
    cmd = "wget -q https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt -O ./data/val_labels.txt"
    run_command(cmd, "Downloading validation labels")
    
    # Organize images into class folders
    print("Organizing images into class folders...")
    
    # Create symlink for easier access
    target_dir = './data/imagenet/val'
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    
    # If labels file exists, organize by class
    if os.path.exists('./data/val_labels.txt'):
        with open('./data/val_labels.txt', 'r') as f:
            labels = [line.strip() for line in f]
        
        # Get all validation images
        val_images = sorted([f for f in os.listdir(val_dir) if f.endswith('.JPEG')])
        
        if len(labels) == len(val_images):
            print(f"Creating class directories for {len(set(labels))} classes...")
            for img, label in zip(val_images, labels):
                class_dir = os.path.join(target_dir, label)
                os.makedirs(class_dir, exist_ok=True)
                
                src = os.path.join(val_dir, img)
                dst = os.path.join(class_dir, img)
                
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        else:
            print(f"WARNING: Label count ({len(labels)}) != image count ({len(val_images)})")
            print("Creating single directory structure...")
            os.symlink(val_dir, target_dir, target_is_directory=True)
    else:
        # Just create a symlink to the val directory
        print("Creating symlink to validation directory...")
        if not os.path.exists(target_dir):
            os.symlink(val_dir, target_dir, target_is_directory=True)
    
    print(f"✓ Validation images organized at: {target_dir}")
    return True

def verify_setup():
    """Verify the dataset is properly set up."""
    print("\n" + "="*80)
    print("VERIFYING DATASET SETUP")
    print("="*80)
    
    expected_path = './data/imagenet/val'
    if not os.path.exists(expected_path):
        print(f"❌ ERROR: {expected_path} does not exist!")
        return False
    
    # Count images
    import glob
    images = glob.glob(os.path.join(expected_path, '**/*.JPEG'), recursive=True)
    image_count = len(images)
    
    print(f"✓ Dataset path exists: {expected_path}")
    print(f"✓ Found {image_count} images")
    
    if image_count < 2000:
        print(f"⚠️  WARNING: Only {image_count} images found, need at least 2000")
        return False
    
    print("\n✅ Dataset setup successful!")
    print(f"You can now run: python run_experiment_from_config.py exp_001 --config_dir ./configs")
    return True

def main():
    """Main setup function."""
    print("\n" + "="*80)
    print("IMAGENET VALIDATION DATASET SETUP FOR GOOGLE COLAB")
    print("="*80)
    
    # Step 1: Install Kaggle CLI
    if not install_kaggle():
        print("\n❌ Failed to install Kaggle CLI")
        return
    
    # Step 2: Setup Kaggle credentials
    if not setup_kaggle_credentials():
        print("\n❌ Failed to setup Kaggle credentials")
        return
    
    # Step 3: Download ImageNet
    if not download_imagenet():
        print("\n❌ Failed to download ImageNet")
        return
    
    # Step 4: Extract ImageNet
    if not extract_imagenet():
        print("\n❌ Failed to extract ImageNet")
        return
    
    # Step 5: Organize validation images
    if not organize_val_images():
        print("\n❌ Failed to organize validation images")
        return
    
    # Step 6: Verify setup
    verify_setup()

if __name__ == "__main__":
    main()
