"""
Final ImageNet Setup - Working Solution
========================================
Downloads ImageNet validation set using Academic Torrents (most reliable)
"""

import os
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required packages."""
    print("="*80)
    print("INSTALLING DEPENDENCIES")
    print("="*80)
    subprocess.run("pip install -q academictorrents pillow", shell=True)
    print("✓ Dependencies installed\n")

def download_via_academic_torrents():
    """Download ImageNet validation via Academic Torrents."""
    print("="*80)
    print("DOWNLOADING IMAGENET VALIDATION VIA ACADEMIC TORRENTS")
    print("="*80)
    print("This will download ~6.3 GB")
    print("This may take 15-30 minutes...\n")
    
    try:
        import academictorrents as at
        
        # Download ImageNet validation set
        # Hash for ILSVRC2012 validation set
        download_path = at.get(
            '5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5',
            datastore='./data/imagenet_raw'
        )
        
        print(f"\n✓ Downloaded to: {download_path}")
        return download_path
        
    except Exception as e:
        print(f"✗ Academic Torrents failed: {e}")
        return None

def download_via_wget():
    """Try direct download via wget."""
    print("\n" + "="*80)
    print("ALTERNATIVE: DIRECT DOWNLOAD")
    print("="*80)
    
    os.makedirs('./data/imagenet_raw', exist_ok=True)
    
    # Try direct download from mirror
    urls = [
        # Add working mirror URLs here if available
        # "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar"
    ]
    
    for url in urls:
        print(f"Trying: {url}")
        result = subprocess.run(
            f"wget -c {url} -P ./data/imagenet_raw/",
            shell=True,
            capture_output=True
        )
        if result.returncode == 0:
            return './data/imagenet_raw/'
    
    return None

def extract_and_organize():
    """Extract and organize ImageNet into class folders."""
    print("\n" + "="*80)
    print("ORGANIZING IMAGENET VALIDATION SET")
    print("="*80)
    
    # Create output directory
    val_dir = Path('./data/imagenet/val')
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for tar file
    tar_files = list(Path('./data/imagenet_raw').glob('*.tar'))
    
    if not tar_files:
        print("✗ No tar file found")
        return False
    
    tar_file = tar_files[0]
    print(f"Extracting: {tar_file}")
    
    # Extract
    subprocess.run(f"tar -xf {tar_file} -C {val_dir}", shell=True)
    
    # Download and run organization script
    print("Organizing images into class folders...")
    subprocess.run(
        "wget -q https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh -O valprep.sh",
        shell=True
    )
    subprocess.run("chmod +x valprep.sh", shell=True)
    subprocess.run(f"./valprep.sh {val_dir}", shell=True)
    
    print(f"✓ Organized images in: {val_dir}")
    return True

def verify_dataset():
    """Verify dataset is ready."""
    print("\n" + "="*80)
    print("VERIFYING DATASET")
    print("="*80)
    
    val_dir = Path('./data/imagenet/val')
    
    if not val_dir.exists():
        print(f"❌ {val_dir} does not exist")
        return False
    
    # Count images
    import glob
    images = glob.glob(str(val_dir / '**/*.JPEG'), recursive=True)
    num_images = len(images)
    
    print(f"Found {num_images} images")
    
    if num_images >= 2000:
        print("✅ Dataset ready for experiments!")
        return True
    else:
        print(f"⚠️ Only {num_images} images found")
        return False

def print_manual_instructions():
    """Print manual download instructions."""
    print("\n" + "="*80)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print("""
If automatic download fails, download ImageNet manually:

1. Academic Torrents (Recommended):
   - Go to: https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5
   - Download: ILSVRC2012_img_val.tar (6.3 GB)
   - Upload to Google Drive
   
2. In Colab, run:
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Extract
   !mkdir -p ./data/imagenet/val
   !tar -xf /content/drive/MyDrive/ILSVRC2012_img_val.tar -C ./data/imagenet/val/
   
   # Organize
   !wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
   !chmod +x valprep.sh
   !./valprep.sh ./data/imagenet/val/

3. Run your experiment:
   !python run_experiment_from_config.py exp_001 --config_dir ./configs --device cuda
""")

def main():
    print("="*80)
    print("IMAGENET VALIDATION SETUP - FINAL SOLUTION")
    print("="*80)
    
    # Install dependencies
    install_dependencies()
    
    # Try Academic Torrents
    print("\nAttempting automatic download via Academic Torrents...")
    download_path = download_via_academic_torrents()
    
    if download_path:
        # Extract and organize
        if extract_and_organize():
            # Verify
            if verify_dataset():
                print("\n" + "="*80)
                print("SUCCESS! Ready to run experiments")
                print("="*80)
                return
    
    # If automatic failed, print manual instructions
    print("\n❌ Automatic download failed")
    print_manual_instructions()

if __name__ == "__main__":
    main()
