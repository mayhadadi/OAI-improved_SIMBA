"""
Tiny ImageNet Dataset Loader (Minimal Version)
===============================================
Simple loader for Tiny ImageNet that auto-downloads and integrates
seamlessly with existing ImageNet experiment configs.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import urllib.request
from urllib.error import HTTPError
import zipfile
import os
import json
from typing import Tuple, List


def download_tiny_imagenet(dataset_path: str = "./data/tiny_imagenet"):
    """
    Download Tiny ImageNet dataset (1000 classes, subset of ImageNet).
    
    Args:
        dataset_path: Where to save the dataset
        
    Returns:
        Path to the extracted dataset
    """
    # GitHub URL for the dataset
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial10/"
    file_name = "TinyImageNet.zip"
    
    # Create directory
    os.makedirs(dataset_path, exist_ok=True)
    
    # Download if needed
    file_path = os.path.join(dataset_path, file_name)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading Tiny ImageNet from {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
            print("Download complete!")
        except HTTPError as e:
            raise Exception(
                f"Failed to download Tiny ImageNet. Error: {e}\n"
                "Please download manually from the repository."
            )
        
        # Extract
        if file_name.endswith(".zip"):
            print("Extracting Tiny ImageNet...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            print("Extraction complete!")
    else:
        print(f"Tiny ImageNet already downloaded at {dataset_path}")
    
    return os.path.join(dataset_path, "TinyImageNet/")


def load_tiny_imagenet_dataset(
    dataset_path: str = "./data/tiny_imagenet",
    download: bool = True
) -> Tuple[torchvision.datasets.ImageFolder, List[str]]:
    """
    Load Tiny ImageNet dataset.
    
    Args:
        dataset_path: Path to dataset directory
        download: Whether to download if not present
    
    Returns:
        dataset: ImageFolder dataset (already normalized)
        label_names: List of class names
    """
    if download:
        imagenet_path = download_tiny_imagenet(dataset_path)
    else:
        imagenet_path = os.path.join(dataset_path, "TinyImageNet/")
    
    # Check dataset exists
    if not os.path.isdir(imagenet_path):
        raise FileNotFoundError(
            f"Tiny ImageNet not found at {imagenet_path}. Set download=True to download."
        )
    
    # ImageNet normalization (applied in the transform)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    
    # Transform: Images are already 224x224, just need ToTensor + Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    # Load dataset
    dataset = torchvision.datasets.ImageFolder(
        root=imagenet_path,
        transform=transform
    )
    
    # Load class names
    label_list_path = os.path.join(imagenet_path, "label_list.json")
    with open(label_list_path, "r") as f:
        label_names = json.load(f)
    
    print(f"✓ Loaded Tiny ImageNet: {len(dataset)} images, {len(label_names)} classes")
    
    return dataset, label_names


if __name__ == "__main__":
    # Test the loader
    print("Testing Tiny ImageNet loader...")
    dataset, labels = load_tiny_imagenet_dataset(download=True)
    print(f"Dataset size: {len(dataset)}")
    print(f"First image shape: {dataset[0][0].shape}")
    print(f"First 5 classes: {labels[:5]}")