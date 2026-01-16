import torchvision.models as models
import urllib.request
from urllib.error import HTTPError
import zipfile
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from typing import List, Tuple
import random
random.seed(42)
from torchvision.utils import save_image

from gfcs import GFCS
from utils import NormalizedModel


DATASET_PATH = "./imagenet_data"


def download_imagenet():
    """
    download tiny imagenet, which is a subset of imagenet, it is downloaded with class id mapper, this subset has 1000 classes
    same as the 1k imagenet datasets, so it can be aligned with the pytorch vision pretrained models like resnet, densenet, etc.
    credits to yisroel's HW1 that contained the script

    Note that it also normalize the data, so NormaNormalizedModel will not be used on victim model.
    
    return:
        dataset: torch dataset, each index returns a tuple (image tensor of shape [3, 224, 224], label int)
        label_names: list of class names, to be used as a mapper
    """
    # Github URL where the dataset is stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial10/"
    # Files to download
    pretrained_files = [(DATASET_PATH, "TinyImageNet.zip")]
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(DATASET_PATH, exist_ok=True)

    # For each file, check whether it already exists. If not, try downloading it.
    for dir_name, file_name in pretrained_files:
        file_path = os.path.join(dir_name, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", e)
            if file_name.endswith(".zip"):
                print("Unzipping file...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(file_path.rsplit("/",1)[0])

    # Mean and Std from ImageNet
    NORM_MEAN = np.array([0.485, 0.456, 0.406])
    NORM_STD = np.array([0.229, 0.224, 0.225])
    # No resizing and center crop necessary as images are already preprocessed.
    plain_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN,
                            std=NORM_STD)
    ])
    # Load dataset and create data loader
    imagenet_path = os.path.join(DATASET_PATH, "TinyImageNet/")
    assert os.path.isdir(imagenet_path), f"Could not find the ImageNet dataset at expected path \"{imagenet_path}\". "
    dataset = torchvision.datasets.ImageFolder(root=imagenet_path, transform=plain_transforms)

    # Load label names to interpret the label numbers 0 to 999
    with open(os.path.join(imagenet_path, "label_list.json"), "r") as f:
        label_names = json.load(f)



    return dataset, label_names


def sample_from_imagenet(tiny_imagenet_dataset, random_state=42):
    random_idx = random.randint(0, len(tiny_imagenet_dataset) - 1)
    return tiny_imagenet_dataset[random_idx][0].unsqueeze(0), tiny_imagenet_dataset[random_idx][1]


def run_gfcs_attack(
    victim: nn.Module,
    surrogates: List[nn.Module],
    images: List[Tuple[torch.Tensor, int]],
    epsilon: float = 2.0,
    max_queries: int = 10000,
    device: str = 'cuda',
    label_names=None
) -> dict:
    """
    Run GFCS attack on a set of images.
    Modified: no use of NormalizedModel, log true label and predicted label (to make sure tiny imagenet is good), and added label names
    Returns:
        results: Dictionary with attack statistics
    """
    # Wrap models with normalization
    victim_wrapped = victim#NormalizedModel(victim).to(device).eval()
    surrogates_wrapped = [(s).to(device).eval() for s in surrogates]
    
    # Initialize attacker
    attacker = GFCS(
        victim_model=victim_wrapped,
        surrogate_models=surrogates_wrapped,
        epsilon=epsilon,
        max_queries=max_queries,
        targeted=False,
        device=device
    )
    
    if label_names is None:
        raise Exception("please provide label names, it should be downloaded with the tiny imagenet dataset")
    
    results = {
        'success_count': 0,
        'total_count': 0,
        'query_counts': [],
        'gradient_query_counts': [],
        'coimage_query_counts': [],
        'perturbation_norms': []
    }
    
    print(f"\nRunning GFCS attack on {len(images)} images...")
    print(f"Parameters: epsilon={epsilon}, max_queries={max_queries}")
    print("-" * 60)
    
    # with open(os.path.join(DATASET_PATH, 'wnids.txt'), 'r') as f:
    #     tiny_wnids = [line.strip() for line in f.readlines()]
    imagenet_path = os.path.join(DATASET_PATH, "TinyImageNet/")


    for i, (x, true_class) in enumerate(images):
        
        # Get actual prediction from victim
        with torch.no_grad():
            pred_logits = victim_wrapped(x)
            pred_class = pred_logits.argmax(dim=1).item()

        print(f"pred_class={pred_class} ({label_names[pred_class]}) , true_class={true_class} ({label_names[true_class]})")
        # save_image(x, f'{i}_{true_class}.png', normalize=True)


        # Skip if already misclassified
        if pred_class != true_class:
            print(f"Image {i+1}: Already misclassified, skipping")
            continue
        
        start_time = time.time()
        
        # Run attack
        x_adv, stats = attacker.attack(x, true_class)
        
        elapsed = time.time() - start_time
        
        # Compute perturbation norm
        perturbation_norm = torch.norm(x_adv - x).item()
        
        results['total_count'] += 1
        results['query_counts'].append(stats['total_queries'])
        results['gradient_query_counts'].append(stats['gradient_queries'])
        results['coimage_query_counts'].append(stats['coimage_queries'])
        results['perturbation_norms'].append(perturbation_norm)
        
        if stats['success']:
            results['success_count'] += 1
            print(f"Image {i+1}: SUCCESS - Queries: {stats['total_queries']}, "
                  f"Grad: {stats['gradient_queries']}, ODS: {stats['coimage_queries']}, "
                  f"L2: {perturbation_norm:.2f}, Time: {elapsed:.2f}s")
        else:
            print(f"Image {i+1}: FAILED - Queries: {stats['total_queries']}, "
                  f"L2: {perturbation_norm:.2f}, Time: {elapsed:.2f}s")
    
    # Compute aggregate statistics
    print("\n" + "=" * 60)
    print("GFCS ATTACK RESULTS")
    print("=" * 60)
    
    if results['total_count'] > 0:
        success_rate = results['success_count'] / results['total_count'] * 100
        median_queries = np.median(results['query_counts'])
        mean_queries = np.mean(results['query_counts'])
        median_grad = np.median(results['gradient_query_counts'])
        median_ods = np.median(results['coimage_query_counts'])
        mean_l2 = np.mean(results['perturbation_norms'])
        
        print(f"Success Rate: {success_rate:.1f}% ({results['success_count']}/{results['total_count']})")
        print(f"Median Queries: {median_queries:.0f}")
        print(f"Mean Queries: {mean_queries:.1f}")
        print(f"Median Gradient Queries: {median_grad:.0f}")
        print(f"Median ODS Queries: {median_ods:.0f}")
        print(f"Mean L2 Norm: {mean_l2:.2f}")
    else:
        print("No valid images to attack")
    
    return results

def main():
    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    victim = models.resnet50(pretrained=True).eval()
    surrogates = [
        models.vgg19(pretrained=True).eval(),
        models.resnet34(pretrained=True).eval(),
        models.densenet121(pretrained=True).eval(),
        models.mobilenet_v2(pretrained=True).eval(),
    ]


    dataset, label_names = download_imagenet()

    adjusted_images = []
    n_samples = 10
    n_samples = min(n_samples, len(dataset))

    print("sampling from imagenet dataset {n_samples}...")
    for i in range():
        x, true_class = sample_from_imagenet(dataset, random_state=42)
        adjusted_images.append((x.to(device), true_class))


    results = run_gfcs_attack(
        victim=victim,
        surrogates=surrogates,
        images=adjusted_images,
        epsilon=2.0,
        max_queries=10000,
        device="cuda",
        label_names=label_names
    )


if __name__ == "__main__":
    main()
    # dataset, label_names = download_imagenet()

    # print(len(label_names))
    # print(len(dataset))
    # print(len(dataset[0]))
    # print(dataset[0][0].shape)
    # print(label_names[dataset[0][1]])

    # print(dataset)