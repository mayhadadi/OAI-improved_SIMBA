"""
CIFAR-10 Pretrained Models Loader
==================================
Loads pretrained models for CIFAR-10 from torch.hub (chenyaofo/pytorch-cifar-models)
"""

import torch
import torch.nn as nn


def load_cifar10_model(model_name: str, device: str = 'cuda') -> nn.Module:
    """Load a CIFAR-10 pretrained model from torch.hub."""
    model_map = {
        'resnet20': 'cifar10_resnet20',
        'resnet32': 'cifar10_resnet32',
        'resnet44': 'cifar10_resnet44', 
        'resnet56': 'cifar10_resnet56',
        'resnet110': 'cifar10_resnet110',
        'vgg11': 'cifar10_vgg11_bn',
        'vgg13': 'cifar10_vgg13_bn',
        'vgg16': 'cifar10_vgg16_bn',
        'vgg19': 'cifar10_vgg19_bn',
        'mobilenet_v2': 'cifar10_mobilenetv2_x1_0',
    }
    
    hub_model_name = model_map.get(model_name.lower())
    if hub_model_name is None:
        raise ValueError(f"Model '{model_name}' not available for CIFAR-10. Available: {list(model_map.keys())}")
    
    print(f"  Loading {hub_model_name} from torch.hub...")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", hub_model_name, pretrained=True, verbose=False)
    print(f"  ✓ Loaded {hub_model_name}")
    
    return model.to(device).eval()
