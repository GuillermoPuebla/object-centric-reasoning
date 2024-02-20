import torch
import torch.nn as nn
from torchvision import models


def get_ResNet50(weights='DEFAULT'):
    model_ft = models.resnet50(weights=weights)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)
    return model_ft

def get_ResNet152(weights='DEFAULT'):
    model_ft = models.resnet152(weights=weights)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)
    return model_ft
