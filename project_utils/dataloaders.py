import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import albumentations as A


transform_3c_imagenet_hvflip = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
transform_3c_imagenet_hvnoflip = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
transform_3c_simple_hvflip = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
    ])
transform_3c_simple_hvnoflip = transforms.Compose([
    transforms.ToTensor()
    ])
transform_3c_slot_attention_hvflip = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
    ])
transform_3c_slot_attention_hvnoflip = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
    ])
transform_1c_simple_hvflip = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ])
transform_1c_simple_hvnoflip = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ])
transform_rand_noise_1c_hvnoflip = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.9710], [0.1677])
    ])
flip_transform = A.Compose(
    transforms=[
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5)],
    additional_targets={'image0': 'image'}
    )
transform_1c_slot_attention_hvflip = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
    ])
transform_1c_slot_attention_hvnoflip = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
    ])
transform_CLIPViT_hvflip = transforms.Compose([
	transforms.Resize(size=224, max_size=None, antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
transform_CLIPViT_hvnoflip = transforms.Compose([
	transforms.Resize(size=224, max_size=None, antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])


class SD_dataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        split, 
        normalization='samplewise1', 
        squeeze=True, 
        invert=False,
        data_format=None
        ):
        """Same-different SVRT dataset (SVRT task 1 and variants).
        Args:
            root_dir: directory of svrt1 dataset.
            split: 'test', 'train', or 'val'.
            normalization: 'imagenet', 'samplewise3', 'samplewise1', 'slotattention' or None (for OCRA model).
            squeeze: whether to eliminate channel dimension.
            invert: whether to invert image (for OCRA model).
            data_format: None (standard) or 'OCRA'.
        """
        self.dir = f'{root_dir}/{split}'
        self.annotations = pd.read_csv(f'{root_dir}/{split}_annotations.csv')
        self.normalization = normalization
        if self.normalization == 'imagenet':
            if split == 'train':
                self.transform = transform_3c_imagenet_hvflip
            else:
                self.transform = transform_3c_imagenet_hvnoflip
        elif self.normalization == 'samplewise3':
            if split == 'train':
                self.transform = transform_3c_simple_hvflip
            else:
                self.transform = transform_3c_simple_hvnoflip
        elif self.normalization == 'samplewise1':
            if split == 'train':
                self.transform = transform_1c_simple_hvflip
            else:
                self.transform = transform_1c_simple_hvnoflip
        elif self.normalization == 'slotattention':
            if split == 'train':
                self.transform = transform_3c_slot_attention_hvflip
            else:
                self.transform = transform_3c_slot_attention_hvnoflip
        elif self.normalization == 'OCRAbs':
            if split == 'train':
                self.transform = transform_1c_slot_attention_hvflip
            else:
                self.transform = transform_1c_slot_attention_hvnoflip
        elif self.normalization == 'CLIPViT':
            if split == 'train':
                self.transform = transform_CLIPViT_hvflip
            else:
                self.transform = transform_CLIPViT_hvnoflip
        elif self.normalization is None:
            if split == 'train':
                self.transform = transform_1c_simple_hvflip
            else:
                self.transform = transform_1c_simple_hvnoflip
        else:
            raise ValueError('unrecognized normalization!')
        self.squeeze = squeeze
        self.invert = invert
        self.data_format = data_format

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the right image names according to the label
        img_name = f'{self.dir}/{idx}.png'
        img = self.transform(Image.open(img_name))
        # Build data tensors
        if self.normalization == 'samplewise1' or self.normalization == 'samplewise3':
            # Samplewise normalization
            mean_img = torch.mean(img, dim=(1,2)).tolist()
            std_img = torch.std(img, dim=(1,2), unbiased=False).add(0.000001).tolist()
            transform_normalize = transforms.Normalize(
                mean=mean_img,
                std=std_img
                )    
            img = transform_normalize(img)
        
        if self.squeeze:
            img = torch.squeeze(img)
            
        if self.invert:
            img = TF.invert(img)
        
        # Get target and optional loss weights
        if self.data_format == 'OCRA':
            y_index = self.annotations.iloc[idx, 1]
            y = np.array([1, 1, 0, 0])
            y[2+y_index] = 1
            y = torch.Tensor(y)
            loss_w = np.array([1, 1, 2, 2])
            loss_w = torch.Tensor(loss_w)
            data = img, y, loss_w
        else:
            y = self.annotations.iloc[idx, 1]
            data = img, y

        return data

class MTS_dataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        split, 
        normalization='samplewise1', 
        squeeze=True, 
        invert=False,
        data_format=None
        ):
        """Same-different SVRT dataset (SVRT task 1 and variants).
        Args:
            root_dir: directory of svrt1 dataset.
            split: 'test', 'train', or 'val'.
            normalization: 'imagenet', 'samplewise3', 'samplewise1', 'slotattention' or None (for OCRA model).
            squeeze: whether to eliminate channel dimension.
            invert: whether to invert image (for OCRA model).
            data_format: None (standard) or 'OCRA'.
        """
        self.dir = f'{root_dir}/{split}'
        self.annotations = pd.read_csv(f'{root_dir}/{split}_annotations.csv')
        self.normalization = normalization
        if self.normalization == 'imagenet':
            self.transform = transform_3c_imagenet_hvnoflip
        elif self.normalization == 'samplewise3':
            self.transform = transform_3c_simple_hvnoflip
        elif self.normalization == 'samplewise1':
            self.transform = transform_1c_simple_hvnoflip
        elif self.normalization == 'slotattention':
            self.transform = transform_3c_slot_attention_hvnoflip
        elif self.normalization == 'OCRAbs':
            self.transform = transform_1c_slot_attention_hvnoflip
        elif self.normalization == 'CLIPViT':
            self.transform = transform_CLIPViT_hvnoflip
        elif self.normalization is None:
            self.transform = transform_1c_simple_hvnoflip
        else:
            raise ValueError('unrecognized normalization!')
        self.squeeze = squeeze
        self.invert = invert
        self.data_format = data_format

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the right image names according to the label
        img_name = f'{self.dir}/{idx}.png'
        img = self.transform(Image.open(img_name))
        # Build data tensors
        if self.normalization == 'samplewise1' or self.normalization == 'samplewise3':
            # Samplewise normalization
            mean_img = torch.mean(img, dim=(1,2)).tolist()
            std_img = torch.std(img, dim=(1,2), unbiased=False).add(0.000001).tolist()
            transform_normalize = transforms.Normalize(
                mean=mean_img,
                std=std_img
                )    
            img = transform_normalize(img)
        
        if self.squeeze:
            img = torch.squeeze(img)
            
        if self.invert:
            img = TF.invert(img)
        
        # Get target and optional loss weights
        if self.data_format == 'OCRA':
            y_index = self.annotations.iloc[idx, 1]
            y = np.array([1, 1, 1, 1, 1, 0, 0])
            y[5+y_index] = 1
            y = torch.Tensor(y)
            loss_w = np.array([1, 1, 1, 1, 1, 2, 2])
            loss_w = torch.Tensor(loss_w)
            data = img, y, loss_w
        else:
            y = self.annotations.iloc[idx, 1]
            data = img, y

        return data

class RMTS_dataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        split, 
        normalization='samplewise1', 
        squeeze=True, 
        invert=False,
        data_format=None
        ):
        """RMTS datasets.
        Args:
            root_dir: directory of dataset.
            split: 'test', 'train', or 'val'.
            normalization: 'imagenet', 'samplewise3', 'samplewise1', 'slotattention' or None.
            squeeze: whether to eliminate channel dimension.
            invert: whether to invert image (for OCRA model).
            data_format: None (standard) or 'OCRA'.
        """
        self.dir = f'{root_dir}/{split}'
        self.annotations = pd.read_csv(f'{root_dir}/{split}_annotations.csv')
        self.normalization = normalization
        if self.normalization == 'imagenet':
            self.transform = transform_3c_imagenet_hvnoflip
        elif self.normalization == 'samplewise3':
            self.transform = transform_3c_simple_hvnoflip
        elif self.normalization == 'samplewise1':
            self.transform = transform_1c_simple_hvnoflip
        elif self.normalization == 'slotattention':
            self.transform = transform_3c_slot_attention_hvnoflip
        elif self.normalization == 'OCRAbs':
            self.transform = transform_1c_slot_attention_hvnoflip
        elif self.normalization == 'CLIPViT':
            self.transform = transform_CLIPViT_hvnoflip
        elif self.normalization is None:
            self.transform = transform_1c_simple_hvnoflip
        else:
            raise ValueError('unrecognized normalization!')
        self.squeeze = squeeze
        self.invert = invert
        self.data_format = data_format
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the right image names according to the label
        img_name = f'{self.dir}/{idx}.png'
        img = self.transform(Image.open(img_name))
        # Build data tensors
        if self.normalization == 'samplewise1' or self.normalization == 'samplewise3':
            # Samplewise normalization
            mean_img = torch.mean(img, dim=(1,2)).tolist()
            std_img = torch.std(img, dim=(1,2), unbiased=False).add(0.000001).tolist()
            transform_normalize = transforms.Normalize(
                mean=mean_img,
                std=std_img
                )    
            img = transform_normalize(img)
        
        if self.squeeze:
            img = torch.squeeze(img)
        if self.invert:
            img = TF.invert(img)
        
        # Get target and optional loss weights
        if self.data_format == 'OCRA':
            y_index = self.annotations.iloc[idx, 2]
            y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
            y[8+y_index] = 1
            y = torch.Tensor(y)
            loss_w = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2])
            loss_w = torch.Tensor(loss_w)
            data = img, y, loss_w
            
        else:
            y = self.annotations.iloc[idx, 2] # ID, base (s/d), label (l/r)
            data = img, y

        return data

class SOSD_dataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        split, 
        normalization='samplewise1', 
        invert=False,
        squeeze=False, 
        data_format=None
        ):
        """Second order same-different SVRT dataset.
        Args:
            root_dir: directory of dataset.
            split: 'test', 'train', or 'val'.
            normalization: 'imagenet', 'samplewise3', 'samplewise1', 'slotattention' or None (for OCRA model).
            squeeze: whether to eliminate channel dimension.
            invert: whether to invert image (for OCRA model).
            data_format: None (standard) or 'OCRA'.
        """
        self.dir = f'{root_dir}/{split}'
        self.annotations = pd.read_csv(f'{root_dir}/{split}_annotations.csv')
        self.normalization = normalization
        if self.normalization == 'imagenet':
            if split == 'train':
                self.transform = transform_3c_imagenet_hvflip
            else:
                self.transform = transform_3c_imagenet_hvnoflip
        elif self.normalization == 'samplewise3':
            if split == 'train':
                self.transform = transform_3c_simple_hvflip
            else:
                self.transform = transform_3c_simple_hvnoflip
        elif self.normalization == 'samplewise1':
            if split == 'train':
                self.transform = transform_1c_simple_hvflip
            else:
                self.transform = transform_1c_simple_hvnoflip
        elif self.normalization == 'slotattention':
            if split == 'train':
                self.transform = transform_3c_slot_attention_hvflip
            else:
                self.transform = transform_3c_slot_attention_hvnoflip
        elif self.normalization == 'OCRAbs':
            if split == 'train':
                self.transform = transform_1c_slot_attention_hvflip
            else:
                self.transform = transform_1c_slot_attention_hvnoflip
        elif self.normalization == 'CLIPViT':
            if split == 'train':
                self.transform = transform_CLIPViT_hvflip
            else:
                self.transform = transform_CLIPViT_hvnoflip
        elif self.normalization is None:
            if split == 'train':
                self.transform = transform_1c_simple_hvflip
            else:
                self.transform = transform_1c_simple_hvnoflip
        else:
            raise ValueError('unrecognized normalization!')
        self.squeeze = squeeze
        self.invert = invert
        self.data_format = data_format

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the right image names according to the label
        img_name = f'{self.dir}/{idx}.png'
        img = self.transform(Image.open(img_name))
        # Build data tensors
        if self.normalization == 'samplewise1' or self.normalization == 'samplewise3':
            # Samplewise normalization
            mean_img = torch.mean(img, dim=(1,2)).tolist()
            std_img = torch.std(img, dim=(1,2), unbiased=False).add(0.000001).tolist()
            transform_normalize = transforms.Normalize(
                mean=mean_img,
                std=std_img
                )    
            img = transform_normalize(img)
        
        if self.squeeze:
            img = torch.squeeze(img)
        
        if self.invert:
            img = TF.invert(img)
        
        # Get target and optional loss weights
        if self.data_format == 'OCRA':
            y_index = self.annotations.iloc[idx, 3] # ID, top label, bottom label, label
            y = np.array([1, 1, 1, 1, 1, 1, 0, 0])
            y[6+y_index] = 1
            y = torch.Tensor(y)
            loss_w = np.array([1, 1, 1, 1, 1, 1, 2, 2])
            loss_w = torch.Tensor(loss_w)
            data = img, y, loss_w
        else:
            y = self.annotations.iloc[idx, 3]
            data = img, y

        return data

    
# Method for creating directory if it doesn't exist yet
def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
