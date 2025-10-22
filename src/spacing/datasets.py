"""
Dataset classes for Spacing Predictor
"""

import cv2
import torch
from torch.utils.data import Dataset
from albumentations import (
    Compose, PadIfNeeded, CenterCrop,
    Transpose, HorizontalFlip, VerticalFlip,
    ShiftScaleRotate, Normalize
)
from albumentations.pytorch import ToTensorV2

from .config import CFG


class TrainDataset(Dataset):
    def __init__(self, df, transform=None, crop_ratio=None):
        self.df = df
        self.file_paths = df['file_path'].values
        self.labels = df[CFG.target_cols].values
        self.transform = transform
        self.crop_ratio = crop_ratio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = torch.tensor(self.labels[idx]).float()
        return image, label


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image


def get_transforms(*, data):
    if data == 'train':
        return Compose([
            PadIfNeeded(min_height=CFG.size, min_width=CFG.size, border_mode=0, value=0),
            CenterCrop(height=CFG.size, width=CFG.size),
            # Light augmentations
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            PadIfNeeded(min_height=CFG.size, min_width=CFG.size, border_mode=0, value=0),
            CenterCrop(height=CFG.size, width=CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])