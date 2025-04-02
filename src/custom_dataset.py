import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import os
import random
from PIL import Image

from utlis.config import *

class SiameseFaceDataset(Dataset):
    def __init__(self, data_path, transform=None, pairs_per_class=5):
        self.data_path = data_path
        self.transform = transform or self.default_transform()

        self.classes = os.listdir(data_path)
        self.image_paths = self._load_image_paths() # class_name --> images_paths
        self.pairs = self._generate_pairs(pairs_per_class)

    def _load_image_paths(self):
        """Load images paths, and store the label and the images"""
        image_paths = {}
        for cls_name in os.listdir(self.data_path):
            cls_dir = os.path.join(self.data_path, cls_name)
            if not os.path.exists(cls_dir):
                continue
            images = []
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                images.append(img_path)
            image_paths[cls_name] = images
        
        return image_paths
    
    def _generate_pairs(self, pairs_per_class):
        """To generate pairs, same or different"""
        pairs = []

        # get same class, 0 for same class
        for cls_name, images in self.image_paths.items():
            if len(images) < 2:
                continue
            for _ in range(pairs_per_class):
                img1, img2 = random.sample(images, 2)
                pairs.append((img1, img2, 0))

        # get different class, 1 for different class
        for cls_name, images in self.image_paths.items():
            other_classes = [c for c in self.classes if c != cls_name]
            for _ in range(pairs_per_class):
                other_class = random.choice(other_classes)
                img1 = random.choice(images)
                img2 = random.choice(self.image_paths[other_class])
                pairs.append((img1, img2, 1))

        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        img1, img2, label = self.pairs[index]
        
        img1 = Image.open(img1).convert("RGB")
        img1 = self.transform(img1)

        img2 = Image.open(img2).convert("RGB")
        img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.int32)
    
    @staticmethod
    def default_transform():
        """Create default image transformations."""
        return TRANSFORM["test"]
    

def get_data_loader(data_path, data_type, img_size=128, batch_size=32):
    """Create a DataLoader for the SiameseFaceDataset."""
    if data_type == "train":
        data_set = SiameseFaceDataset(data_path=data_path, transform=TRANSFORM["train"])
        loader = DataLoader(dataset=data_set, batch_size=BATCH_SIZE, shuffle=True)
    else:
        data_set = SiameseFaceDataset(data_path=data_path, transform=TRANSFORM["test"])
        loader = DataLoader(dataset=data_set, batch_size=BATCH_SIZE, shuffle=False)

    return loader