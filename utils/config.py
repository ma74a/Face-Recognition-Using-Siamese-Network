import torch
from torchvision import transforms
import os


# Paths
TRAIN_DIR = "TRAIN_PATH"
VAL_DIR = "VAL_PATH"
MODELS = "models"

os.makedirs(MODELS, exist_ok=True)

# Feacture Vector
EMBEDDING_DIM = 256

# Hyperparameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 30
MARGIN = 1.0
THRESHOLD = 0.5

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image size
IMG_SIZE = 224

TRANSFORM = {
    "train": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), shear=10)
]),
    "test" : transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}