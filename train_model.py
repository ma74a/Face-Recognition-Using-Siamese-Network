import torch.optim as optim
import torch

from src.model import SiameseNetwork, ContrastiveLoss
from src.custom_dataset import get_data_loaders
from src.trainning import SiameseTrainer
from utils.visualize import show_images
from utils.config import *




def main():
    train_loader = get_data_loaders(TRAIN_DIR, "train")
    val_loader = get_data_loaders(VAL_DIR, "val")


    model = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    trainer = SiameseTrainer(model=model, train_loader=train_loader, val_loader=val_loader, 
                            optimizer=optimizer, criterion=criterion)

    trainer.train()
    trainer.save_model(path=MODELS+"model_v1.pth")

if __name__ == "__main__":
    main()