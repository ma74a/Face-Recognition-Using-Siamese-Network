from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch

from utils.config import *
from tqdm import tqdm

class SiameseTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, threshold=THRESHOLD):
        self.model = model.to(DEVICE)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.thrsehold = threshold
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.1)

        self.total_train_losses = []
        self.total_val_losses = []

    def train(self, epochs=20):
        print("Starting Training...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for img1, img2, label in progress:
                img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

                self.optimizer.zero_grad()
                output1, output2 = self.model(img1, img2)
                loss = self.criterion(output1, output2, label)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                progress.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(self.train_loader)
            val_loss, val_acc = self.validate()

            self.scheduler.step(val_loss)

            self.total_train_losses.append(avg_loss)
            self.total_val_losses.append(val_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for img1, img2, label in self.val_loader:
                img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

                output1, output2 = self.model(img1, img2)
                loss = self.criterion(output1, output2, label)
                total_loss += loss.item()

                distance = F.pairwise_distance(output1, output2)

                predictions = (distance >= self.thrsehold).long()

                # Compute accuracy
                correct += (predictions == label).sum().item()
                total += label.size(0)

        val_loss = total_loss / len(self.val_loader)
        val_acc = (correct / total) * 100  
        # print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        return val_loss, val_acc

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.model.to(DEVICE)
        print(f"Model loaded from {path}")
