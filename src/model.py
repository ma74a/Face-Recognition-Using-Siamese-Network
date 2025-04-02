import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from utlis.config import *

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size=EMBEDDING_DIM, freeze_blocks=5):
        super(SiameseNetwork, self).__init__()
        self.backbone = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        
        for i, child in enumerate(self.backbone.children()):
            if i < freeze_blocks:
                for param in child.parameters():
                    param.requires_grad = False

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_size)
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # Normalize embeddings
        return x
    
    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2
    

# L=(1−Y) * D^2 + Y*max(0,m−D)^2 similar(0) - dissimilar(1)
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """Compute the contrastive loss between two embeddings."""
        distance = F.pairwise_distance(output1, output2, keepdim=True)

        # 0 for similar
        similar = (1 - label) * torch.pow(distance, 2)

        # 1 for dissimilar
        dissimilar = label * torch.pow(torch.clamp(self.margin-distance, min=0.0), 2)
        
        loss = torch.mean(similar + dissimilar)
        return loss
