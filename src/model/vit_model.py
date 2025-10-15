import torch
import torch.nn as nn
import timm


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained)
        # replace head
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)


    def forward(self, x):
        return self.backbone(x)