# model_cbam_contrast.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from cbam import CBAM

# -----------------------------
# 1. EfficientNet + CBAM backbone
# -----------------------------
class EfficientNetB1_CBAM_Contrastive(nn.Module):
    def __init__(self, num_classes=2, feature_dim=128, pretrained=True):
        super(EfficientNetB1_CBAM_Contrastive, self).__init__()
        weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
        self.base_model = models.efficientnet_b1(weights=weights)

        # EfficientNet-B1 最后一层通道数
        in_channels = 1280
        self.cbam = CBAM(in_channels)

        # 分类层
        in_features = self.base_model.classifier[1].in_features
        self.fc = nn.Linear(in_features, num_classes)

        # 投影头 (projection head) —— 对比学习部分
        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, feature_dim)
        )

    def forward(self, x, return_features=False, contrastive=False):
        # Backbone
        x = self.base_model.features(x)
        x = self.cbam(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)

        if contrastive:
            # 输出用于对比学习的投影特征
            z = self.projector(x)
            z = F.normalize(z, dim=1)
            return z

        logits = self.fc(x)
        if return_features:
            return logits, x
        else:
            return logits
