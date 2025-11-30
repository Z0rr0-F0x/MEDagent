import torch
import torch.nn as nn
from torchvision import models
from cbam import CBAM

class EfficientNetB1_CBAM(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetB1_CBAM, self).__init__()
        weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
        self.base_model = models.efficientnet_b1(weights=weights)

        # ✅ 获取最后一层卷积的通道数（通过遍历）
        in_channels = None
        for layer in reversed(self.base_model.features):
            if hasattr(layer, "out_channels"):
                in_channels = layer.out_channels
                break
            elif hasattr(layer[-1], "out_channels"):
                in_channels = layer[-1].out_channels
                break
        if in_channels is None:
            raise ValueError("无法自动检测EfficientNet-B1最后卷积层通道数。")

        # ✅ 插入 CBAM
        self.cbam = CBAM(in_channels)

        # ✅ 修改分类头
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.cbam(x)  # 注意力增强
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.classifier(x)
        return x