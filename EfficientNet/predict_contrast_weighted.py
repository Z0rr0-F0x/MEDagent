# predict_contrast_weighted.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from model_cbam_contrast import EfficientNetB1_CBAM_Contrastive

# ----------------------------
# 路径与设备
# ----------------------------
test_dir = '/commondocument/group6/MEDagent_demo/EfficientNet/BreastCancer/test'
model_path = '/commondocument/group6/MEDagent_demo/EfficientNet/Diagnostic Module/best_cbam_contrast.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 数据加载与预处理
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
class_names = test_dataset.classes

# ----------------------------
# 模型加载
# ----------------------------
model = EfficientNetB1_CBAM_Contrastive(num_classes=2, pretrained=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------
# 推理阶段
# ----------------------------
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # cancer 概率

# ----------------------------
# 结果输出
# ----------------------------
print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - EfficientNetB1_CBAM_Contrastive (Weighted)")
plt.tight_layout()
plt.show()

# 保存结果（可选）
np.savez("test_predictions_weighted.npz",
         labels=all_labels, preds=all_preds, probs=all_probs)
