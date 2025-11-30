# predict_cbam.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from model_cbam import EfficientNetB1_CBAM  # ✅ 确保和 model_cbam.py 在同一路径下

# -------------------------------
# 1. 参数设置
# -------------------------------
test_dir = "/commondocument/group6/MEDagent_demo/EfficientNet/BreastCancer/test"
batch_size = 32
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_efficientnet_b1_cbam.pth"  # ✅ 你训练保存的模型路径

# -------------------------------
# 2. 数据预处理
# -------------------------------
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# 3. 模型加载
# -------------------------------
model = EfficientNetB1_CBAM(num_classes=num_classes, pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# -------------------------------
# 4. 推理与指标计算
# -------------------------------
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# -------------------------------
# 5. 输出分类报告
# -------------------------------
print("✅ Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

# -------------------------------
# 6. 混淆矩阵可视化
# -------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes)
plt.title("Confusion Matrix (EfficientNetB1 + CBAM)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_cbam.png", dpi=300)
plt.show()