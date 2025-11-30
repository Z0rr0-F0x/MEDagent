# predict_contrast.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model_cbam_contrast import EfficientNetB1_CBAM_Contrastive

# ----------------------------
# 1️⃣ 路径与设备
# ----------------------------
test_dir = "/commondocument/group6/MEDagent_demo/EfficientNet/BreastCancer/test"
model_path = "best_cbam_contrast.pth"  # 训练时保存的模型权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2️⃣ 数据加载
# ----------------------------
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

class_names = test_dataset.classes  # ['0_NoCancer', '1_Cancer']

# ----------------------------
# 3️⃣ 加载模型
# ----------------------------
model = EfficientNetB1_CBAM_Contrastive(num_classes=2, pretrained=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------
# 4️⃣ 模型推理
# ----------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ----------------------------
# 5️⃣ 分类报告
# ----------------------------
print("\n✅ Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# ----------------------------
# 6️⃣ 混淆矩阵
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6, 5)) 
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (EfficientNet-B1 + CBAM + SupCon)")
plt.tight_layout()
plt.savefig("confusion_matrix_cbam_contrast.png", dpi=300)
plt.show()