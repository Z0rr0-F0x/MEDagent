# train_contrast_weighted.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np

from model_cbam_contrast import EfficientNetB1_CBAM_Contrastive
from loss_contrastive import SupConLoss

# -----------------------------
# 1️⃣ 数据增强
# -----------------------------
# 对少数类（Cancer）增强更多
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dir = '/commondocument/group6/MEDagent_demo/EfficientNet/BreastCancer/train'
valid_dir = '/commondocument/group6/MEDagent_demo/EfficientNet/BreastCancer/valid'

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# -----------------------------
# 2️⃣ 类别加权 / WeightedRandomSampler
# -----------------------------
targets = [label for _, label in train_dataset.samples]
class_sample_count = np.array([targets.count(i) for i in range(len(train_dataset.classes))])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in targets])
samples_weight = torch.from_numpy(samples_weight).double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

# -----------------------------
# 3️⃣ 模型、损失、优化器
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetB1_CBAM_Contrastive(num_classes=2, pretrained=True).to(device)

# 类别加权交叉熵
class_weights = torch.tensor([class_sample_count[1]/class_sample_count[0], 1.0], dtype=torch.float).to(device)
criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
criterion_con = SupConLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

lambda_contrast = 0.05  # 对比损失权重降低，先保证分类能力
num_epochs = 30          # 增加训练轮次
best_acc = 0.0

# -----------------------------
# 4️⃣ 训练与验证
# -----------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # 分类损失
        logits = model(images)
        loss_ce = criterion_ce(logits, labels)

        # 对比学习损失
        z = model(images, contrastive=True)
        loss_con = criterion_con(z, labels)

        # 联合损失
        loss = loss_ce + lambda_contrast * loss_con
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    # -----------------------------
    # 验证阶段
    # -----------------------------
    model.eval()
    val_correct, val_total = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss_ce_val = criterion_ce(outputs, labels)
            val_loss += loss_ce_val.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss = val_loss / len(valid_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

    # -----------------------------
    # 保存最优模型
    # -----------------------------
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_cbam_contrast.pt")
        print(f"✅ New best model saved (val_acc={val_acc:.4f})")

print(f"\nTraining completed. Best validation accuracy: {best_acc:.4f}")
