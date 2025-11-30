# train_contrast.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_cbam_contrast import EfficientNetB1_CBAM_Contrastive
from loss_contrastive import SupConLoss

# -----------------------------
# 数据预处理
# -----------------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

transform_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dir = '/commondocument/group6/MEDagent_demo/EfficientNet/BreastCancer/train'
valid_dir = '/commondocument/group6/MEDagent_demo/EfficientNet/BreastCancer/valid'

train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform_valid)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

# -----------------------------
# 模型与优化器
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetB1_CBAM_Contrastive(num_classes=2, pretrained=True).to(device)

criterion_ce = nn.CrossEntropyLoss()
criterion_con = SupConLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

lambda_contrast = 0.1  # 对比学习权重
best_acc = 0.0         # 保存当前最优准确率

# -----------------------------
# 训练与验证
# -----------------------------
for epoch in range(10):
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # 分类损失
        logits = model(images)
        loss_ce = criterion_ce(logits, labels)

        # 对比损失（投影特征）
        z = model(images, contrastive=True)
        loss_con = criterion_con(z, labels)

        # 联合损失
        loss = loss_ce + lambda_contrast * loss_con
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Epoch [{epoch+1}/10] | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")

    # -----------------------------
    # 验证阶段
    # -----------------------------
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.4f}")

    # -----------------------------
    # ✅ 保存最优模型
    # -----------------------------
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_cbam_contrast.pth")
        print(f"✅ New best model saved (val_acc={val_acc:.4f})")

print("\nTraining completed. Best validation accuracy: {:.4f}".format(best_acc))
