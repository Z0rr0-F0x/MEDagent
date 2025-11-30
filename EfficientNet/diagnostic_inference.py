# diagnostic_inference_weighted.py
import torch
from torchvision import transforms
from PIL import Image
import os
from model_cbam_contrast import EfficientNetB1_CBAM_Contrastive

# ----------------------------
# 路径与设备
# ----------------------------
module_dir = "/commondocument/group6/MEDagent_demo/EfficientNet/Diagnostic Module"
image_path = os.path.join(module_dir, "test_image.png")
model_path = os.path.join(module_dir, "best_cbam_contrast.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 模型加载
# ----------------------------
model = EfficientNetB1_CBAM_Contrastive(num_classes=2, pretrained=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------
# 图片预处理（保持与训练一致）
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# ----------------------------
# 推理
# ----------------------------
with torch.no_grad():
    logits = model(image)
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    pred_prob = probs[0, pred_class].item()

# ----------------------------
# 输出结果
# ----------------------------
class_names = ['NoCancer', 'Cancer']
print(f"Predicted Class: {class_names[pred_class]}")
print(f"Probability: {pred_prob:.4f}")
print(f"All Class Probabilities: {probs.cpu().numpy()}")
