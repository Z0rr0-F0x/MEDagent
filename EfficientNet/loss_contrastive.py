# loss_contrastive.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning"""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = 1
        anchor_feature = features
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, features.T),
            self.temperature)

        # 为数值稳定性去除对角线（自己与自己对比）
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
        mask = mask * logits_mask

        # log_softmax
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean()
        return loss
