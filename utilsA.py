import torch
import numpy as np

def calculate_iou(preds, labels, threshold=0.5):
    """
    Calcula Intersection over Union.
    preds: Tensor (B, 1, H, W) con probabilidades 0-1
    labels: Tensor (B, 1, H, W) con ground truth 0 o 1
    """
    preds_bin = (preds > threshold).float()
    
    intersection = (preds_bin * labels).sum()
    union = preds_bin.sum() + labels.sum() - intersection
    
    # Evitar división por cero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    return (intersection / union).item()

def calculate_accuracy(preds, labels, threshold=0.5):
    preds_bin = (preds > threshold).float()
    correct = (preds_bin == labels).float().sum()
    total = torch.numel(labels)
    return (correct / total).item()