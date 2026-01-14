# losses_b.py
import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Usamos VGG19 preentrenada hasta features.35 (conv4_4 aprox)
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg_sub = nn.Sequential(*list(vgg.children())[:36]).to(device).eval()
        for param in self.vgg_sub.parameters():
            param.requires_grad = False
            
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    def forward(self, x, y):
        # x, y shape: (B, 6, H, W). Tomamos solo los 3 primeros canales (RGB)
        x_rgb = x[:, :3, :, :]
        y_rgb = y[:, :3, :, :]
        
        # Normalizar para VGG (espera valores 0-1, asumo que entran -1 a 1 de Tanh)
        x_rgb = (x_rgb + 1) / 2.0
        y_rgb = (y_rgb + 1) / 2.0
        
        x_norm = (x_rgb - self.mean) / self.std
        y_norm = (y_rgb - self.mean) / self.std
        
        loss = nn.functional.l1_loss(self.vgg_sub(x_norm), self.vgg_sub(y_norm))
        return loss

class GANLoss(nn.Module):
    def __init__(self, mode='lsgan'):
        super().__init__()
        self.mode = mode
        if mode == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == 'lsgan':
            self.loss = nn.MSELoss()
            
    def __call__(self, prediction, target_is_real):
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)