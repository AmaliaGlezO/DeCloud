import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolución => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetMultibanda(nn.Module):
    def __init__(self, n_channels=6, n_classes=1, bilinear=True):
        super(UNetMultibanda, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024 // factor))

        # Decoder
        self.up1 = self._up_block(1024, 512 // factor, bilinear)
        self.up2 = self._up_block(512, 256 // factor, bilinear)
        self.up3 = self._up_block(256, 128 // factor, bilinear)
        self.up4 = self._up_block(128, 64, bilinear)
        
        # Salida
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _up_block(self, in_ch, out_ch, bilinear):
        """Bloque auxiliar para upsampling"""
        if bilinear:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv = DoubleConv(in_ch, out_ch) # Ajuste de canales ocurre en la conv
            return nn.Sequential(upsample, conv) # Simplificado para este ejemplo
            
        else:
            return nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder con Skip Connections (nota: implementación simplificada)
        # Para unir x4 y el upsample de x5, necesitamos asegurar tamaños iguales
        # Aquí asumimos imágenes múltiplos de 16 (ej. 256x256)
        
        # Upsampling manual para conectar con skip connections
        # (Implementación estándar de U-Net requeriría crop o pad si las dim no coinciden)
        
        def up_concat(x_dec, x_enc, layer):
             x_up = F.interpolate(x_dec, scale_factor=2, mode='bilinear', align_corners=True)
             # Concatenación en canal
             return layer(torch.cat([x_enc, x_up], dim=1))

        # Nota: Ajusté la lógica para usar bloques secuenciales estándar arriba, 
        # pero la conexión skip es crítica:
        
        # Corrección lógica de U-Net explícita:
        curr = x5
        curr = F.interpolate(curr, scale_factor=2, mode='bilinear', align_corners=True)
        curr = torch.cat([x4, curr], dim=1)
        curr = self.up1[1](curr) # Aplicar DoubleConv

        curr = F.interpolate(curr, scale_factor=2, mode='bilinear', align_corners=True)
        curr = torch.cat([x3, curr], dim=1)
        curr = self.up2[1](curr)

        curr = F.interpolate(curr, scale_factor=2, mode='bilinear', align_corners=True)
        curr = torch.cat([x2, curr], dim=1)
        curr = self.up3[1](curr)

        curr = F.interpolate(curr, scale_factor=2, mode='bilinear', align_corners=True)
        curr = torch.cat([x1, curr], dim=1)
        curr = self.up4[1](curr)

        logits = self.outc(curr)
        return self.sigmoid(logits)