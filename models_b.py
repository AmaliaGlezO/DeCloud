import torch
import torch.nn as nn
import torch.nn.functional as F

# --- OPCIÓN 1: GENERADOR U-NET (Pix2Pix) ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, spectral=True):
        super().__init__()
        # Spectral Normalization estabiliza GANs
        layer1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        layer2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        if spectral:
            layer1 = nn.utils.spectral_norm(layer1)
            layer2 = nn.utils.spectral_norm(layer2)
            
        self.conv = nn.Sequential(
            layer1, nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_ch),
            layer2, nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_ch)
        )
    def forward(self, x): return self.conv(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=7, out_channels=6, features=64):
        super().__init__()
        # Entrada: 6 bandas + 1 máscara = 7 canales
        self.enc1 = DoubleConv(in_channels, features)
        self.enc2 = DoubleConv(features, features*2)
        self.enc3 = DoubleConv(features*2, features*4)
        self.enc4 = DoubleConv(features*4, features*8)
        
        self.bottleneck = DoubleConv(features*8, features*16)
        
        self.up4 = DoubleConv(features*16 + features*8, features*8)
        self.up3 = DoubleConv(features*8 + features*4, features*4)
        self.up2 = DoubleConv(features*4 + features*2, features*2)
        self.up1 = DoubleConv(features*2 + features, features)
        
        self.final = nn.Conv2d(features, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.up4(torch.cat([self.upsample(b), e4], dim=1))
        d3 = self.up3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.up2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.up1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Salida Tanh para rango [-1, 1]
        return torch.tanh(self.final(d1))

# --- OPCIÓN 2: TRANSFORMER BOTTLENECK ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (Batch, Seq, Dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class ConvTransformerModel(nn.Module):
    """
    Encoder Convolucional -> Transformer Bottleneck -> Decoder Convolucional
    Reduce la imagen espacialmente para que el Transformer no explote en memoria.
    """
    def __init__(self, in_channels=7, out_channels=6, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        
        # Encoder: Reduce 256x256 -> 16x16 (Factor 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1), # 128x128
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, embed_dim, 4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(embed_dim), nn.LeakyReLU(0.2),
        )
        
        # Transformer Bottleneck
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Positional Encoding (Simplificado aprendible)
        self.pos_embed = nn.Parameter(torch.randn(1, 16*16, embed_dim) * 0.02)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, 2, 1), # 32x32
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 64x64
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 128x128
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1), # 256x256
            nn.Tanh()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # Encode
        feat = self.encoder(x) # (B, Embed, H/16, W/16)
        
        # Flatten para Transformer
        feat_flat = feat.flatten(2).transpose(1, 2) # (B, Seq, Embed)
        feat_flat = feat_flat + self.pos_embed
        
        # Transformer Process
        for layer in self.transformer_blocks:
            feat_flat = layer(feat_flat)
            
        # Reshape
        feat_restored = feat_flat.transpose(1, 2).view(b, -1, 16, 16)
        
        # Decode
        out = self.decoder(feat_restored)
        return out

def obtener_modelo_b(tipo="pix2pix", device="cpu"):
    if tipo == "pix2pix":
        return UNetGenerator().to(device)
    elif tipo == "transformer":
        return ConvTransformerModel().to(device)
    else:
        raise ValueError("Tipo desconocido")