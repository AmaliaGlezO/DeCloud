# discriminator.py
import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=13, features=64):
        super().__init__()
        # Input: (6 bandas nubladas + 1 mascara) + (6 bandas reconstruidas) = 13
        
        model = [
            nn.utils.spectral_norm(nn.Conv2d(in_channels, features, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(features, features*2, 4, stride=2, padding=1)),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(features*2, features*4, 4, stride=2, padding=1)),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(features*4, features*8, 4, stride=1, padding=1)),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features*8, 1, 4, stride=1, padding=1) 
            # Output: 30x30 patch map (aprox)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)