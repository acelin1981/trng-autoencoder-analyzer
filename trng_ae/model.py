import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    """
    CNN autoencoder for 8x64x64 bit-plane inputs.
    - Input:  (B, 8, 64, 64) floats in {0,1}
    - Output: (B, 8, 64, 64) reconstructed
    """
    def __init__(self, in_ch: int = 8, base_ch: int = 32, bottleneck_ch: int = 64):
        super().__init__()

        # Encoder: 64->32->16
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(base_ch, bottleneck_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder: 16->32->64
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 16 -> 32
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),  # 32 -> 64
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        z = self.bottleneck(z)
        y = self.dec(z)
        return y
