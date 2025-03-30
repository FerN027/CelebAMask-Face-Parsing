import torch
import torch.nn as nn
import torch.nn.functional as F

class DualDilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation1=1, dilation2=1):
        super().__init__()
        mid_channels = out_channels // 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=dilation1,
                     dilation=dilation1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=dilation2,
                     dilation=dilation2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels*2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        out = torch.cat([feat1, feat2], dim=1)
        out = self.fusion(out)
        out = F.relu(out + self.shortcut(x))
        return out

class EnhancedPyramidPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        reduced_channels = in_channels // 8

        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
                nn.BatchNorm2d(reduced_channels),
                nn.ReLU()
            ) for size in [1, 2, 4]
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + reduced_channels*3, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        h, w = x.shape[2:]
        outputs = [x]

        for path in self.paths:
            feat = path(x)
            outputs.append(F.interpolate(feat, (h, w), mode='bilinear'))

        out = torch.cat(outputs, dim=1)
        return self.fusion(out)

class EnhancedFaceParsingNet(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )

        self.encoder1 = self._make_layer(24, 24, blocks=2, dilations=[[1,2], [2,4]])
        self.encoder2 = self._make_layer(24, 48, blocks=2, dilations=[[2,4], [4,8]], stride=2)
        self.encoder3 = self._make_layer(48, 96, blocks=2, dilations=[[4,8], [8,16]], stride=2)
        self.encoder4 = self._make_layer(96, 144, blocks=2, dilations=[[8,16], [16,1]], stride=2)
        self.encoder5 = self._make_layer(144, 192, blocks=2, dilations=[[16,1], [1,1]], stride=2)

        self.ppm = EnhancedPyramidPool(192)

        self.refine4 = nn.Sequential(
            nn.Conv2d(144, 72, 1),
            nn.BatchNorm2d(72),
            nn.ReLU()
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(96, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(48, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(24, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )

        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(192, 144, 4, stride=2, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(216, 96, 4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(144, 48, 4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(72, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )

        self.final = nn.Conv2d(24, num_classes, 1)

    def _make_layer(self, in_channels, out_channels, blocks, dilations, stride=1):
        layers = []
        layers.append(DualDilatedBlock(in_channels, out_channels,
                                     dilations[0][0], dilations[0][1]))

        for i in range(1, blocks):
            layers.append(DualDilatedBlock(out_channels, out_channels,
                                         dilations[i][0], dilations[i][1]))

        if stride > 1:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)          # [B, 24, 256, 256]
        e1 = self.encoder1(x1)      # [B, 24, 256, 256]
        e2 = self.encoder2(e1)      # [B, 48, 128, 128]
        e3 = self.encoder3(e2)      # [B, 96, 64, 64]
        e4 = self.encoder4(e3)      # [B, 144, 32, 32]
        e5 = self.encoder5(e4)      # [B, 192, 16, 16]

        # PPM at deeper level
        ppm = self.ppm(e5)          # [B, 192, 16, 16]

        # Decoder with skip connections
        d5 = self.decoder5(ppm)                                # [B, 144, 32, 32]
        d4 = self.decoder4(torch.cat([d5, self.refine4(e4)], 1))  # [B, 96, 64, 64]
        d3 = self.decoder3(torch.cat([d4, self.refine3(e3)], 1))  # [B, 48, 128, 128]
        d2 = self.decoder2(torch.cat([d3, self.refine2(e2)], 1))  # [B, 24, 256, 256]
        d1 = self.decoder1(torch.cat([d2, self.refine1(e1)], 1))  # [B, 24, 512, 512]

        return self.final(d1)
