import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attn = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(combined))
        return x * attn


class MultiAttentionBlock(nn.Module):
    def __init__(self, channels, seq_len, reduction=4):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size=7)
        self.norm = nn.LayerNorm([channels])
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv1d(channels * 4, channels, 1),
        )
        self.norm2 = nn.LayerNorm([channels])

    def forward(self, x):
        residual = x
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        x = self.norm((x + residual).permute(0, 2, 1)).permute(0, 2, 1)

        residual = x
        x = self.ffn(x)
        x = self.norm2((x + residual).permute(0, 2, 1)).permute(0, 2, 1)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class Attention1DCNN(nn.Module):
    def __init__(self, input_length=64, channels=None):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        self.input_length = input_length
        self.encoder_channels = channels
        self.decoder_channels = list(reversed(channels))

        self.encoder_convs = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsample = nn.ModuleList()

        in_ch = 2
        current_len = input_length
        for out_ch in channels:
            self.encoder_convs.append(ConvBlock(in_ch, out_ch, kernel_size=7, padding=3))
            self.encoder_attns.append(MultiAttentionBlock(out_ch, current_len))
            self.downsample.append(nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1))
            in_ch = out_ch
            current_len = current_len // 2

        self.bottleneck = nn.Sequential(
            ResBlock(channels[-1]),
            ResBlock(channels[-1]),
        )

        self.decoder_convs = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsample = nn.ModuleList()

        in_ch = channels[-1]
        current_len = input_length // (2 ** len(channels))
        for i, out_ch in enumerate(self.decoder_channels):
            self.upsample.append(nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1))
            self.decoder_convs.append(ConvBlock(out_ch * 2, out_ch, kernel_size=5, padding=2))
            self.decoder_attns.append(MultiAttentionBlock(out_ch, current_len * (2 ** (i + 1))))
            in_ch = out_ch

        self.output_conv = nn.Sequential(
            nn.Conv1d(channels[0], 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 2, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() == 3 and x.shape[1] == 1:
            x = torch.cat([x.real, x.imag], dim=1)

        skip_connections = []
        for conv, attn, down in zip(self.encoder_convs, self.encoder_attns, self.downsample):
            x = conv(x)
            x = attn(x)
            skip_connections.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for i, (up, conv, attn) in enumerate(
            zip(self.upsample, self.decoder_convs, self.decoder_attns)
        ):
            x = up(x)
            skip = skip_connections[-(i + 1)]
            if x.shape[-1] != skip.shape[-1]:
                x = x[:, :, : skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = conv(x)
            x = attn(x)

        x = self.output_conv(x)
        return x


class ImpulsiveNoiseDetector(nn.Module):
    def __init__(self, input_length=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.channel_attn = ChannelAttention(128, reduction=4)
        self.spatial_attn = SpatialAttention(kernel_size=5)

        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, 7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = torch.stack([x.real, x.imag], dim=1)
        elif x.shape[1] == 1:
            x = torch.cat([x.real, x.imag], dim=1)

        features = self.encoder(x)
        features = self.channel_attn(features)
        features = self.spatial_attn(features)
        mask = self.decoder(features)
        return mask.squeeze(1)
