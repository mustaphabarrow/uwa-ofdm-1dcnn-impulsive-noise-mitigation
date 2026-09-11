import torch
import torch.nn as nn
import torch.nn.functional as F


class CAB(nn.Module):
    """Channel Attention Block (Fig. 2 of paper).

    Uses shared Conv1d(1x1) MLP on both avg-pooled and max-pooled features,
    adds both attention maps, then applies residual connection.
    M_CAB = [σ(F_{1xC}(δ(F_{1xC/2}(AvgPool(M)))) + σ(F_{1xC}(δ(F_{1xC/2}(MaxPool(M)))))] · M + M
    """

    def __init__(self, C):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(C, C // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(C // 2, C, 1),
            nn.Sigmoid(),
        )

    def forward(self, M):
        avg_out = self.shared_mlp(self.avg_pool(M))
        max_out = self.shared_mlp(self.max_pool(M))
        attn = avg_out + max_out
        return attn * M + M


class SAB(nn.Module):
    """Spatial Attention Block (Fig. 3 of paper).

    Uses a single Conv1d(C, 1, kernel_size=1) + Sigmoid to produce a
    1-D spatial weight map, then applies residual connection.
    M_SAB = σ(F_{1x1}(M)) · M + M
    """

    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv1d(C, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, M):
        attn = self.sigmoid(self.conv(M))
        return attn * M + M


class MAB(nn.Module):
    """Multi-Attention Block (Fig. 4a of paper).

    CAB and SAB connected in series: Input → CAB → SAB → Output.
    """

    def __init__(self, C):
        super().__init__()
        self.cab = CAB(C)
        self.sab = SAB(C)

    def forward(self, M):
        return self.sab(self.cab(M))


class Attention1DCNN(nn.Module):
    """1DCNN-MAM architecture from paper (Fig. 4b, Table I).

    Sequential structure:
      Input(2, K)
      → Conv1d(2→C, k=3) + LeakyReLU
      → MAB-1 → Conv1d(C→C, k=3) + LeakyReLU
      → MAB-2 → Conv1d(C→C, k=3) + LeakyReLU
      → MAB-3 → Conv1d(C→C, k=3) + LeakyReLU
      → MAB-4 → Conv1d(C→2, k=3) + LeakyReLU
      Output(2, K)

    `num_mab` controls the number of MAB blocks (Fig. 5 ablation). For the
    default num_mab=4 the sub-module names match the transfer-learning part
    split exactly (mab1..mab4, conv1..conv3); for other depths the loop
    enumerates mab1..mabN / conv1..conv(N-1) with the same naming scheme.

    Operates on frequency-domain OFDM symbols split into real/imag channels.
    """

    def __init__(self, input_length=64, channels=None, dropout=0.1, num_mab=4):
        super().__init__()
        if channels is None:
            channels = 64
        C = channels
        self.C = C
        self.num_mab = num_mab

        self.conv_in = nn.Conv1d(2, C, 3, padding=1)
        for i in range(1, num_mab + 1):
            setattr(self, f"mab{i}", MAB(C))
            if i < num_mab:
                setattr(self, f"conv{i}", nn.Conv1d(C, C, 3, padding=1))
        self.conv_out = nn.Conv1d(C, 2, 3, padding=1)

        self.lrelu = nn.LeakyReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-init the output projection so the network starts at zero output.
        # This avoids huge initial FFT/null-loss magnitude and guarantees the
        # loss landscape is well-conditioned at epoch 0 (pure denoising task).
        if isinstance(self.conv_out, nn.Conv1d):
            nn.init.zeros_(self.conv_out.weight)
            if self.conv_out.bias is not None:
                nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() == 3 and x.shape[1] == 1:
            x = torch.cat([x.real, x.imag], dim=1)

        x = self.lrelu(self.conv_in(x))
        for i in range(1, self.num_mab + 1):
            x = getattr(self, f"mab{i}")(x)
            if i < self.num_mab:
                x = self.lrelu(getattr(self, f"conv{i}")(x))
        x = self.lrelu(self.conv_out(x))
        return x


class ImpulsiveNoiseDetector(nn.Module):
    """Separate lightweight detector for impulsive noise locations (not used in main pipeline)."""

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
        mask = self.decoder(features)
        return mask.squeeze(1)
