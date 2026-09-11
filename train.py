import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from ofdm_system import OFDMSystem
from uwa_channel import UWAChannel
from impulsive_noise import ImpulsiveNoiseGenerator
from model import Attention1DCNN, ImpulsiveNoiseDetector


class UWADataset(Dataset):
    """Generates received baseband OFDM symbols corrupted by ambient +
    impulsive noise, in the TIME DOMAIN (paper Section III-A).

    The channel is diagonalized in frequency, so the noiseless baseband
    signal is r[n] = IFFT(H.X) with X = transmitted frequency-domain symbols.
    Composite noise u[n] = w[n] + i[n] (Gaussian Mixture, Eq. 2.2) is added
    in the time domain:  y[n] = r[n] + w[n] + i[n].

    INPUT  = Psi(y)     = [real(y^T); imag(y^T)]  (2 channels, length K)
    TARGET = Psi(r)     = [real(r^T);  imag(r^T)]  (clean baseband signal)

    The network must REMOVE the noise, not copy the input.
    """

    def __init__(self, n_samples, n_subcarriers=64, cp_length=16, snr_range=(0, 30),
                 sir_range=(-20, -10), p=0.02, flat_channel=False):
        self.n_samples = n_samples
        self.n_subcarriers = n_subcarriers
        self.cp_length = cp_length
        self.snr_range = snr_range
        self.sir_range = sir_range
        self.p = p
        self.flat_channel = flat_channel

        self.ofdm = OFDMSystem(n_subcarriers, cp_length, "QPSK")
        self.channel = UWAChannel(n_subcarriers, cp_length)
        self.noise_gen = ImpulsiveNoiseGenerator(n_subcarriers, cp_length)
        self.null_mask = self.ofdm.null_mask.copy()

        self.data = []
        self.labels = []
        self._generate()

    def _generate(self):
        for _ in range(self.n_samples):
            bits = self.ofdm.generate_random_bits(1)
            _, tx_symbols = self.ofdm.transmit(bits)

            if self.flat_channel:
                H = np.ones(self.n_subcarriers, dtype=complex)
            else:
                H = self.channel.generate_frequency_response(n_taps=6)

            rx_freq = self.channel.apply_channel_freq(tx_symbols, H)

            # Noiseless baseband received signal in the time domain
            r_time = np.fft.ifft(rx_freq, axis=-1)[0]  # (K,)

            snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])
            sir_db = np.random.uniform(self.sir_range[0], self.sir_range[1])
            y_time = self.noise_gen.add_gm_noise(
                r_time, snr_db=snr_db, sir_db=sir_db, p=self.p,
            )

            # INPUT: Psi(y) = [real(y^T); imag(y^T)]
            input_features = np.stack([
                np.real(y_time),
                np.imag(y_time),
            ]).astype(np.float32)

            # TARGET: Psi(r) = clean baseband signal without any noise
            target_features = np.stack([
                np.real(r_time),
                np.imag(r_time),
            ]).astype(np.float32)

            self.data.append(input_features)
            self.labels.append(target_features)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])


class JointTaskLoss(nn.Module):
    """Composite loss from paper Eq. (2):

    L_task = L_mse + lambda * L_null

    L_mse  = MSE(output, target) in the real/imag (2xK) time domain.
    L_null = mean over batch of the energy of the FFT-domain output mapping
             onto the NULL subcarriers:

             L_null_j = || Theta_n * F * Psi^{-1}(y~_j) ||^2 ,
             F = FFT, Theta_n = selection matrix of the null subcarriers.
    """

    def __init__(self, null_mask, lambda_null=1.0):
        super().__init__()
        self.lambda_null = lambda_null
        # (K,) float mask of null subcarrier positions (frequency domain)
        self.register_buffer(
            "null_mask",
            torch.tensor(np.asarray(null_mask, dtype=np.float32)),
        )
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        L_mse = self.mse(output, target)

        # Psi^{-1}(output): recombine real/imag channels into complex time-domain
        output_complex = torch.complex(output[:, 0, :], output[:, 1, :])  # (B, K)

        # F: FFT to the frequency domain, normalized so magnitudes are on the
        # same power scale as the time-domain signal (K = FFT window length).
        K = output_complex.shape[-1]
        output_fft = torch.fft.fft(output_complex, dim=-1) / K  # (B, K)

        # Theta_n: mean energy over null subcarriers (per-sample bins, then batch)
        null_energy = torch.mean(
            torch.abs(output_fft[:, self.null_mask.bool()]) ** 2
        )

        total = L_mse + self.lambda_null * null_energy
        return total, L_mse.item(), null_energy.item()


class Trainer:
    def __init__(self, model, null_mask, device="cpu", lr=1e-3, weight_decay=1e-4,
                 lambda_null=1.0):
        self.model = model.to(device)
        self.device = device
        self.criterion = JointTaskLoss(null_mask, lambda_null=lambda_null).to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for data, target in dataloader:
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss, _, _ = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                loss, _, _ = self.criterion(output, target)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader, n_epochs=100, save_path="model.pth"):
        best_val_loss = float("inf")

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}] Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                }, save_path)

        print(f"Best Validation Loss: {best_val_loss:.6f}")
        return self.train_losses, self.val_losses


def split_dataset(dataset, val_ratio=0.2, seed=42):
    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    n_val = int(n * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def create_source_domain_model(n_subcarriers=64, num_mab=4):
    return Attention1DCNN(input_length=n_subcarriers, num_mab=num_mab)


def create_target_domain_model(n_subcarriers=64, num_mab=4):
    return Attention1DCNN(input_length=n_subcarriers, num_mab=num_mab)


# Paper Section IV-B / Fig. 6: TF-Net is split into three parts during fine-tuning.
#   Part 1 (initial layers, lr=5e-4): conv_in, mab1, conv1, mab2
#   Part 2 (middle layers,  lr=1e-3): conv2, mab3, conv3, mab4  -> reinitialized
#   Part 3 (output layers,  lr=0):    conv_out                   -> retained, frozen
PART1_PREFIXES = ("conv_in.", "mab1.", "conv1.", "mab2.")
PART2_PREFIXES = ("conv2.", "mab3.", "conv3.", "mab4.")
PART3_PREFIXES = ("conv_out.",)


def transfer_learning_finetune(source_model, target_model, reinit_part2=True):
    """Copy source weights into target.

    Paper Section IV-B: Part 1 (initial layers) and Part 3 (output layers)
    retain the source-domain weights; Part 2 (middle layers) is randomly
    re-initialized so it re-learns target-domain feature alignments. The
    output layer (Part 3) is frozen during fine-tuning.

    If reinit_part2 is False, all layers retain the source weights (warm
    start); this is safer when the source model already generalizes to the
    target domain and the fine-tuning data budget is small.
    """
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    transferred = []
    for key in source_dict:
        if key in target_dict and source_dict[key].shape == target_dict[key].shape:
            target_dict[key] = source_dict[key].clone()
            transferred.append(key)
    target_model.load_state_dict(target_dict)

    if reinit_part2:
        with torch.no_grad():
            for name, param in target_model.named_parameters():
                if name.startswith(PART2_PREFIXES):
                    if param.dim() >= 2:
                        nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="leaky_relu")
                    else:
                        nn.init.zeros_(param)

    # Part 3 (output layer) frozen -> learning rate 0; never updated during fine-tuning
    for name, param in target_model.named_parameters():
        if name.startswith(PART3_PREFIXES):
            param.requires_grad = False

    n_frozen = sum(1 for p in target_model.parameters() if not p.requires_grad)
    n_total = sum(1 for p in target_model.parameters())
    print(f"Transferred {len(transferred)} params, frozen {n_frozen}/{n_total} "
          f"(Part 3 output layer only), Part 2 reinitialized: {reinit_part2}")

    return target_model


def transfer_learning_train(source_model_path, target_loader, val_loader,
                            n_subcarriers=64, n_epochs=20, device="cpu",
                            lambda_null=1.0, reinit_part2=False):
    source_model = create_source_domain_model(n_subcarriers)
    checkpoint = torch.load(source_model_path, map_location=device)
    source_model.load_state_dict(checkpoint["model_state_dict"])
    source_model.eval()

    target_model = create_target_domain_model(n_subcarriers)
    target_model = transfer_learning_finetune(source_model, target_model, reinit_part2=reinit_part2)

    null_mask = OFDMSystem(n_subcarriers).null_mask
    criterion = JointTaskLoss(null_mask, lambda_null=lambda_null).to(device)

    part1_params = []
    part2_params = []
    for name, param in target_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(PART2_PREFIXES):
            part2_params.append(param)
        else:
            part1_params.append(param)

    optimizer = optim.Adam([
        {"params": part1_params, "lr": 5e-4},
        {"params": part2_params, "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    save_path = "transfer_model.pth"

    for epoch in range(n_epochs):
        target_model.train()
        total_train = 0
        for data, target in target_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = target_model(data)
            loss, _, _ = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train += loss.item()
        scheduler.step()
        avg_train = total_train / max(len(target_loader), 1)

        target_model.eval()
        total_val = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                output = target_model(data)
                loss, _, _ = criterion(output, target)
                total_val += loss.item()
        avg_val = total_val / max(len(val_loader), 1)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if (epoch + 1) % 5 == 0:
            print(f"TL Epoch [{epoch+1}/{n_epochs}] Train: {avg_train:.6f} Val: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": target_model.state_dict(),
                "val_loss": avg_val,
            }, save_path)

    print(f"TL Best Validation Loss: {best_val_loss:.6f}")
    return target_model, train_losses, val_losses