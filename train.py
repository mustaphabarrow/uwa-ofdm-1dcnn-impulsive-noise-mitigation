import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from ofdm_system import OFDMSystem
from uwa_channel import UWAChannel
from impulsive_noise import ImpulsiveNoiseGenerator
from model import Attention1DCNN, ImpulsiveNoiseDetector


class UWADataset(Dataset):
    def __init__(self, n_samples, n_subcarriers=64, cp_length=16, snr_range=(0, 30),
                 impulsive_ratio=0.1, impulsive_power_ratio=100, flat_channel=False):
        self.n_samples = n_samples
        self.n_subcarriers = n_subcarriers
        self.cp_length = cp_length
        self.snr_range = snr_range
        self.impulsive_ratio = impulsive_ratio
        self.impulsive_power_ratio = impulsive_power_ratio
        self.flat_channel = flat_channel

        self.ofdm = OFDMSystem(n_subcarriers, cp_length, "QPSK")
        self.channel = UWAChannel(n_subcarriers, cp_length)
        self.noise_gen = ImpulsiveNoiseGenerator(n_subcarriers, cp_length)

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

            snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])
            tx_power = np.mean(np.abs(rx_freq) ** 2)
            snr_linear = 10 ** (snr_db / 10)
            awgn_power = tx_power / snr_linear
            noise = np.sqrt(awgn_power / 2) * (
                np.random.randn(*rx_freq.shape) + 1j * np.random.randn(*rx_freq.shape)
            )

            rx_noisy = self.noise_gen.add_impulsive_noise_block(
                rx_freq, snr_db=snr_db,
                impulsive_ratio=self.impulsive_ratio,
                impulsive_power_ratio=self.impulsive_power_ratio,
            )

            rx_equalized = rx_noisy / H

            noise_real = np.real(rx_equalized - tx_symbols)
            noise_imag = np.imag(rx_equalized - tx_symbols)
            noisy_real = np.real(rx_equalized)
            noisy_imag = np.imag(rx_equalized)

            input_features = np.stack([noisy_real.flatten(), noisy_imag.flatten()])
            target = np.stack([noise_real.flatten(), noise_imag.flatten()])

            self.data.append(input_features.astype(np.float32))
            self.labels.append(target.astype(np.float32))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])


class BERDataset(Dataset):
    def __init__(self, n_samples, n_subcarriers=64, cp_length=16,
                 snr_range=(0, 30), impulsive_ratio=0.1):
        self.n_samples = n_samples
        self.n_subcarriers = n_subcarriers
        self.cp_length = cp_length
        self.snr_range = snr_range
        self.impulsive_ratio = impulsive_ratio

        self.ofdm = OFDMSystem(n_subcarriers, cp_length, "QPSK")
        self.channel = UWAChannel(n_subcarriers, cp_length)
        self.noise_gen = ImpulsiveNoiseGenerator(n_subcarriers, cp_length)

        self.data = []
        self.labels = []
        self._generate()

    def _generate(self):
        for _ in range(self.n_samples):
            bits = self.ofdm.generate_random_bits(1)
            _, tx_symbols = self.ofdm.transmit(bits)

            H = self.channel.generate_frequency_response(n_taps=6)
            rx_freq = self.channel.apply_channel_freq(tx_symbols, H)

            snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])
            tx_power = np.mean(np.abs(rx_freq) ** 2)
            snr_linear = 10 ** (snr_db / 10)
            awgn_power = tx_power / snr_linear
            noise = np.sqrt(awgn_power / 2) * (
                np.random.randn(*rx_freq.shape) + 1j * np.random.randn(*rx_freq.shape)
            )

            rx_noisy = self.noise_gen.add_impulsive_noise_block(
                rx_freq, snr_db=snr_db,
                impulsive_ratio=self.impulsive_ratio,
                impulsive_power_ratio=100,
            )

            input_features = np.stack([
                np.real(rx_noisy.flatten()),
                np.imag(rx_noisy.flatten()),
            ]).astype(np.float32)
            target = np.array([
                1.0 if np.mean(np.abs(rx_noisy - rx_freq)) > np.std(noise) * 2 else 0.0
            ]).astype(np.float32)

            self.data.append(input_features)
            self.labels.append(target)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])


class Trainer:
    def __init__(self, model, device="cpu", lr=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
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
                loss = self.criterion(output, target)
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


def transfer_learning_finetune(pretrained_model, new_model, freeze_layers=3):
    pretrained_dict = pretrained_model.state_dict()
    new_dict = new_model.state_dict()

    pretrained_keys = list(pretrained_dict.keys())
    for i, key in enumerate(pretrained_keys):
        if i < freeze_layers * 2:
            if key in new_dict:
                new_dict[key] = pretrained_dict[key]
                for param in new_model.parameters():
                    if any(p is new_dict[key] for p in [param]):
                        param.requires_grad = False

    new_model.load_state_dict(new_dict)

    for name, param in new_model.named_parameters():
        if any(f"blocks.{i}" in name for i in range(freeze_layers)):
            param.requires_grad = False

    return new_model


def create_source_domain_model(n_subcarriers=64):
    return Attention1DCNN(input_length=n_subcarriers)


def create_target_domain_model(n_subcarriers=64):
    return Attention1DCNN(input_length=n_subcarriers)


def transfer_learning_train(source_model_path, target_loader, val_loader,
                           n_subcarriers=64, n_epochs=50, device="cpu"):
    source_model = create_source_domain_model(n_subcarriers)
    checkpoint = torch.load(source_model_path, map_location=device)
    source_model.load_state_dict(checkpoint["model_state_dict"])

    target_model = create_target_domain_model(n_subcarriers)
    target_model = transfer_learning_finetune(source_model, target_model, freeze_layers=2)

    for param in target_model.parameters():
        param.requires_grad = True

    trainer = Trainer(target_model, device=device, lr=1e-4)
    train_losses, val_losses = trainer.train(target_loader, val_loader,
                                              n_epochs=n_epochs,
                                              save_path="transfer_model.pth")
    return target_model, train_losses, val_losses
