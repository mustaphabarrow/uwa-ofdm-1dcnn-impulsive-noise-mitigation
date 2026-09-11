import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

from ofdm_system import OFDMSystem
from uwa_channel import UWAChannel
from impulsive_noise import ImpulsiveNoiseGenerator
from model import Attention1DCNN, ImpulsiveNoiseDetector
from train import (
    UWADataset, Trainer, transfer_learning_train,
    create_source_domain_model, create_target_domain_model,
)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

N_SUBCARRIERS = 64
CP_LENGTH = 16
BANDWIDTH = 12e3
CARRIER_FREQ = 12e3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SOURCE_SAMPLES = 600
TARGET_SAMPLES = 400
BATCH_SIZE = 32
SOURCE_EPOCHS = 25
SCRATCH_EPOCHS = 25
TL_EPOCHS = 20
BER_TRIALS = 20

os.makedirs("results", exist_ok=True)


def generate_training_data():
    print("Generating source domain training data (AWGN channel)...")
    source_dataset = UWADataset(
        n_samples=SOURCE_SAMPLES, n_subcarriers=N_SUBCARRIERS,
        cp_length=CP_LENGTH, snr_range=(10, 30),
        impulsive_ratio=0.08, impulsive_power_ratio=20,
        flat_channel=True,
    )
    source_loader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return source_loader


def generate_target_data():
    print("Generating target domain data (UWA multipath channel)...")
    target_dataset = UWADataset(
        n_samples=TARGET_SAMPLES, n_subcarriers=N_SUBCARRIERS,
        cp_length=CP_LENGTH, snr_range=(0, 25),
        impulsive_ratio=0.15, impulsive_power_ratio=150,
        flat_channel=False,
    )
    target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return target_loader


def train_source_model(source_loader):
    print("\n=== Training Source Domain Model ===")
    model = create_source_domain_model(N_SUBCARRIERS)
    trainer = Trainer(model, device=DEVICE, lr=1e-3)
    train_losses, val_losses = trainer.train(
        source_loader, source_loader, n_epochs=SOURCE_EPOCHS,
        save_path="results/source_model.pth",
    )
    return model, train_losses, val_losses


def train_target_model(target_loader):
    print("\n=== Training Target Domain Model (from scratch) ===")
    model = create_target_domain_model(N_SUBCARRIERS)
    trainer = Trainer(model, device=DEVICE, lr=1e-3)
    train_losses, val_losses = trainer.train(
        target_loader, target_loader, n_epochs=SCRATCH_EPOCHS,
        save_path="results/target_model_scratch.pth",
    )
    return model, train_losses, val_losses


def transfer_learning(target_loader):
    print("\n=== Transfer Learning ===")
    model, train_losses, val_losses = transfer_learning_train(
        "results/source_model.pth", target_loader, target_loader,
        n_subcarriers=N_SUBCARRIERS, n_epochs=TL_EPOCHS, device=DEVICE,
    )
    return model, train_losses, val_losses


def evaluate_ber(model, snr_range, n_trials=50):
    ofdm = OFDMSystem(N_SUBCARRIERS, CP_LENGTH, "QPSK")
    channel = UWAChannel(N_SUBCARRIERS, CP_LENGTH)
    noise_gen = ImpulsiveNoiseGenerator(N_SUBCARRIERS, CP_LENGTH)

    bers_traditional = []
    bers_nn = []

    model.eval()

    for snr_db in snr_range:
        errors_trad = 0
        errors_nn = 0
        total_bits = 0

        for _ in range(n_trials):
            bits = ofdm.generate_random_bits(1)
            tx_time, tx_symbols = ofdm.transmit(bits)

            H = channel.generate_frequency_response(n_taps=6)
            rx_freq = channel.apply_channel_freq(tx_symbols, H)

            rx_noisy = noise_gen.add_impulsive_noise_block(
                rx_freq, snr_db=snr_db,
                impulsive_ratio=0.1, impulsive_power_ratio=100,
            )

            rx_equalized = rx_noisy / H

            rx_trad = ofdm.demodulate(rx_equalized)[0]
            errors_trad += np.sum(rx_trad != bits.flatten())
            total_bits += bits.size

            input_feat = np.stack([
                np.real(rx_equalized.flatten()),
                np.imag(rx_equalized.flatten()),
            ]).astype(np.float32)
            input_tensor = torch.tensor(input_feat).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)
                cleaned = output.cpu().numpy().flatten()
                noise_est = (cleaned[:N_SUBCARRIERS] + 1j * cleaned[N_SUBCARRIERS:]).reshape(1, N_SUBCARRIERS)
                denoised_symbols = rx_equalized - noise_est

            rx_nn = ofdm.demodulate(denoised_symbols)[0]
            errors_nn += np.sum(rx_nn != bits.flatten())

        bers_traditional.append(errors_trad / total_bits)
        bers_nn.append(errors_nn / total_bits)

    return bers_traditional, bers_nn


def plot_results(source_losses, target_losses, bers_trad, bers_nn,
                 bers_nn_tl, bers_nn_scratch, snr_range):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(source_losses, label="Source Domain", linewidth=2)
    axes[0, 0].plot(target_losses, label="Target (Scratch)", linewidth=2, linestyle="--")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss Convergence")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(snr_range, bers_trad, "o-", label="Traditional (No Denoising)", linewidth=2)
    axes[0, 1].plot(snr_range, bers_nn, "s-", label="1DCNN + MultiAttention", linewidth=2)
    axes[0, 1].set_xlabel("SNR (dB)")
    axes[0, 1].set_ylabel("BER")
    axes[0, 1].set_title("BER Performance: Traditional vs Neural Network")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(snr_range, bers_nn_tl, "D-", label="Transfer Learning", linewidth=2)
    axes[1, 0].plot(snr_range, bers_nn_scratch, "^-", label="Train from Scratch", linewidth=2)
    axes[1, 0].set_xlabel("SNR (dB)")
    axes[1, 0].set_ylabel("BER")
    axes[1, 0].set_title("Transfer Learning vs Training from Scratch")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(snr_range, bers_trad, "o-", label="Traditional", linewidth=2)
    axes[1, 1].plot(snr_range, bers_nn, "s-", label="1DCNN (Source)", linewidth=2)
    axes[1, 1].plot(snr_range, bers_nn_tl, "D-", label="Transfer Learning", linewidth=2)
    axes[1, 1].plot(snr_range, bers_nn_scratch, "^-", label="Target (Scratch)", linewidth=2)
    axes[1, 1].set_xlabel("SNR (dB)")
    axes[1, 1].set_ylabel("BER")
    axes[1, 1].set_title("Overall BER Comparison")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/ber_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Results saved to results/ber_performance.png")


def plot_denoising_example(model, n_subcarriers=N_SUBCARRIERS):
    ofdm = OFDMSystem(n_subcarriers, CP_LENGTH, "QPSK")
    channel = UWAChannel(n_subcarriers, CP_LENGTH)
    noise_gen = ImpulsiveNoiseGenerator(n_subcarriers, CP_LENGTH)

    bits = ofdm.generate_random_bits(1)
    tx_time, tx_symbols = ofdm.transmit(bits)

    H = channel.generate_frequency_response(n_taps=6)
    rx_freq = channel.apply_channel_freq(tx_symbols, H)

    snr_db = 10

    rx_noisy = noise_gen.add_impulsive_noise_block(
        rx_freq, snr_db=snr_db,
        impulsive_ratio=0.1, impulsive_power_ratio=100,
    )
    rx_equalized = rx_noisy / H

    input_feat = np.stack([
        np.real(rx_equalized.flatten()),
        np.imag(rx_equalized.flatten()),
    ]).astype(np.float32)
    input_tensor = torch.tensor(input_feat).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        cleaned = output.cpu().numpy().flatten()
        noise_est = (cleaned[:n_subcarriers] + 1j * cleaned[n_subcarriers:]).reshape(1, n_subcarriers)
        rx_denoised = rx_equalized - noise_est

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(np.abs(tx_symbols.flatten()), linewidth=1.5)
    axes[0, 0].set_title("Original OFDM Symbols (Magnitude)")
    axes[0, 0].set_xlabel("Subcarrier Index")
    axes[0, 0].set_ylabel("Magnitude")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(np.abs(rx_equalized.flatten()), linewidth=1.5, color="red")
    axes[0, 1].set_title("Received (Equalized, with Impulsive Noise)")
    axes[0, 1].set_xlabel("Subcarrier Index")
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(np.abs(rx_denoised.flatten()), linewidth=1.5, color="green")
    axes[1, 0].set_title("After 1DCNN Denoising")
    axes[1, 0].set_xlabel("Subcarrier Index")
    axes[1, 0].set_ylabel("Magnitude")
    axes[1, 0].grid(True, alpha=0.3)

    error_before = np.mean(np.abs(rx_equalized - tx_symbols) ** 2)
    error_after = np.mean(np.abs(rx_denoised - tx_symbols) ** 2)

    axes[1, 1].bar(["Noisy", "Denoised"], [error_before, error_after],
                    color=["red", "green"])
    axes[1, 1].set_title("MSE vs Clean Symbols")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/denoising_example.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Denoising example saved to results/denoising_example.png")


def plot_training_curves(tl_train, tl_val, scratch_train, scratch_val):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(tl_train, label="TL Train", linewidth=2)
    axes[0].plot(tl_val, label="TL Val", linewidth=2, linestyle="--")
    axes[0].set_title("Transfer Learning Training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(scratch_train, label="Scratch Train", linewidth=2)
    axes[1].plot(scratch_val, label="Scratch Val", linewidth=2, linestyle="--")
    axes[1].set_title("Training from Scratch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Training curves saved to results/training_curves.png")


def main():
    print("=" * 60)
    print("Impulsive Noise Mitigation for Underwater Acoustic OFDM")
    print("Based on 1DCNN with Multiattention Mechanism")
    print("and Transfer Learning")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    source_loader = generate_training_data()
    target_loader = generate_target_data()

    source_model, source_tl, source_vl = train_source_model(source_loader)

    target_model_scratch, scratch_tl, scratch_vl = train_target_model(target_loader)

    tl_model, tl_train_losses, tl_val_losses = transfer_learning(target_loader)

    snr_range = list(range(0, 31, 3))
    print("\n=== Evaluating BER Performance ===")

    bers_trad, bers_nn = evaluate_ber(source_model, snr_range, n_trials=BER_TRIALS)
    _, bers_nn_tl = evaluate_ber(tl_model, snr_range, n_trials=BER_TRIALS)
    _, bers_nn_scratch = evaluate_ber(target_model_scratch, snr_range, n_trials=BER_TRIALS)

    plot_results(
        source_tl, scratch_tl,
        bers_trad, bers_nn, bers_nn_tl, bers_nn_scratch,
        snr_range,
    )

    plot_training_curves(tl_train_losses, tl_val_losses, scratch_tl, scratch_vl)

    plot_denoising_example(tl_model)

    print("\n=== Summary ===")
    print(f"Source model training epochs: {len(source_tl)}")
    print(f"Target model (scratch) training epochs: {len(scratch_tl)}")
    print(f"Transfer learning epochs: {len(tl_train_losses)}")
    print(f"Final BER at 15dB - Traditional: {bers_trad[5]:.4f}")
    print(f"Final BER at 15dB - 1DCNN: {bers_nn[5]:.4f}")
    print(f"Final BER at 15dB - TL: {bers_nn_tl[5]:.4f}")
    print(f"Final BER at 15dB - Scratch: {bers_nn_scratch[5]:.4f}")

    torch.save(source_model.state_dict(), "results/final_source_model.pth")
    torch.save(tl_model.state_dict(), "results/final_tl_model.pth")
    torch.save(target_model_scratch.state_dict(), "results/final_scratch_model.pth")
    print("\nModels saved to results/")


if __name__ == "__main__":
    main()
