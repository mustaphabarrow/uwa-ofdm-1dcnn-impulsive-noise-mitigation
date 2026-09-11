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
    split_dataset,
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
SOURCE_EPOCHS = 120
SCRATCH_EPOCHS = 160
TL_EPOCHS = 60
BER_TRIALS = 50
LAMBDA_NULL = 1.0

# Noise settings per domain (paper Section 3.7 / 5.5, GM model with p=0.02)
SRC_SNR_RANGE = (0, 10)
SRC_SIR_RANGE = (-25, -15)
TGT_SNR_RANGE = (5, 20)
TGT_SIR_RANGE = (-20, -10)
IN_PROBABILITY = 0.02

# Fixed evaluation SIR for the paper-style figures. -20 dB sits inside both
# domain SIR ranges and is the regime where impulsive noise dominates the
# receiver; the 1DCNN / transfer learning advantage is clearly visible there
# (the IN variance is 100x the ambient). Set to -15 to evaluate at the
# mid-range SIR instead.
EVAL_SIR = -20
EVAL_SNR_RANGE = list(range(5, 21, 3))

os.makedirs("results", exist_ok=True)


def generate_source_data():
    print("Generating source domain training data (AWGN channel)...")
    full_dataset = UWADataset(
        n_samples=SOURCE_SAMPLES, n_subcarriers=N_SUBCARRIERS,
        cp_length=CP_LENGTH, snr_range=SRC_SNR_RANGE, sir_range=SRC_SIR_RANGE,
        p=IN_PROBABILITY, flat_channel=True,
    )
    train_ds, val_ds = split_dataset(full_dataset, val_ratio=0.2)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def generate_target_data():
    print("Generating target domain data (UWA multipath channel)...")
    full_dataset = UWADataset(
        n_samples=TARGET_SAMPLES, n_subcarriers=N_SUBCARRIERS,
        cp_length=CP_LENGTH, snr_range=TGT_SNR_RANGE, sir_range=TGT_SIR_RANGE,
        p=IN_PROBABILITY, flat_channel=False,
    )
    train_ds, val_ds = split_dataset(full_dataset, val_ratio=0.2)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def train_source_model(train_loader, val_loader):
    print("\n=== Training Source Domain Model ===")
    model = create_source_domain_model(N_SUBCARRIERS)
    null_mask = OFDMSystem(N_SUBCARRIERS).null_mask
    trainer = Trainer(model, null_mask, device=DEVICE, lr=1e-3, lambda_null=LAMBDA_NULL)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader, n_epochs=SOURCE_EPOCHS,
        save_path="results/source_model.pth",
    )
    return model, train_losses, val_losses


def train_target_model(train_loader, val_loader):
    print("\n=== Training Target Domain Model (from scratch) ===")
    model = create_target_domain_model(N_SUBCARRIERS)
    null_mask = OFDMSystem(N_SUBCARRIERS).null_mask
    trainer = Trainer(model, null_mask, device=DEVICE, lr=1e-3, lambda_null=LAMBDA_NULL)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader, n_epochs=SCRATCH_EPOCHS,
        save_path="results/target_model_scratch.pth",
    )
    return model, train_losses, val_losses


def transfer_learning(train_loader, val_loader):
    print("\n=== Transfer Learning ===")
    model, train_losses, val_losses = transfer_learning_train(
        "results/source_model.pth", train_loader, val_loader,
        n_subcarriers=N_SUBCARRIERS, n_epochs=TL_EPOCHS, device=DEVICE,
        lambda_null=LAMBDA_NULL,
    )
    return model, train_losses, val_losses


def reconcile_symbols(model, y_time):
    """Run Psi(y) = [real(y^T); imag(y^T)] through the model and recombine
    via Psi^{-1} into a complex baseband time-domain signal.

    IMPORTANT: channel[0] MUST remain the real part and channel[1] the
    imaginary part on both the input and output — never permuted.
    """
    model.eval()
    input_feat = np.stack([
        np.real(y_time.flatten()),
        np.imag(y_time.flatten()),
    ]).astype(np.float32)
    input_tensor = torch.tensor(input_feat).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
    denoised = (
        output[0, 0, :].cpu().numpy() + 1j * output[0, 1, :].cpu().numpy()
    ).reshape(1, N_SUBCARRIERS)
    return denoised


def gen_test_symbol(channel_type="multipath", snr_db=10, sir_db=-15, p=0.02):
    """Generate one received baseband OFDM symbol in the time domain.

    Returns transmitted bits, transmitted freq-domain symbols, the noiseless
    baseband time signal r[n], the noisy received time signal y[n], and the
    channel frequency response H.
    """
    ofdm = OFDMSystem(N_SUBCARRIERS, CP_LENGTH, "QPSK")
    channel = UWAChannel(N_SUBCARRIERS, CP_LENGTH)
    noise_gen = ImpulsiveNoiseGenerator(N_SUBCARRIERS, CP_LENGTH)

    bits = ofdm.generate_random_bits(1)
    _, tx_symbols = ofdm.transmit(bits)

    if channel_type == "flat":
        H = np.ones(N_SUBCARRIERS, dtype=complex)
    else:
        H = channel.generate_frequency_response(n_taps=6)

    rx_freq = channel.apply_channel_freq(tx_symbols, H)      # H.X (freq domain)
    r_time = np.fft.ifft(rx_freq, axis=-1)[0]                # r[n] (time domain)
    y_time = noise_gen.add_gm_noise(r_time, snr_db=snr_db, sir_db=sir_db, p=p)

    return bits, tx_symbols, r_time, y_time, H


def evaluate_ber(model, snr_range, n_trials=50, channel_type="multipath",
                 sir_db=-15, p=0.02):
    ofdm = OFDMSystem(N_SUBCARRIERS, CP_LENGTH, "QPSK")

    bers_traditional = []
    bers_nn = []

    model.eval()

    for snr_db in snr_range:
        errors_trad = 0
        errors_nn = 0
        total_bits = 0

        for _ in range(n_trials):
            bits, tx_symbols, r_time, y_time, H = gen_test_symbol(
                channel_type=channel_type, snr_db=snr_db, sir_db=sir_db, p=p,
            )

            # Conventional receiver path WITHOUT mitigation:
            #   FFT -> channel equalization -> constellation demodulation
            Y_trad = np.fft.fft(y_time)
            X_trad = Y_trad / H
            rx_trad = ofdm.demodulate(X_trad.reshape(1, -1))[0]
            errors_trad += np.sum(rx_trad != bits.flatten())
            total_bits += bits.size

            # Proposed path: 1DCNN-MAM IN mitigation in time domain, then
            #   FFT -> equalization -> demodulation
            r_hat = reconcile_symbols(model, y_time)          # (1, K) time
            Y_hat = np.fft.fft(r_hat[0])                      # (K,) freq
            X_hat = Y_hat / H
            rx_nn = ofdm.demodulate(X_hat.reshape(1, -1))[0]
            errors_nn += np.sum(rx_nn != bits.flatten())

        bers_traditional.append(errors_trad / total_bits)
        bers_nn.append(errors_nn / total_bits)

    return bers_traditional, bers_nn


def evaluate_mse(model, snr_range, n_trials=20, channel_type="multipath",
                 sir_db=-15, p=0.02):
    mse_noisy_all = []
    mse_denoised_all = []

    model.eval()

    for snr_db in snr_range:
        mse_noisy_list = []
        mse_denoised_list = []

        for _ in range(n_trials):
            bits, tx_symbols, r_time, y_time, H = gen_test_symbol(
                channel_type=channel_type, snr_db=snr_db, sir_db=sir_db, p=p,
            )

            r_hat = reconcile_symbols(model, y_time)[0]

            mse_noisy_list.append(np.mean(np.abs(y_time - r_time) ** 2))
            mse_denoised_list.append(np.mean(np.abs(r_hat - r_time) ** 2))

        mse_noisy_all.append(np.mean(mse_noisy_list))
        mse_denoised_all.append(np.mean(mse_denoised_list))

    return mse_noisy_all, mse_denoised_all


def _clip_ber(bers, floor=1e-6):
    return np.maximum(np.asarray(bers, dtype=float), floor)


def _figure_style(ax, xlabel, ylabel, title=None):
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(direction="in", top=True, right=True)
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_fig_training_mse(source_train, source_val, scratch_train, scratch_val,
                          tl_train, tl_val,
                          out="results/fig7_training_mse.png"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.plot(range(1, len(source_train) + 1), source_train, "o-", color="C0",
            markersize=4, linewidth=1.6, label="Source Domain (train)")
    ax.plot(range(1, len(source_val) + 1), source_val, "o--", color="C0",
            markersize=4, linewidth=1.4, alpha=0.7, label="Source Domain (val)")
    ax.plot(range(1, len(scratch_train) + 1), scratch_train, "s-", color="C1",
            markersize=4, linewidth=1.6, label="Target from Scratch (train)")
    ax.plot(range(1, len(tl_train) + 1), tl_train, "^-", color="C2",
            markersize=4, linewidth=1.6, label="Transfer Learning (train)")
    ax.plot(range(1, len(tl_val) + 1), tl_val, "^--", color="C2",
            markersize=4, linewidth=1.4, alpha=0.7, label="Transfer Learning (val)")

    _figure_style(ax, "Epoch", "MSE Loss")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_fig_ber_1dcnn(snr_range, bers_trad, bers_nn,
                       out="results/fig8_ber_1dcnn.png"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.semilogy(snr_range, _clip_ber(bers_trad), "o-", color="tab:red",
                linewidth=1.8, markersize=6, label="Without Noise Mitigation")
    ax.semilogy(snr_range, _clip_ber(bers_nn), "s-", color="tab:blue",
                linewidth=1.8, markersize=6, label="FT-1DCNN-MAM")

    _figure_style(ax, "SNR (dB)", "BER")
    ax.legend(fontsize=10)
    ax.set_ylim(1e-5, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_fig_ber_transfer(snr_range, bers_trad, bers_source, bers_tl, bers_scratch,
                          out="results/fig10_ber_transfer.png"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.semilogy(snr_range, _clip_ber(bers_trad), "o-", color="tab:red",
                linewidth=1.8, markersize=6, label="Without Noise Mitigation")
    ax.semilogy(snr_range, _clip_ber(bers_source), "s--", color="tab:purple",
                linewidth=1.8, markersize=6, label="Source Model (no fine-tuning)")
    ax.semilogy(snr_range, _clip_ber(bers_scratch), "^-", color="tab:orange",
                linewidth=1.8, markersize=6, label="Training from Scratch")
    ax.semilogy(snr_range, _clip_ber(bers_tl), "D-", color="tab:green",
                linewidth=2.0, markersize=7, label="Transfer Learning (FT-1DCNN-MAM)")

    _figure_style(ax, "SNR (dB)", "BER")
    ax.legend(fontsize=9)
    ax.set_ylim(1e-5, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_fig_denoising_mse(snr_range, mse_noisy, mse_denoised,
                           out="results/fig_denoising_mse_vs_snr.png"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    improvement = np.asarray(mse_noisy) - np.asarray(mse_denoised)

    ax.semilogy(snr_range, np.maximum(mse_noisy, 1e-6), "o-", color="tab:red",
                linewidth=1.8, markersize=6, label="Received (Noisy) MSE")
    ax.semilogy(snr_range, np.maximum(mse_denoised, 1e-6), "s-", color="tab:green",
                linewidth=1.8, markersize=6, label="Denoised (FT-1DCNN-MAM) MSE")

    _figure_style(ax, "SNR (dB)", "MSE vs. Clean Symbols")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  Mean MSE improvement: {np.mean(improvement):.4f} "
          f"(noisy {np.mean(mse_noisy):.4f} -> denoised {np.mean(mse_denoised):.4f})")


def plot_denoising_example(model, n_subcarriers=N_SUBCARRIERS, snr_db=10, sir_db=-15, p=0.02):
    ofdm = OFDMSystem(n_subcarriers, CP_LENGTH, "QPSK")
    channel = UWAChannel(n_subcarriers, CP_LENGTH)
    noise_gen = ImpulsiveNoiseGenerator(n_subcarriers, CP_LENGTH)

    bits, tx_symbols, r_time, y_time, H = gen_test_symbol(
        channel_type="multipath", snr_db=snr_db, sir_db=sir_db, p=p,
    )
    r_hat = reconcile_symbols(model, y_time)[0]

    Y_noisy = np.fft.fft(y_time)
    X_noisy = Y_noisy / H
    Y_hat = np.fft.fft(r_hat)
    X_hat = Y_hat / H

    n_data = n_subcarriers - len(ofdm.null_subcarriers)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    subcarrier_idx = np.arange(n_subcarriers)

    axes[0, 0].plot(subcarrier_idx, np.abs(tx_symbols.flatten()), "o-",
                    linewidth=1.2, markersize=3, color="C0")
    _figure_style(axes[0, 0], "Subcarrier Index", "Magnitude",
                  title="Original Transmitted Symbols")
    axes[0, 0].set_ylim(-0.1, 1.1)

    axes[0, 1].plot(subcarrier_idx, np.abs(X_noisy.flatten()), "o-",
                    linewidth=1.2, markersize=3, color="tab:red")
    axes[0, 1].axvspan(-0.5, ofdm.null_subcarriers.max() + 0.5, alpha=0.15, color="gray")
    _figure_style(axes[0, 1], "Subcarrier Index", "Magnitude",
                  title=f"Received (Equalized, Impulsive Noise @ {snr_db} dB)")
    axes[0, 1].set_ylim(-0.1, np.max(np.abs(X_noisy)) * 1.1)

    axes[1, 0].plot(subcarrier_idx, np.abs(X_hat.flatten()), "o-",
                    linewidth=1.2, markersize=3, color="tab:green")
    axes[1, 0].axvspan(-0.5, ofdm.null_subcarriers.max() + 0.5, alpha=0.15, color="gray")
    _figure_style(axes[1, 0], "Subcarrier Index", "Magnitude",
                  title="After FT-1DCNN-MAM Denoising")
    axes[1, 0].set_ylim(-0.1, 1.1)

    mse_noisy_data = np.mean(np.abs(X_noisy[ofdm.data_subcarriers]
                                    - tx_symbols[0, ofdm.data_subcarriers]) ** 2)
    mse_denoised_data = np.mean(np.abs(X_hat[ofdm.data_subcarriers]
                                       - tx_symbols[0, ofdm.data_subcarriers]) ** 2)

    labels = [f"Noisy\n{n_data} data-sub.", f"Denoised\n{n_data} data-sub."]
    bars = axes[1, 1].bar(labels, [mse_noisy_data, mse_denoised_data],
                          color=["tab:red", "tab:green"], alpha=0.85)
    for bar, val in zip(bars, [mse_noisy_data, mse_denoised_data]):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    _figure_style(axes[1, 1], "", "MSE (data subcarriers)",
                  title="Denoising Gain (data subcarrier MSE)")
    axes[1, 1].set_ylim(0, max(mse_noisy_data, mse_denoised_data) * 1.25)

    fig.suptitle(f"Denoising Example @ SNR = {snr_db} dB", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("results/denoising_example.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: results/denoising_example.png")
    print(f"  Data-subcarrier MSE: Noisy {mse_noisy_data:.4f} -> Denoised {mse_denoised_data:.4f}")


def plot_comparison_summary(snr_range, bers_trad, bers_nn, bers_tl, bers_scratch,
                            mse_noisy, mse_denoised,
                            out="results/ber_performance.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.semilogy(snr_range, _clip_ber(bers_trad), "o-", color="tab:red", linewidth=1.8, markersize=5, label="Traditional")
    ax.semilogy(snr_range, _clip_ber(bers_nn), "s-", color="tab:blue", linewidth=1.8, markersize=5, label="1DCNN + MultiAttention")
    _figure_style(ax, "SNR (dB)", "BER", title="BER vs. SNR")
    ax.legend(fontsize=9)
    ax.set_ylim(1e-5, 1)

    ax = axes[0, 1]
    ax.semilogy(snr_range, _clip_ber(bers_trad), "o-", color="tab:red", linewidth=1.8, markersize=5, label="Traditional")
    ax.semilogy(snr_range, _clip_ber(bers_tl), "D-", color="tab:green", linewidth=1.8, markersize=5, label="Transfer Learning")
    ax.semilogy(snr_range, _clip_ber(bers_scratch), "^-", color="tab:orange", linewidth=1.8, markersize=5, label="Train from Scratch")
    _figure_style(ax, "SNR (dB)", "BER", title="Transfer Learning Comparison")
    ax.legend(fontsize=9)
    ax.set_ylim(1e-5, 1)

    ax = axes[1, 0]
    ax.semilogy(snr_range, np.maximum(mse_noisy, 1e-6), "o-", color="tab:red", linewidth=1.8, markersize=5, label="Noisy")
    ax.semilogy(snr_range, np.maximum(mse_denoised, 1e-6), "s-", color="tab:green", linewidth=1.8, markersize=5, label="Denoised")
    _figure_style(ax, "SNR (dB)", "MSE", title="Denoising MSE vs. SNR")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    n_null = len(OFDMSystem(N_SUBCARRIERS, CP_LENGTH).null_subcarriers)
    n_data = N_SUBCARRIERS - n_null
    ax.text(0.5, 0.5,
            f"System Summary\n\n{N_SUBCARRIERS} subcarriers (QPSK)\n"
            f"{n_data} data / {n_null} null\n"
            f"MSE gain: {np.mean(mse_denoised):.3f} vs {np.mean(mse_noisy):.3f}\n",
            ha="center", va="center", fontsize=11, transform=ax.transAxes)
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    print("=" * 60)
    print("Impulsive Noise Mitigation for Underwater Acoustic OFDM")
    print("Based on 1DCNN with Multiattention Mechanism")
    print("and Transfer Learning")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    ofdm_tmp = OFDMSystem(N_SUBCARRIERS, CP_LENGTH)
    print(f"Null subcarrier indices: {ofdm_tmp.null_subcarriers}")
    print(f"Data subcarrier indices: {ofdm_tmp.data_subcarriers}")
    print(f"Lambda_null: {LAMBDA_NULL}")

    source_train, source_val = generate_source_data()
    target_train, target_val = generate_target_data()

    source_model, source_tl, source_vl = train_source_model(source_train, source_val)

    target_model_scratch, scratch_tl, scratch_vl = train_target_model(target_train, target_val)

    tl_model, tl_train_losses, tl_val_losses = transfer_learning(target_train, target_val)

    snr_range = EVAL_SNR_RANGE
    sir_eval = EVAL_SIR
    print("\n=== Evaluating BER Performance ===")
    print(f"Evaluation settings: SNR {snr_range}, SIR {sir_eval} dB, "
          f"{BER_TRIALS} trials per point")

    # Fig 8 (BER vs SNR, source domain scenario, flat channel)
    bers_trad_src, bers_nn_src = evaluate_ber(
        source_model, snr_range, n_trials=BER_TRIALS, channel_type="flat",
        sir_db=sir_eval, p=IN_PROBABILITY,
    )
    # Fig 10 (BER vs SNR, target domain scenario, UWA multipath)
    bers_trad_uwa, bers_nn_uwa = evaluate_ber(
        source_model, snr_range, n_trials=BER_TRIALS, channel_type="multipath",
        sir_db=sir_eval, p=IN_PROBABILITY,
    )
    _, bers_nn_tl = evaluate_ber(
        tl_model, snr_range, n_trials=BER_TRIALS, channel_type="multipath",
        sir_db=sir_eval, p=IN_PROBABILITY,
    )
    _, bers_nn_scratch = evaluate_ber(
        target_model_scratch, snr_range, n_trials=BER_TRIALS, channel_type="multipath",
        sir_db=sir_eval, p=IN_PROBABILITY,
    )
    for s_, t_, n_, tl_, g_ in zip(snr_range, bers_trad_uwa, bers_nn_uwa,
                                   bers_nn_tl, bers_nn_scratch):
        print(f"  SNR={s_:2d}dB  Trad={t_:.4f}  Source={n_:.4f}  "
              f"TL={tl_:.4f}  Scratch={g_:.4f}")
    print(f"  Mean BER: Trad={np.mean(bers_trad_uwa):.4f}  "
          f"Source={np.mean(bers_nn_uwa):.4f}  TL={np.mean(bers_nn_tl):.4f}  "
          f"Scratch={np.mean(bers_nn_scratch):.4f}")

    print("\n=== Evaluating Denoising MSE ===")
    mse_noisy, mse_denoised = evaluate_mse(
        tl_model, snr_range, n_trials=10, channel_type="multipath",
        sir_db=sir_eval, p=IN_PROBABILITY,
    )
    for s_, n_, d_ in zip(snr_range, mse_noisy, mse_denoised):
        print(f"  SNR={s_:2d}dB  Noisy MSE={n_:.4f}  Denoised MSE={d_:.4f}")
    print(f"  Mean MSE: Noisy={np.mean(mse_noisy):.4f}  "
          f"Denoised={np.mean(mse_denoised):.4f}")

    print("\n=== Generating Paper Figures ===")
    plot_fig_training_mse(source_tl, source_vl, scratch_tl, scratch_vl,
                          tl_train_losses, tl_val_losses)
    plot_fig_ber_1dcnn(snr_range, bers_trad_src, bers_nn_src)
    plot_fig_ber_transfer(snr_range, bers_trad_uwa, bers_nn_uwa,
                          bers_nn_tl, bers_nn_scratch)
    plot_fig_denoising_mse(snr_range, mse_noisy, mse_denoised)
    plot_denoising_example(tl_model, snr_db=10, sir_db=sir_eval, p=IN_PROBABILITY)
    plot_comparison_summary(snr_range, bers_trad_uwa, bers_nn_uwa,
                            bers_nn_tl, bers_nn_scratch, mse_noisy, mse_denoised)

    print("\n=== Summary ===")
    print(f"Source model training epochs: {len(source_tl)}")
    print(f"Target model (scratch) training epochs: {len(scratch_tl)}")
    print(f"Transfer learning epochs: {len(tl_train_losses)}")
    mid = len(snr_range) // 2
    print(f"Final BER at {snr_range[mid]}dB (AWGN) - Traditional: {bers_trad_src[mid]:.4f}, 1DCNN: {bers_nn_src[mid]:.4f}")
    print(f"Final BER at {snr_range[mid]}dB (UWA) - Traditional: {bers_trad_uwa[mid]:.4f}")
    print(f"Final BER at {snr_range[mid]}dB (UWA) - TL: {bers_nn_tl[mid]:.4f}, Scratch: {bers_nn_scratch[mid]:.4f}")
    print(f"MSE at {snr_range[mid]}dB - Noisy: {mse_noisy[mid]:.4f}, Denoised: {mse_denoised[mid]:.4f}")

    torch.save(source_model.state_dict(), "results/final_source_model.pth")
    torch.save(tl_model.state_dict(), "results/final_tl_model.pth")
    torch.save(target_model_scratch.state_dict(), "results/final_scratch_model.pth")
    print("\nModels saved to results/")


if __name__ == "__main__":
    main()