"""Fig. 5-style ablation: effect of MAB depth on BER.

Trains 1DCNN-MAM variants with num_mab in {1, 2, 3, 4} on the target
(multipath UWA) domain, then evaluates each at EVAL_SIR = -15 dB (the
secondary evaluation SIR) across the SNR sweep, against the traditional
no-mitigation baseline. Produces results/fig5_mab_depth_ablation.png.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ofdm_system import OFDMSystem
from train import UWADataset, Trainer, create_target_domain_model, split_dataset
from main import (
    evaluate_ber, evaluate_mse, N_SUBCARRIERS, CP_LENGTH, DEVICE,
)

os.makedirs("results", exist_ok=True)

ABLATION_SIR = -15
N_SAMPLES = 400
N_EPOCHS = 120
BATCH_SIZE = 32
SNR_RANGE = list(range(5, 21, 3))
MAB_DEPTHS = [1, 2, 3, 4]
N_TRIALS = 100


def main():
    print("=" * 60)
    print("Fig. 5-style ablation: MAB depth vs BER")
    print(f"Evaluation SIR = {ABLATION_SIR} dB, SNR {SNR_RANGE}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Shared target-domain dataset so every depth sees identical data.
    ds = UWADataset(N_SAMPLES, N_SUBCARRIERS, CP_LENGTH,
                    snr_range=(5, 20), sir_range=(-20, -10),
                    p=0.02, flat_channel=False)
    train_ds, val_ds = split_dataset(ds, val_ratio=0.2)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    null_mask = OFDMSystem(N_SUBCARRIERS).null_mask

    depth_bers = {}
    depth_val = {}
    plot_only = "--plot-only" in sys.argv
    if not plot_only:
        for depth in MAB_DEPTHS:
            print(f"\n=== Training 1DCNN-MAM with {depth} MAB block(s) ===")
            model = create_target_domain_model(N_SUBCARRIERS, num_mab=depth)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params / 1e3:.1f}K")
            _, val_losses = Trainer(model, null_mask, device=DEVICE,
                                    lr=1e-3, lambda_null=1.0).train(
                train_loader, val_loader, n_epochs=N_EPOCHS,
                save_path=f"results/ablation_mab{depth}.pth")
            depth_val[depth] = val_losses[-1]
    else:
        print("--plot-only: loading trained checkpoints from results/.",)

    # Traditional (no mitigation) baseline on the same setting.
    ofdm = OFDMSystem(N_SUBCARRIERS, CP_LENGTH, "QPSK")
    trad = _traditional_ber(ofdm, SNR_RANGE, N_TRIALS)

    # Evaluate each depth on the specified SIR (also reachable via --plot-only).
    print("\n=== Evaluating each MAB depth ===")
    for depth in MAB_DEPTHS:
        print(f"  Evaluating {depth} MAB model...")
        model = create_target_domain_model(N_SUBCARRIERS, num_mab=depth)
        ckpt = torch.load(f"results/ablation_mab{depth}.pth", map_location=DEVICE)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
        model.eval()
        _, bers = evaluate_ber(model, SNR_RANGE, n_trials=N_TRIALS,
                               channel_type="multipath", sir_db=ABLATION_SIR,
                               p=0.02)
        depth_bers[depth] = bers
        if depth not in depth_val:
            depth_val[depth] = float(ckpt.get("val_loss", 0.0)) if isinstance(ckpt, dict) else 0.0

    print("\n=== BER vs SNR at SIR = -15 dB ===")
    header = "SNR(dB)" + "".join(f"  {d}MAB" for d in MAB_DEPTHS) + "  Trad"
    print(header)
    for i, snr_db in enumerate(SNR_RANGE):
        row = f"{snr_db:6d} " + "".join(
            f"  {depth_bers[d][i]:.4f}" for d in MAB_DEPTHS)
        print(f"{row}  {trad[i]:.4f}")

    print("\n=== Mean BER (lower is better) ===")
    means = {d: float(np.mean(depth_bers[d])) for d in MAB_DEPTHS}
    for d in MAB_DEPTHS:
        print(f"  {d} MAB(s): {means[d]:.4f}  (val_loss={depth_val[d]:.5f})")
    print(f"  Traditional: {np.mean(trad):.4f}")
    best = min(means, key=means.get)
    print(f"  Best depth by mean BER: {best} MAB(s)")

    _plot_fig5(SNR_RANGE, depth_bers, trad, means,
               f"{ABLATION_SIR} dB")
    print(f"\nSaved: results/fig5_mab_depth_ablation.png")


def _traditional_ber(ofdm, snr_range, n_trials):
    from main import gen_test_symbol
    bers = []
    for snr_db in snr_range:
        err = tot = 0
        for _ in range(n_trials):
            bits, _, _, y_time, H = gen_test_symbol(
                channel_type="multipath", snr_db=snr_db,
                sir_db=ABLATION_SIR, p=0.02)
            X = np.fft.fft(y_time) / H
            rx = ofdm.demodulate(X.reshape(1, -1))[0]
            err += np.sum(rx != bits.flatten())
            tot += bits.size
        bers.append(err / tot)
    return bers


def _plot_fig5(snr_range, depth_bers, trad, means, sir_label):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for d, c in zip(MAB_DEPTHS, colors):
        ax.semilogy(snr_range, np.maximum(depth_bers[d], 1e-6), "s-",
                    color=c, linewidth=1.8, markersize=5,
                    label=f"{d} MAB (mean {means[d]:.3f})")
    ax.semilogy(snr_range, np.maximum(trad, 1e-6), "o-", color="tab:red",
                linewidth=1.8, markersize=5, label="Traditional (no mitigation)")
    ax.set_ylim(1e-2, 5e-1)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(f"Effect of MAB Depth on BER (SIR = {sir_label})")
    ax.legend(fontsize=9)

    ax2 = fig.add_axes([0.70, 0.55, 0.25, 0.28])
    ax2.bar([f"{d} MAB" for d in MAB_DEPTHS],
            [means[d] for d in MAB_DEPTHS], color=colors)
    ax2.set_title("Mean BER", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.savefig("results/fig5_mab_depth_ablation.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()