import torch
import numpy as np
from torch.utils.data import DataLoader

from ofdm_system import OFDMSystem
from uwa_channel import UWAChannel
from impulsive_noise import ImpulsiveNoiseGenerator
from model import Attention1DCNN
from train import (
    UWADataset, Trainer, JointTaskLoss, transfer_learning_train,
    create_source_domain_model, split_dataset,
)

N_SUBCARRIERS = 64
CP_LENGTH = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

ofdm = OFDMSystem(N_SUBCARRIERS, CP_LENGTH)
print(f"Null subcarriers: {ofdm.null_subcarriers}")
print(f"Data subcarriers: {ofdm.data_subcarriers}")
print(f"Null mask shape: {ofdm.null_mask.shape}, sum={ofdm.null_mask.sum()}")

full_ds = UWADataset(80, N_SUBCARRIERS, CP_LENGTH, snr_range=(10, 20))
train_ds, val_ds = split_dataset(full_ds, val_ratio=0.2)
source_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
source_val = DataLoader(val_ds, batch_size=16, shuffle=False)

model = create_source_domain_model(N_SUBCARRIERS)
trainer = Trainer(model, ofdm.null_mask, device=DEVICE, lr=1e-3)
tl, vl = trainer.train(source_loader, source_val, n_epochs=10, save_path="results/source_model.pth")
print(f"Source training: train_loss={tl[-1]:.6f}, val_loss={vl[-1]:.6f}")

target_ds = UWADataset(40, N_SUBCARRIERS, CP_LENGTH, snr_range=(5, 15), flat_channel=False)
train_ds_t, val_ds_t = split_dataset(target_ds, val_ratio=0.2)
target_loader = DataLoader(train_ds_t, batch_size=8, shuffle=True)
target_val = DataLoader(val_ds_t, batch_size=8, shuffle=False)

tl_model, tl_losses, tl_val = transfer_learning_train(
    "results/source_model.pth", target_loader, target_val,
    n_subcarriers=N_SUBCARRIERS, n_epochs=5, device=DEVICE,
)
print(f"TL training: train_loss={tl_losses[-1]:.6f}, val_loss={tl_val[-1]:.6f}")

print("\n--- Verifying forward/backward pass shapes ---")
dummy_input = torch.randn(2, 2, N_SUBCARRIERS).to(DEVICE)
dummy_target = torch.randn(2, 2, N_SUBCARRIERS).to(DEVICE)
dummy_output = model(dummy_input)
assert dummy_output.shape == (2, 2, N_SUBCARRIERS), f"Bad output shape: {dummy_output.shape}"

criterion = JointTaskLoss(ofdm.null_mask, lambda_null=1.0).to(DEVICE)
loss, mse_val, null_val = criterion(dummy_output, dummy_target)
loss.backward()
print(f"Output shape: {dummy_output.shape} OK")
print(f"Loss: {loss.item():.4f} (mse={mse_val:.4f}, null={null_val:.4f})")

print("\n--- Verifying evaluation pipeline (time-domain flow) ---")
tl_model.eval()
bits = ofdm.generate_random_bits(1)
_, tx_symbols = ofdm.transmit(bits)
channel = UWAChannel(N_SUBCARRIERS, CP_LENGTH)
noise_gen = ImpulsiveNoiseGenerator(N_SUBCARRIERS, CP_LENGTH)

H = channel.generate_frequency_response(n_taps=6)
rx_freq = channel.apply_channel_freq(tx_symbols, H)
r_time = np.fft.ifft(rx_freq, axis=-1)[0]
y_time = noise_gen.add_gm_noise(r_time, snr_db=15, sir_db=-15, p=0.02)

input_feat = np.stack([np.real(y_time), np.imag(y_time)]).astype(np.float32)
input_tensor = torch.tensor(input_feat).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = tl_model(input_tensor)
    assert output.shape == (1, 2, N_SUBCARRIERS), f"Bad eval output shape: {output.shape}"
    r_hat = (output[0, 0, :].cpu().numpy() + 1j * output[0, 1, :].cpu().numpy()).reshape(1, N_SUBCARRIERS)

Y_trad = np.fft.fft(y_time) / H
Y_hat = np.fft.fft(r_hat[0]) / H
rx_trad = ofdm.demodulate(Y_trad.reshape(1, -1))
rx_nn = ofdm.demodulate(Y_hat.reshape(1, -1))
print(f"Denoised shape: {r_hat.shape} OK")
print(f"Demodulated bits shape: {rx_nn.shape} OK")
print(f"Transmitted bits shape: {bits.shape} OK")

ber_trad = np.mean(rx_trad[0] != bits[0])
ber_nn = np.mean(rx_nn[0] != bits[0])
print(f"Traditional BER: {ber_trad:.4f}")
print(f"NN BER: {ber_nn:.4f}")

print("\n--- Verifying data consistency ---")
sample_input, sample_target = full_ds[0]
print(f"Dataset input shape: {sample_input.shape}")
print(f"Dataset target shape: {sample_target.shape}")
print(f"Input range: [{sample_input.min():.3f}, {sample_input.max():.3f}]")
print(f"Target range: [{sample_target.min():.3f}, {sample_target.max():.3f}]")
assert sample_input.shape == (2, N_SUBCARRIERS), f"Bad input shape: {sample_input.shape}"
assert sample_target.shape == (2, N_SUBCARRIERS), f"Bad target shape: {sample_target.shape}"
assert not np.allclose(sample_input.numpy(), sample_target.numpy()), \
    "Target must be the CLEAN symbols, NOT the noisy input (the network must denoise)"
print("Paper-consistent: target = Psi(clean symbols), input = Psi(noisy equalized symbols)")

print("\nAll pipeline checks passed!")
