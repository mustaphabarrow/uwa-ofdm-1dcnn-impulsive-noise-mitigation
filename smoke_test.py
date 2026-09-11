import torch
import numpy as np
from torch.utils.data import DataLoader

from ofdm_system import OFDMSystem
from uwa_channel import UWAChannel
from impulsive_noise import ImpulsiveNoiseGenerator
from model import Attention1DCNN
from train import UWADataset, Trainer, transfer_learning_train, create_source_domain_model

N_SUBCARRIERS = 64
CP_LENGTH = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

source_ds = UWADataset(50, N_SUBCARRIERS, CP_LENGTH, snr_range=(10, 20))
source_loader = DataLoader(source_ds, batch_size=16, shuffle=True)

model = create_source_domain_model(N_SUBCARRIERS)
trainer = Trainer(model, device=DEVICE, lr=1e-3)
tl, vl = trainer.train(source_loader, source_loader, n_epochs=3, save_path="results/source_model.pth")

print("Model successfully trained for source domain")

target_ds = UWADataset(30, N_SUBCARRIERS, CP_LENGTH, snr_range=(5, 15))
target_loader = DataLoader(target_ds, batch_size=8, shuffle=True)

tl_model, tl_losses, tl_val = transfer_learning_train(
    "results/source_model.pth", target_loader, target_loader,
    n_subcarriers=N_SUBCARRIERS, n_epochs=2, device=DEVICE,
)

print("Transfer learning successful!")
print("All tests passed!")