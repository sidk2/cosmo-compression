import os
import torch
import torch.utils
import torch.nn as nn
from torch.utils import data as torchdata
import lightning
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything
import torch.nn as nn
import Pk_library as PKL
import numpy as np
import wandb
import os
from cosmo_compression.data import data
from cosmo_compression.model import represent
from matplotlib.colors import Normalize

seed_everything(137)

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

MAP_TYPE = "Mcdm"
MAP_RESOLUTION = 256

mean = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["mean"]
std = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["std"]

device: str = "cuda" if torch.cuda.is_available() else "cpu"

train_data = data.CAMELS(
        idx_list=range(14_000),
        map_type='Mcdm',
        parameters=['Omega_m', 'sigma_8',],
        )

val_data = data.CAMELS(
    idx_list=range(14_000, 15_000),
    map_type='Mcdm',
    parameters=['Omega_m', 'sigma_8',],
)

y, cosmo = val_data[0]
y = torch.tensor(y).cuda().unsqueeze(0)

batch = y, cosmo

fm: lightning.LightningModule = represent.Represent.load_from_checkpoint(
    "camels_gdn_time_for_encoding/step=step=22000-val_loss=0.361.ckpt"
).to('cuda')

fm.eval()

n_samples = [2, 5, 10, 20, 50]
x0 = torch.randn_like(y)

fig, ax = plt.subplots(3, 6, figsize=(24, 12))  # Modify the layout for 3 rows
ax[0, 0].imshow(y[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy())
ax[0, 0].set_title("Original")

y = y * std + mean

delta_fields_orig_1 = y[0, 0, :, :] / np.mean(y[0, 0, :, :].detach().cpu().numpy()) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1.detach().cpu().numpy(), 25.0, "None", 1, verbose=False)
k_fin = Pk2D.k
Pk_fin = Pk2D.Pk

ax[1, 0].plot(k_fin, Pk_fin)
ax[1, 0].set_xscale('log')
ax[1, 0].set_yscale('log')

# Define the color normalization for the difference images
vmin = float('inf')
vmax = float('-inf')

# First pass through the data to determine the color scale range for the difference
for i, n_sampling_steps in enumerate(n_samples):
    t = torch.linspace(0, 1, n_sampling_steps).cuda()

    hs = [fm.encoder((y-mean)/std, ts) for ts in t]  # List of tensors
    h = torch.cat(hs, dim=1)

    pred = fm.decoder.predict(
        x0,
        h=h,
        t=t,
        n_sampling_steps=n_sampling_steps,
    )
    pred = pred * std + mean
    diff = torch.abs(y - pred).cpu().numpy()[0, 0, :, :]  # Absolute pixel-wise difference

    # Find global min and max for scaling
    vmin = min(vmin, diff.min())
    vmax = max(vmax, diff.max())

for i, n_sampling_steps in enumerate(n_samples):
    t = torch.linspace(0, 1, n_sampling_steps).cuda()

    hs = [fm.encoder((y-mean)/std, ts) for ts in t]  # List of tensors
    h = torch.cat(hs, dim=1)

    pred = fm.decoder.predict(
        x0,
        h=h,
        t=t,
        n_sampling_steps=n_sampling_steps,
    )
    pred = pred * std + mean
    delta_fields_orig_1 = pred[0, 0, :, :] / np.mean(pred[0, 0, :, :].detach().cpu().numpy()) - 1
    Pk2D = PKL.Pk_plane(delta_fields_orig_1.detach().cpu().numpy(), 25.0, "None", 1, verbose=False)
    k_pred = Pk2D.k
    Pk_pred = Pk2D.Pk
    
    ax[1, i+1].plot(k_fin, Pk_fin, label='original')
    ax[1, i+1].plot(k_pred, Pk_pred, label='reconstructed')
    ax[1, i+1].set_xscale('log')
    ax[1, i+1].set_yscale('log')
    ax[1, i+1].legend()

    ax[0, i+1].imshow(pred[0, 0, :, :].detach().cpu().numpy())
    ax[0, i+1].set_title(f"{n_sampling_steps} steps")

    # Plot the pixel-to-pixel difference in the third row with a consistent color scale
    diff = (y - pred).cpu().numpy()[0, 0, :, :]  # Absolute pixel-wise difference
    im = ax[2, i+1].imshow(diff, cmap='seismic', vmin=vmin, vmax=vmax)
    ax[2, i+1].set_title(f"Diff {n_sampling_steps} steps")

# Add a colorbar for the difference images
fig.colorbar(im, ax=ax[2, :], orientation='horizontal', fraction=0.02, pad=0.1)

plt.savefig("cosmo_compression/results/test_reconst.png")
