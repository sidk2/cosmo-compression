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

# seed_everything(137)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

MAP_TYPE = "Mcdm"
MAP_RESOLUTION = 256

device: str = "cuda" if torch.cuda.is_available() else "cpu"

val_data = data.CAMELS(
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'WDM'],
        suite="IllustrisTNG",
        dataset="WDM",
        map_type="Mcdm"
    )

y, cosmo = val_data[np.random.randint(0, len(val_data))]
y = torch.tensor(y).cuda().unsqueeze(0)

batch = y, cosmo

print(cosmo[-1])
fm: lightning.LightningModule = represent.Represent.load_from_checkpoint(
    "reversion_1/step=step=59100-val_loss=0.339.ckpt"
).to('cuda')

fm.eval()

n_samples = [30]
x0 = torch.randn_like(y)

fig, ax = plt.subplots(3, len(n_samples) + 1, figsize=(6*len(n_samples)+6, 12))  # Modify the layout for 3 rows
ax[0, 0].imshow(y[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy())
ax[0, 0].set_title("Original")

print(val_data.mean, val_data.std)

y = y * val_data.std + val_data.mean

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
    # List of tensors
    h = fm.encoder((y-val_data.mean)/val_data.std)
    spatial, repr = h
    repr = repr.unsqueeze(0)
    h = spatial, repr

    pred = fm.decoder.predict(
        x0,
        h=h,
        n_sampling_steps=n_sampling_steps,
    )
    pred = pred * val_data.std + val_data.mean
    diff = torch.abs(y - pred).cpu().numpy()[0, 0, :, :]  # Absolute pixel-wise difference

    # Find global min and max for scaling
    vmin = min(vmin, diff.min())
    vmax = max(vmax, diff.max())

for i, n_sampling_steps in enumerate(n_samples):
    t = torch.linspace(0, 1, n_sampling_steps).cuda()

    h = fm.encoder((y-val_data.mean)/val_data.std)
    spatial, repr = h
    repr = repr.unsqueeze(0)
    h = spatial, repr


    pred = fm.decoder.predict(
        x0,
        h=h,
        n_sampling_steps=n_sampling_steps,
    )
    pred = pred * val_data.std + val_data.mean
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
