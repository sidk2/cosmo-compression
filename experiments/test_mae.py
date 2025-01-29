from typing import List
import os
import time

import torch
import torch.utils
from torch.utils import data as torchdata

import Pk_library as PKL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import lightning
from torch.utils.data import DataLoader

from cosmo_compression.data import data
from cosmo_compression.model import represent

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device: str = "cuda" if torch.cuda.is_available() else "cpu"

dataset: torchdata.Dataset = data.CAMELS(
    idx_list=range(14_600, 15_000),
    map_type="Mcdm",
    dataset="LH",
    parameters=["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "Omega_b"],
)

data_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

batch_size = 32
mean = data.NORM_DICT["Mcdm"][256]["mean"]
std = data.NORM_DICT["Mcdm"][256]["std"]

# Load models
fm: represent.Represent = represent.Represent.load_from_checkpoint(
    "cosmo_16x16_attend_32ch_16window/step=step=8900-val_loss=0.371.ckpt"
).to(device)
fm.eval()

tot_diff = 0
ps_diff = 0
# Iterate through the DataLoader
for idx, (params, imgs) in enumerate(data_loader):
    print(idx)
    Pk_orig = np.zeros((batch_size, 181))
    for i, img in enumerate(imgs):
        y = img.clone().cpu().numpy().squeeze() * std + mean
        delta_fields_orig_1 = y / np.mean(y) - 1
        Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, 'None', 1, verbose=False)
        k_orig = Pk2D.k
        Pk_orig[i, :] = Pk2D.Pk

    imgs = imgs.cuda()  # Move batch to GPU


    # Encode the batch of images
    latents = fm.encoder(imgs)

    # Generate predictions in batch
    preds = fm.decoder.predict(x0=torch.randn_like(imgs), h=latents)

    Pk_pred = np.zeros((batch_size, 181))
    for i, img in enumerate(preds):
        y_pred = img.clone().cpu().numpy().squeeze() * std + mean
        delta_fields_pred_1 = y_pred / np.mean(y_pred) - 1
        Pk2D_pred = PKL.Pk_plane(delta_fields_pred_1, 25.0, 'None', 1, verbose=False)
        k_pred = Pk2D_pred.k
        Pk_pred[i, :] = Pk2D_pred.Pk

    # Calculate differences for the batch
    batch_diff = (imgs.cpu().numpy()[:, 0, :, :] - preds.cpu().numpy()[:, 0, :, :]) * std

    # Accumulate the differences
    tot_diff += np.sum(np.abs(batch_diff))
    ps_diff += np.sum(np.abs(Pk_pred - Pk_orig))

print(tot_diff / 256/256/400)
print(ps_diff / 400 / 181)

cosmo, img = dataset[0]
img = torch.tensor(img).unsqueeze(0).cuda()

latents = fm.encoder(img)

pred = fm.decoder.predict(x0=torch.randn_like(img), h=latents)
# num_repeats = 5

# # Extract the last entry along the desired dimension (assuming the first dimension here)
# last_entry = pred[-1:]

# # Repeat the last entry and concatenate
# pred = torch.cat([pred, last_entry.repeat(num_repeats, *([1] * (pred.dim() - 1)))], dim=0)

# # Define the animation function
# def update_plot(i):
#     y_pred = pred[i].cpu().numpy()[0, 0, :, :] * std + mean
#     delta_fields_pred_1 = y_pred / np.mean(y_pred) - 1
#     Pk2D_pred = PKL.Pk_plane(delta_fields_pred_1, 25.0, "None", 1, verbose=False)
#     k_pred = Pk2D_pred.k
#     Pk_pred = Pk2D_pred.Pk

#     difference = y - y_pred

#     # Update image and plots
#     img1.set_data(y)
#     img2.set_data(y_pred)
#     img3.set_data(difference)
#     img3.set_clim(-4, 4)
    
#     line_orig.set_data(k_orig, Pk_orig)
#     line_pred.set_data(k_pred, Pk_pred)

#     return img1, img2, img3, line_orig, line_pred

# Pre-compute the original power spectrum
y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, "None", 1, verbose=False)
k_orig = Pk2D.k
Pk_orig = Pk2D.Pk

pred = pred.detach().cpu().numpy()[0,0,:,:]*std+mean
delta_fields_pred_1 = pred / np.mean(pred) - 1
Pk2D = PKL.Pk_plane(delta_fields_pred_1, 25.0, "None", 1, verbose=False)
k_pred = Pk2D.k
Pk_pred = Pk2D.Pk

# Initialize the figure and axes
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Original Image
ax1 = axs[0, 0]
img1 = ax1.imshow(y, cmap="viridis", origin="lower")
ax1.set_title("Original Image")
ax1.axis("off")
plt.colorbar(img1, ax=ax1, label="Density")

# Predicted Image
ax2 = axs[0, 1]
img2 = ax2.imshow(y, cmap="viridis", origin="lower")
ax2.set_title("Reconstructed Image")
ax2.axis("off")
plt.colorbar(img2, ax=ax2, label="Density")

# Power Spectra
ax3 = axs[1, 0]
line_orig, = ax3.plot(k_orig, Pk_orig, label="Original")
line_pred, = ax3.plot(k_pred, Pk_pred, label="Reconstructed")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_title("Power Spectra")
ax3.set_xlabel("Wavenumber $k\,[h/Mpc]$")
ax3.set_ylabel("$P(k)\,[(Mpc/h)^2]$")
ax3.legend()

# Difference Image
ax4 = axs[1, 1]
img3 = ax4.imshow(y-pred, cmap="seismic", origin="lower", vmin=-1, vmax=1)
ax4.set_title("Error Map")
ax4.axis("off")
plt.colorbar(img3, ax=ax4, label="Difference")

# Adjust layout
plt.tight_layout()

# Create the animation
# anim = FuncAnimation(fig, update_plot, frames=len(pred), blit=True)

# Save or display the animation
plt.savefig("cosmo_compression/results/mae.png")
# Alternatively, display in a Jupyter Notebook
# from IPython.display import HTML
# HTML(anim.to_jshtml())
