from typing import List
import os
import time

import torch
import torch.utils
from torch.utils import data as torchdata

import Pk_library as PKL
import numpy as np
import matplotlib.pyplot as plt

import lightning
from torch.utils.data import DataLoader

from cosmo_compression.data import data
from cosmo_compression.model import represent

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    "cosmo_rolling_latent_64ch2wind/step=step=30200-val_loss=0.382.ckpt"
).to(device)
fm.eval()

tot_diff = 0
ps_diff = 0
# Iterate through the DataLoader
# for idx, (params, imgs) in enumerate(data_loader):
#     print(idx)
#     Pk_orig = np.zeros((batch_size, 181))
#     for i, img in enumerate(imgs):
#         y = img.clone().cpu().numpy().squeeze() * std + mean
#         delta_fields_orig_1 = y / np.mean(y) - 1
#         Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, 'None', 1, verbose=False)
#         k_orig = Pk2D.k
#         Pk_orig[i, :] = Pk2D.Pk

#     imgs = imgs.cuda()  # Move batch to GPU


#     # Encode the batch of images
#     latents = fm.encoder(imgs)
#     h, orig = latents
#     # h = h.unsqueeze(0)  # Adjust dimension for batching if necessary

#     latents = (h, orig)

#     # Generate predictions in batch
#     preds = fm.decoder.predict(x0=torch.randn_like(imgs), h=latents)

#     Pk_pred = np.zeros((batch_size, 181))
#     for i, img in enumerate(preds):
#         y_pred = img.clone().cpu().numpy().squeeze() * std + mean
#         delta_fields_pred_1 = y_pred / np.mean(y_pred) - 1
#         Pk2D_pred = PKL.Pk_plane(delta_fields_pred_1, 25.0, 'None', 1, verbose=False)
#         k_pred = Pk2D_pred.k
#         Pk_pred[i, :] = Pk2D_pred.Pk

#     # Calculate differences for the batch
#     batch_diff = (imgs.cpu().numpy()[:, 0, :, :] - preds.cpu().numpy()[:, 0, :, :]) * std

#     # Accumulate the differences
#     tot_diff += np.sum(np.abs(batch_diff))
#     ps_diff += np.sum(np.abs(Pk_pred - Pk_orig))

# print(tot_diff / 256/256/400)
# print(ps_diff / 400 / 181)

cosmo, img = dataset[0]
img = torch.tensor(img).unsqueeze(0).cuda()

latents = fm.encoder(img)
h, orig = latents
h = h.unsqueeze(0)
latents = h, orig

pred = fm.decoder.predict(x0=torch.randn_like(img), h=latents)

# Original Power Spectrum
y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, "None", 1, verbose=False)
k_orig = Pk2D.k
Pk_orig = Pk2D.Pk

# Predicted Power Spectrum
y_pred = pred.cpu().numpy()[0, 0, :, :] * std + mean
delta_fields_pred_1 = y_pred / np.mean(y_pred) - 1
Pk2D_pred = PKL.Pk_plane(delta_fields_pred_1, 25.0, "None", 1, verbose=False)
k_pred = Pk2D_pred.k
Pk_pred = Pk2D_pred.Pk

# Create a combined figure
plt.figure(figsize=(12, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(y, cmap="viridis", origin="lower")
plt.colorbar(label="Density")
plt.title("Original Image")
plt.axis("off")

# Predicted (Reconstructed) Image
plt.subplot(2, 2, 2)
plt.imshow(y_pred, cmap="viridis", origin="lower")
plt.colorbar(label="Density")
plt.title("Reconstructed Image")
plt.axis("off")

# Power Spectra
plt.subplot(2, 2, 3)
plt.plot(k_orig, Pk_orig, label="Original")
plt.plot(k_pred, Pk_pred, label="Reconstructed")
plt.xscale("log")
plt.yscale("log")
plt.title("Power Spectra")
plt.xlabel("Wavenumber $k\,[h/Mpc]$")
plt.ylabel("$P(k)\,[(Mpc/h)^2]$")
plt.legend()

difference = y - y_pred
# Error Map (Difference Image)
print(np.mean(np.abs(difference)))
plt.subplot(2, 2, 4)
plt.imshow(
    difference,
    cmap="seismic",
    origin="lower",
    vmin=-np.max(abs(difference)),
    vmax=np.max(abs(difference)),
)
plt.colorbar(label="Difference")
plt.title("Error Map")
plt.axis("off")

# Adjust layout and save
plt.tight_layout()
plt.savefig("cosmo_compression/results/2x2_figure.png")
