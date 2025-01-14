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

device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset: torchdata.Dataset = data.CAMELS(
        idx_list=range(14_600, 15_000),
        map_type='Mcdm',
        dataset='LH',
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2','Omega_b'],
    )

data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

mean = data.NORM_DICT['Mcdm'][256]["mean"]
std = data.NORM_DICT['Mcdm'][256]["std"]

# Load models
fm: represent.Represent = represent.Represent.load_from_checkpoint("img-lat-64ch/step=step=44500-val_loss=0.282.ckpt").to(device)
fm.eval()

tot_diff = 0

# Iterate through the DataLoader
for idx, (params, imgs) in enumerate(data_loader):
    print(idx)
    imgs = imgs.cuda()  # Move batch to GPU

    # Encode the batch of images
    enc_s = time.time()
    latents = fm.encoder(imgs)
    print(f"Encoding one batch took {time.time() - enc_s}.")
    h, orig = latents
    # h = h.unsqueeze(0)  # Adjust dimension for batching if necessary

    latents = (h, orig)

    # Generate predictions in batch
    dec_s = time.time()
    preds = fm.decoder.predict(x0=torch.randn_like(imgs), h=latents)
    print(f"Decoding one batch took {time.time() - dec_s}.")

    # Calculate differences for the batch
    batch_diff = (imgs.cpu().numpy()[:, 0, :, :] - preds.cpu().numpy()[:, 0, :, :]) * std

    # Accumulate the differences
    tot_diff += np.sum(np.abs(batch_diff))

print(tot_diff)

img = imgs[0, 0, :, :]
# Original Power Spectrum
y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, 'None', 1, verbose=False)
k_orig = Pk2D.k
Pk_orig = Pk2D.Pk

# Predicted Power Spectrum
y_pred = preds.cpu().numpy()[0, 0, :, :] * std + mean
delta_fields_pred_1 = y_pred / np.mean(y_pred) - 1
Pk2D_pred = PKL.Pk_plane(delta_fields_pred_1, 25.0, 'None', 1, verbose=False)
k_pred = Pk2D_pred.k
Pk_pred = Pk2D_pred.Pk

# Create a combined figure
plt.figure(figsize=(12, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(y, cmap='viridis', origin='lower')
plt.colorbar(label='Density')
plt.title("Original Image")
plt.axis('off')

# Predicted (Reconstructed) Image
plt.subplot(2, 2, 2)
plt.imshow(y_pred, cmap='viridis', origin='lower')
plt.colorbar(label='Density')
plt.title("Reconstructed Image")
plt.axis('off')

# Power Spectra
plt.subplot(2, 2, 3)
plt.plot(k_orig, Pk_orig, label='Original')
plt.plot(k_pred, Pk_pred, label='Reconstructed')
plt.xscale('log')
plt.yscale('log')
plt.title("Power Spectra")
plt.xlabel("Wavenumber $k\,[h/Mpc]$")
plt.ylabel("$P(k)\,[(Mpc/h)^2]$")
plt.legend()

difference = (y-y_pred)
# Error Map (Difference Image)
print(np.mean(np.abs(difference)))
plt.subplot(2, 2, 4)
plt.imshow(difference, cmap='seismic', origin='lower', vmin=-np.max(abs(difference)), vmax=np.max(abs(difference)))
plt.colorbar(label='Difference')
plt.title("Error Map")
plt.axis('off')

# Adjust layout and save
plt.tight_layout()
plt.savefig('cosmo_compression/results/2x2_figure.png')