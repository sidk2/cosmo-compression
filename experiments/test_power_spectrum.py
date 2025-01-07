from typing import List
import os

import torch
import torch.utils
from torch.utils import data as torchdata

import Pk_library as PKL
import numpy as np
import matplotlib.pyplot as plt

import lightning

from cosmo_compression.data import data
from cosmo_compression.model import represent

os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset: torchdata.Dataset = data.CAMELS(
        map_type='Mcdm',
        dataset='LH',
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2','Omega_b'],
    )

loader = torchdata.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

mean = data.NORM_DICT['Mcdm'][256]["mean"]
std = data.NORM_DICT['Mcdm'][256]["std"]

# Load models
fm: represent.Represent = represent.Represent.load_from_checkpoint("full-img-lat/step=step=25000-val_loss=0.099.ckpt").to(device)
fm.eval()

cosmo, img = dataset[np.random.randint(0, 525)]
img = torch.tensor(img).unsqueeze(0).cuda()

latent = fm.encoder(img)
h, orig = latent
h = h.unsqueeze(0)
latent = h, orig
pred = fm.decoder.predict(x0=torch.randn_like(img), h=latent)

# Original Power Spectrum
y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, 'None', 1, verbose=False)
k_orig = Pk2D.k
Pk_orig = Pk2D.Pk

# Predicted Power Spectrum
y_pred = pred.cpu().numpy()[0, 0, :, :] * std + mean
delta_fields_pred_1 = y_pred / np.mean(y_pred) - 1
Pk2D_pred = PKL.Pk_plane(delta_fields_pred_1, 25.0, 'None', 1, verbose=False)
k_pred = Pk2D_pred.k
Pk_pred = Pk2D_pred.Pk

# Create a combined figure
plt.figure(figsize=(12, 8))

# Power Spectra Plot
plt.subplot(2, 2, (1, 2))  # Span two columns in the first row
plt.plot(k_orig, Pk_orig, label='Original')
plt.plot(k_pred, Pk_pred, label='Predicted')
plt.xscale('log')
plt.yscale('log')
plt.title("Power Spectra")
plt.xlabel("Wavenumber $k\,[h/Mpc]$")
plt.ylabel("$P(k)\,[(Mpc/h)^2]$")
plt.legend()

# Original Image
plt.subplot(2, 2, 3)
plt.imshow(img.cpu().numpy()[0, 0, :, :] * std + mean, cmap='viridis', origin='lower')
plt.colorbar(label='Density')
plt.title("Original Image")
plt.axis('off')

# Predicted Image
plt.subplot(2, 2, 4)
plt.imshow(pred.cpu().numpy()[0, 0, :, :] * std + mean, cmap='viridis', origin='lower')
plt.colorbar(label='Density')
plt.title("Predicted Image")
plt.axis('off')

# Adjust layout and save
plt.tight_layout()
plt.savefig('cosmo_compression/results/combined_figure.png')
