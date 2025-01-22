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

device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset: torchdata.Dataset = data.CAMELS(
        idx_list=range(14_600, 15_000),
        map_type='Mcdm',
        dataset='LH',
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2','Omega_b'],
    )

batch_size = 32
mean = data.NORM_DICT['Mcdm'][256]["mean"]
std = data.NORM_DICT['Mcdm'][256]["std"]

fm: represent.Represent = represent.Represent.load_from_checkpoint("img-lat-2ch/step=step=31000-val_loss=0.380.ckpt").to(device)

params, imgs = dataset[0]
imgs = torch.tensor(imgs).unsqueeze(0).cuda()  # Move batch to GPU

# Encode the batch of images
latents = fm.encoder(imgs)
h, orig = latents
print(torch.min(orig), torch.max(orig))

# Ensure `orig` is moved to the CPU for plotting
orig = orig.squeeze(0).cpu().detach().numpy()  # Squeeze batch dimension

# Plot the original image and its two channels
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image (denormalize for visualization)
original_image = imgs.squeeze(0).cpu().numpy() * std + mean
axes[0].imshow(original_image.squeeze(), cmap="viridis")
axes[0].set_title("Original Image")
axes[0].axis("off")

# First channel of `orig`
axes[1].imshow(orig[0], cmap="viridis")
axes[1].set_title("First Channel of Latent")
axes[1].axis("off")

# Second channel of `orig`
axes[2].imshow(orig[1], cmap="viridis")
axes[2].set_title("Second Channel of Latent")
axes[2].axis("off")

plt.savefig("cosmo_compression/results/2ch_feat_maps.png")