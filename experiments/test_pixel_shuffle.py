from typing import List
import os

import torch
import torch.utils
import torch.nn as nn
from torch.utils import data as torchdata

import Pk_library as PKL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import lightning
import tqdm

from cosmo_compression.data import data
from cosmo_compression.model import represent

torch.manual_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

MAP_TYPE = "Mcdm"
MAP_RESOLUTION = 256

mean = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["mean"]
std = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["std"]

device: str = "cuda" if torch.cuda.is_available() else "cpu"

dataset: torchdata.Dataset = data.CAMELS(
    map_type=MAP_TYPE,
    dataset="1P",
    parameters=["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "Omega_b"],
)

y, cosmo = dataset[0]

y = torch.tensor(y).unsqueeze(0)

P = 128
N = 2
patches = y.unfold(2, P, P).unfold(3, P, P)

# Step 2: Reshape to stack patches channel-wise
# patches shape: (1, 1, num_patches_h, num_patches_w, P, P)
# We want to stack them along the channel dimension
num_patches_h = N // P  # Number of patches along height
num_patches_w = N // P  # Number of patches along width
y_hat = patches.contiguous().view(1, -1, P, P)

fig, ax = plt.subplots(1, 5, figsize=(20, 4))

ax[0].imshow(y_hat[0, 0, : , :].detach().cpu().numpy())
ax[1].imshow(y_hat[0, 1, : , :].detach().cpu().numpy())
ax[2].imshow(y_hat[0, 2, : , :].detach().cpu().numpy())
ax[3].imshow(y_hat[0, 3, : , :].detach().cpu().numpy())
ax[4].imshow(y[0, 0, : , :].detach().cpu().numpy())

plt.savefig("cosmo_compression/results/pixel_shuffle.png")