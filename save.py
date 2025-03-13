import torch
import torch.nn as nn

import wandb
import os

import matplotlib.pyplot as plt
import numpy as np  

from cosmo_compression.model import represent
from cosmo_compression.data import data

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


# Create directories if they don't exist
train_dir = '../../../monolith/global_data/astro_compression/CAMELS/images/train'
test_dir = '../../../monolith/global_data/astro_compression/CAMELS/images/test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Iterate through train data and save images
np.save(os.path.join(train_dir, 'train_data.npy'), train_data.y)
np.save(os.path.join(test_dir, 'test_data.npy'), val_data.y)