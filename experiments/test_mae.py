from typing import List
import os
import time
import tqdm

import torch
import torch.utils
from torch.utils import data as torchdata

import Pk_library as PKL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torchvision.transforms as T


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
fm = represent.Represent.load_from_checkpoint("reversion_4/step=step=44700-val_loss=0.323.ckpt").to(device)
fm.eval()

print("Loaded model.")

tot_diff = 0
ps_diff = 0

# Iterate through the DataLoader
for idx, (imgs, params) in tqdm.tqdm(enumerate(data_loader)):
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
    preds = fm.decoder.predict(x0=torch.randn_like(imgs), h=latents, n_sampling_steps=30)

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
    tot_diff += np.mean(np.abs(batch_diff))
    ps_diff += np.mean(np.abs(np.log10(Pk_pred[np.where(Pk_orig != 0)]) - np.log10(Pk_orig[np.where(Pk_orig != 0)])))

print(tot_diff /len(data_loader))
print(ps_diff / len(data_loader))