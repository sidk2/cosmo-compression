from typing import List
import os

import torch
import torch.utils
import torch.nn as nn
from torch.utils import data as torchdata

import lightning

import Pk_library as PKL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import random

from torchvision.datasets import CelebA
from lightning.pytorch import seed_everything

import torch.nn as nn

import wandb
import os

from cosmo_compression.data import data
from cosmo_compression.model import represent
from cosmo_compression.parameter_estimation import inference

seed_everything(137)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

transform = transforms.Compose([
    transforms.Pad(
        padding=(39, 19),  # Pad 39 pixels on the left/right and 19 pixels on the top/bottom
        fill=0  # Fill with 0s (black padding)
    ),
    transforms.CenterCrop((256, 256)),  # Crop if necessary to get an exact 256x256 resolution
    transforms.ToTensor(),  # Convert images to tensors
    # transforms.Normalize((0), (1))  # Normalize to [-1, 1]
])

train_data = CelebA(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

subset_size = 50000
subset_indices = random.sample(range(len(train_data)), subset_size)
subset_dataset = Subset(train_data, subset_indices)

# Use the subset dataset for training
train_data = subset_dataset

img, label = train_data[0]

mean = torch.mean(img)
std = torch.std(img)
img = (img.unsqueeze(0).cuda() - mean)/std

fm: lightning.LightningModule = represent.Represent.load_from_checkpoint(
    "time_to_code_faces/step=step=26300-val_loss=0.018.ckpt"
).to('cuda')

n_sampling_steps = 80
t = torch.linspace(0, 1, n_sampling_steps).cuda()

hs = [fm.encoder(img, ts) for ts in t]  # List of tensors
h = torch.cat(hs, dim=1)

x0 = torch.randn_like(img, device=img.device)

pred = fm.decoder.predict(x0.cuda(), h=h, t=t, n_sampling_steps=n_sampling_steps)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow((img[0, :, : , :]*std+mean).detach().cpu().permute(1, 2, 0).numpy())
ax[1].imshow((pred[0, :, : , :]*std+mean).detach().cpu().permute(1, 2, 0).numpy())
ax[0].set_title("x")
ax[1].set_title("Reconstructed x")
plt.savefig("cosmo_compression/results/face_reconst.png")