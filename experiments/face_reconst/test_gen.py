from pathlib import Path
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning import Trainer
from lightning.pytorch import seed_everything

import matplotlib.pyplot as plt
import torchmetrics

import wandb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from cosmo_compression.model.represent import Represent

import torch

transform = transforms.Compose([
        transforms.Pad(
            padding=(39, 19),  # Pad 39 pixels on the left/right and 19 pixels on the top/bottom
            fill=0  # Fill with 0s (black padding)
        ),
        transforms.CenterCrop((256, 256)),  # Crop if necessary to get an exact 256x256 resolution
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

train_data = CelebA(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

val_data = CelebA(
    root="./data",
    split="valid",
    download=True,
    transform=transform
)

fm = Represent.load_from_checkpoint('face-data-32h/step=step=3000-val_loss=0.021.ckpt').cuda()

y_fin, pred_tot, img_lat_fin = None, None, None
min_mse = 90900909090
y, _ = val_data[3986]
y = y.unsqueeze(0).cuda()
h = fm.encoder(y)
lat, img_lat = h
lat = lat.unsqueeze(0)
h = lat, img_lat
for idx in range(0,1):
    print(idx)
    # y, _ = train_data[torch.randint(low=0, high=19867, size=[1]).item()]
    
    # if h is not None:
    #     h = self.h_embedding(h)
    x0 = torch.randn_like(y, device = y.device)
    pred = fm.decoder.predict(
        x0,
        h=h,
        n_sampling_steps=50,
    )
    if pred_tot is None:
        pred_tot = pred
    else:
        pred_tot += pred

pred_tot = pred_tot*0.5+0.5
# pred_tot += torch.abs(pred_tot.min())
# pred_tot /= pred_tot.max()
pred_tot = pred_tot.squeeze()

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow((y[0, :, : , :]*0.5+0.5).detach().cpu().permute(1, 2, 0).numpy())
ax[1].imshow(pred_tot.detach().cpu().permute(1, 2, 0).numpy())
ax[0].set_title("x")
ax[1].set_title("Reconstructed x")
plt.savefig("cosmo_compression/results/face.png")

plt.close()

img_lat = img_lat.squeeze(0)  # Remove the batch dimension to make it 32x16x16
num_channels = img_lat.shape[0]

fig, axes = plt.subplots(4, 8, figsize=(24, 12))

for i, ax in enumerate(axes.flat):
    if i < num_channels:
        channel = img_lat[i].detach().cpu().numpy()
        ax.imshow(channel, cmap='viridis')
        ax.set_title(f'Channel {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.savefig("cosmo_compression/results/face_latents.png")
plt.close()