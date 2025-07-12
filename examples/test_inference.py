import os
import torch
import torch.utils
import torch.nn as nn
from torch.utils import data as torchdata
import lightning
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything
import torch.nn as nn
import Pk_library as PKL
import numpy as np
import wandb
import os
from cosmo_compression.data import data
from cosmo_compression.model import represent
from matplotlib.colors import Normalize

# seed_everything(137)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

MAP_TYPE = "Mcdm"
MAP_RESOLUTION = 256

device: str = "cuda" if torch.cuda.is_available() else "cpu"

val_data = data.CAMELS(
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'WDM'],
        suite="IllustrisTNG",
        dataset="WDM",
        map_type="Mcdm"
    )

y, cosmo = val_data[np.random.randint(0, len(val_data))]
y = torch.tensor(y).cuda().unsqueeze(0)

batch = y, cosmo

print(cosmo[-1])
fm: lightning.LightningModule = represent.Represent.load_from_checkpoint(
    "16x16_diti_setup_non_hier/16x16_diti_no_latent_splitting_smol_res/step=step=10200-val_loss=0.332.ckpt"
).to('cuda')

fm.eval()
x0 = torch.randn_like(y)

print(val_data.mean, val_data.std)

y = y * val_data.std + val_data.mean

plt.imshow(y[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy())
plt.savefig("cosmo_compression/results/field.png")

h = fm.encoder((y-val_data.mean)/val_data.std)

num_channels = h.shape[1]

print(h.shape)

# Display the latent channels in a 4 x 4 grid
fig, axs = plt.subplots(4, 4, figsize=(10, 10))
for i in range(4):
    for j in range(4):
        axs[i, j].imshow(h[0, i * 4 + j, :, :].detach().cpu().numpy())
        axs[i, j].set_title(f"Channel {i * 4 + j}")
        axs[i, j].axis("off")

plt.savefig("cosmo_compression/results/latents.png")
