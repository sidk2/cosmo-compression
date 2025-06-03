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
        suite="Astrid",
        dataset="LH",
        map_type="Mcdm"
    )

fm_model_path = "latent_ablation_workshop_outc/no_hierarchical_8/step=step=17100-val_loss=0.354.ckpt"

y, cosmo = val_data[np.random.randint(0, len(val_data))]
y = torch.tensor(y).cuda().unsqueeze(0)

batch = y, cosmo

fm: lightning.LightningModule = represent.Represent.load_from_checkpoint(fm_model_path).to('cuda')

fm.eval()
x0 = torch.randn_like(y)


y = y * val_data.std + val_data.mean

plt.imshow(y[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy())
plt.savefig("cosmo_compression/results/field.png")

h = fm.encoder((y-val_data.mean)/val_data.std)

full_trajectory = fm.decoder.predict(
    x0,
    h=h,
    n_sampling_steps=10,
    full_return = True
)

plt.close()

plt.figure()
plt.imshow(full_trajectory[5, :, :, :].squeeze().detach().cpu().numpy())
plt.savefig("cosmo_compression/results/t_minus_one.png")

plt.close()

plt.figure()
plt.imshow(full_trajectory[6, :, :, :].squeeze().detach().cpu().numpy())
plt.savefig("cosmo_compression/results/t.png")

plt.close()

plt.figure()
plt.imshow(((full_trajectory[6, :, :, :] - full_trajectory[5, :, :, :]) / (1/30)).squeeze().detach().cpu().numpy())
plt.savefig("cosmo_compression/results/v.png")