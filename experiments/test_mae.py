from typing import List
import os
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from torch.utils.data import DataLoader
import numpy as np

from cosmo_compression.data import data
from cosmo_compression.model import represent

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load normalization constants (still required if the model expects normalized inputs)
mean = data.NORM_DICT["Mcdm"][256]["mean"]
std = data.NORM_DICT["Mcdm"][256]["std"]

# Prepare CDM dataset and loader (shared by all models)
cdm_dataset = data.CAMELS(
    idx_list=range(14000, 15000),
    map_type="Mcdm",
    suite="IllustrisTNG",
    dataset="LH",
    parameters=["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "Omega_b"],
)
batch_size = 128
cdm_loader = DataLoader(
    cdm_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

def compute_mean_mse(loader, model):
    """
    Runs a forward pass through `model` on all images in `loader`,
    computes per-image MSE, and returns the mean MSE.
    """
    mse_list = []
    model.eval()
    with torch.no_grad():
        for imgs, _ in tqdm.tqdm(loader):
            imgs = imgs.to(device)
            # Encode, then decode
            latents = model.encoder(imgs)
            preds = model.decoder.predict(x0=torch.randn_like(imgs), h=latents, n_sampling_steps=30, solver="rk4")

            # Move to CPU numpy
            imgs_np = imgs.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # Compute per-image MSE
            diffs = imgs_np[:, 0] - preds_np[:, 0]
            batch_mse = np.mean(diffs**2, axis=(1, 2))
            mse_list.append(batch_mse)

    mse_all = np.concatenate(mse_list, axis=0)
    return float(np.mean(mse_all)), np.std(mse_all)

channels = [4, 8, 12, 16, 20, 64]

# List of checkpoint paths to iterate over
model_paths = [
    "latent_ablation_workshop_outc/no_hierarchical_4/step=step=17100-val_loss=0.354.ckpt",
    "latent_ablation_workshop_outc/no_hierarchical_8/step=step=15500-val_loss=0.327.ckpt",
    "latent_ablation_workshop_outc/no_hierarchical_12/step=step=17700-val_loss=0.324.ckpt",
    "latent_ablation_workshop_outc/no_hierarchical_16/step=step=17700-val_loss=0.312.ckpt",
    "latent_ablation_workshop_outc/no_hierarchical_20/step=step=17700-val_loss=0.304.ckpt",
    "latent_ablation_workshop_outc/no_hierarchical_64/step=step=17700-val_loss=0.257.ckpt",
    # add more paths as needed...
]

for channels, ckpt in zip(channels, model_paths):
    # Load model
    print(f"Loading {ckpt}")
    model = represent.Represent.load_from_checkpoint(ckpt).to(device)

    # Compute mean MSE on CDM dataset
    mean_mse, std_mse = compute_mean_mse(cdm_loader, model)

    # Print only the mean MSE (one line per model)
    print(f"Channels {channels}:  {mean_mse:.6f}, {std_mse:.6f}")
