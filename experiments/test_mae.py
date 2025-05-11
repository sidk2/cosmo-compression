from typing import List
import os
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from cosmo_compression.data import data
from cosmo_compression.model import represent
import Pk_library as PKL  # if still needed

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load normalization constants
mean = data.NORM_DICT["Mcdm"][256]["mean"]
std = data.NORM_DICT["Mcdm"][256]["std"]

# Prepare datasets and loaders
cdm_dataset = data.CAMELS(
    idx_list=range(10000, 15000),
    map_type="Mcdm",
    suite="IllustrisTNG",
    dataset="LH",
    parameters=["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "Omega_b"],
)
wdm_dataset = data.CAMELS(
    idx_list=range(10000, 15000),
    map_type="Mcdm",
    dataset="WDM",
    suite="IllustrisTNG",
    parameters=["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "Omega_b"],
)

batch_size = 128
cdm_loader = DataLoader(cdm_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
wdm_loader = DataLoader(wdm_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Load the trained model
checkpoint_path = "diti_64ch_mask/step=step=25700-val_loss=0.255.ckpt"
model = represent.Represent.load_from_checkpoint(checkpoint_path).to(device)
model.eval()

# Function to compute MSE per image for a given loader
def compute_mse(loader):
    mse_list = []
    first = True
    with torch.no_grad():
        for imgs, params in tqdm.tqdm(loader):
            imgs = imgs.to(device)
            # Forward
            latents = model.encoder(imgs)
            preds = model.decoder.predict(x0=torch.randn_like(imgs), h=latents, n_sampling_steps=30)

            # Move to CPU and unnormalize
            imgs_np = imgs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            
            if first:
                first = False
                # Show the reconstruction in a figure.
                fig, ax = plt.subplots(1, 2, figsize=(8, 8))
                ax[0].imshow(imgs_np[0, 0, :, :], cmap='viridis')
                ax[1].imshow(preds_np[0, 0, :, :], cmap='viridis')
                plt.savefig(f"cosmo_compression/results/reconstruction.png")
                plt.close()

            # Compute per-image MSE
            for i in range(imgs_np.shape[0]):
                diff = imgs_np[i, 0] - preds_np[i, 0]
                mse = np.mean(diff**2)
                mse_list.append(mse)
    return np.array(mse_list)

# Compute MSEs
mse_cdm = compute_mse(cdm_loader)
print(f"Computed {len(mse_cdm)} MSE values for CDM dataset.")
mse_wdm = compute_mse(wdm_loader)
print(f"Computed {len(mse_wdm)} MSE values for WDM dataset.")

# Plot histograms
plt.figure(figsize=(8, 6))
plt.hist(mse_cdm, bins=50, alpha=0.5, label='CDM')
plt.hist(mse_wdm, bins=50, alpha=0.5, label='WDM')
plt.xlabel('Mean Squared Error per Image')
plt.ylabel('Frequency')
plt.title('Histogram of Image-wise MSE for CDM vs WDM')
plt.legend()
plt.tight_layout()
plt.savefig("cosmo_compression/results/mse_histogram.png")

# Print mean and median
print(f"Mean MSE for CDM: {np.mean(mse_cdm):.4f}")
print(f"Median MSE for CDM: {np.median(mse_cdm):.4f}")
print(f"Mean MSE for WDM: {np.mean(mse_wdm):.4f}")
print(f"Median MSE for WDM: {np.median(mse_wdm):.4f}")