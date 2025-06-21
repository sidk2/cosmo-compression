import os
import torch
import torch.utils
import torch.nn as nn
from torch.utils import data as torchdata
import lightning
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything
import Pk_library as PKL
import numpy as np
import wandb
from cosmo_compression.data import data
from cosmo_compression.model import represent
from matplotlib.colors import Normalize
import scienceplots

# Use the science style and set base font sizes
plt.style.use("science")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({
    "font.size": 16,           # Base font size for labels, ticks, legend
    "axes.titlesize": 20,      # Font size for the title
})
#-----------------------------------
# 0) Setup & Load One Sample
#-----------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = "cuda" if torch.cuda.is_available() else "cpu"

MAP_TYPE = "Mcdm"
MAP_RESOLUTION = 256

val_data = data.CAMELS(
    parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'WDM'],
    suite="Astrid",
    dataset="LH",
    map_type=MAP_TYPE
)

fm_model_path = "latent_ablation_workshop_outc/no_hierarchical_8/step=step=15500-val_loss=0.327.ckpt"

# Pick a random validation example
y, cosmo = val_data[np.random.randint(0, len(val_data))]
y = torch.tensor(y).to(device).unsqueeze(0)  # shape [1, 1, 256, 256]

# Un-normalize for plotting the “true” field
y_phys = y * val_data.std + val_data.mean  # using val_data.std & val_data.mean

# Compute the original power spectrum once for reference
orig_np = y_phys.cpu().numpy()[0, 0]           # shape [256,256]
delta_orig = orig_np / np.mean(orig_np) - 1.0
Pk2D_orig = PKL.Pk_plane(delta_orig, 25.0, "None", 1, verbose=False)
k_orig, Pk_orig = Pk2D_orig.k, Pk2D_orig.Pk

# Plot original field
plt.figure()
plt.imshow(orig_np, cmap="viridis", origin="lower")
plt.title("Original Field")
plt.axis("off")
plt.savefig("cosmo_compression/results/field.png")
plt.close()

#-----------------------------------
# 1) Load Pretrained Model & Encode
#-----------------------------------
fm: lightning.LightningModule = represent.Represent.load_from_checkpoint(
    fm_model_path
).to(device)
fm.eval()

# Encode the un-normalized input (i.e., input is already un-normalized)
h = fm.encoder(y)

#-----------------------------------
# 2) Generate Full Trajectory of Reconstructions
#-----------------------------------
# Start from a random latent “x0”
x0 = torch.randn_like(y).to(device)

# Predict full trajectory: returns a list of length (n_sampling_steps + 1),
# each tensor [1, 1, 256, 256]
full_trajectory = fm.decoder.predict(
    x0,
    h=h,
    n_sampling_steps=30,
    full_return=True
)

# Optionally plot the final reconstruction alone
plt.figure()
final_np = full_trajectory[-1].cpu().numpy()[0, 0]
plt.imshow(final_np, cmap="viridis", origin="lower")
plt.title("Final Reconstruction")
plt.axis("off")
plt.savefig("cosmo_compression/results/reconstruction.png")
plt.close()

#-----------------------------------
# 3) Plot Reconstructions + Power Spectra
#-----------------------------------
num_steps = len(full_trajectory)
timesteps = [1, num_steps // 3, 2 * num_steps // 3, num_steps - 1]

fig, axs = plt.subplots(2, len(timesteps), figsize=(4 * len(timesteps), 8))

for idx, t in enumerate(timesteps):
    # 3a) Get the reconstructed field at step t, un-normalize it
    recon_t = full_trajectory[t]                   # tensor shape [1,1,256,256]
    recon_np = recon_t.cpu().numpy()[0, 0]         # shape [256,256]
    recon_phys = recon_np * val_data.std + val_data.mean

    # Top row: show the image
    ax_img = axs[0, idx]
    ax_img.imshow(recon_phys, cmap="viridis", origin="lower")
    ax_img.set_title(f"Reconstruction\n t={t/ (num_steps - 1) : 0.2f}")
    ax_img.axis("off")

    # 3b) Compute power spectrum P(k) of the reconstruction
    delta = recon_phys / np.mean(recon_phys) - 1.0
    Pk2D = PKL.Pk_plane(delta, 25.0, "None", 1, verbose=False)
    k_vals, Pk_vals = Pk2D.k, Pk2D.Pk

    # Bottom row: plot log‐log P(k), and overlay original spectrum
    ax_pk = axs[1, idx]
    ax_pk.plot(k_vals, (1 / (t / num_steps)**2)*Pk_vals, label=f"CosmoFlow", color="C0")
    ax_pk.plot(k_orig, Pk_orig, 'k--', label="Original", color="C1")
    ax_pk.set_xscale("log")
    ax_pk.set_yscale("log")
    ax_pk.set_xlabel("k [h/Mpc]")
    ax_pk.set_ylabel("P(k) [h/Mpc]$^2$")
    ax_pk.legend()

plt.tight_layout()
plt.savefig("cosmo_compression/results/trajectory_reconstructions.pdf")
plt.close()
