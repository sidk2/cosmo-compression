import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import torch
import lightning
import Pk_library as PKL

from cosmo_compression.model import represent
from cosmo_compression.data import data

plt.style.use("science")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({
    "font.size": 20,           # Base font size for labels, ticks, legend
    "axes.titlesize": 20,      # Font size for the title
    "axes.labelsize": 20,      # Font size for x/y labels
    "xtick.labelsize": 20,     # Font size for x‐tick labels
    "ytick.labelsize": 20,     # Font size for y‐tick labels
    "legend.fontsize": 14      # Font size for legend
})

# Path to VAE data
vae_data_path = "cosmo_compression/data/paper_data/vae_field_reconstruction_0602.npy"
vae_output = np.load(vae_data_path)

# Path to model
fm_model_path = "latent_ablation_workshop_outc/no_hierarchical_8/step=step=15500-val_loss=0.327.ckpt"
fm: lightning.LightningModule = represent.Represent.load_from_checkpoint(
    fm_model_path
).to('cuda')

dataset = data.CAMELS(
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2'],
        suite="Astrid",
        dataset="LH",
        map_type="Mcdm"
    )

y, cosmo = dataset[14000]
y = torch.tensor(y).cuda().unsqueeze(0)
batch = y, cosmo

fm.eval()
x0 = torch.randn_like(y)

h = fm.encoder(y)
x1 = fm.decoder.predict(x0, h, n_sampling_steps=50)

y = y * dataset.std + dataset.mean
x1 = x1 * dataset.std + dataset.mean
vae_output = vae_output * dataset.std + dataset.mean

# Compute power spectra

y = y.detach().cpu().numpy().squeeze()
x1 = x1.detach().cpu().numpy().squeeze()
vae_output = vae_output.squeeze()

orig_overdensity = y / np.mean(y) - 1
fm_output_overdensity = x1 / np.mean(x1) - 1
vae_output_overdensity = vae_output / np.mean(vae_output) - 1

orig_pk = PKL.Pk_plane(orig_overdensity, 25.0, "None", 1, verbose=False)
orig_k, orig_pk = orig_pk.k, orig_pk.Pk

fm_pk = PKL.Pk_plane(fm_output_overdensity, 25.0, "None", 1, verbose=False)
fm_k, fm_pk = fm_pk.k, fm_pk.Pk

vae_pk = PKL.Pk_plane(vae_output_overdensity, 25.0, "None", 1, verbose=False)
vae_k, vae_pk = vae_pk.k, vae_pk.Pk
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# Plot original field on left
axs[0, 0].imshow(y)
axs[0, 0].axis("off")
axs[0, 0].set_title("Original Field")

# Plot VAE reconstruction in the middle
axs[0, 1].imshow(vae_output)
axs[0, 1].axis("off")
axs[0, 1].set_title("VAE Reconstruction")

# Plot FM reconstruction on right
axs[1, 0].imshow(x1)
axs[1, 0].axis("off")
axs[1, 0].set_title("CosmoFlow Reconstruction")

# Plot the power spectra of the reconstructions
axs[1, 1].plot(orig_k, orig_pk, label="Original")
axs[1, 1].plot(fm_k, fm_pk, label="CosmoFlow")
axs[1, 1].plot(vae_k, vae_pk, label="VAE")
axs[1, 1].set_title("Power Spectra")
axs[1, 1].set_xscale("log")
axs[1, 1].set_yscale("log")
axs[1, 1].set_xlabel("Wavenumber, $k$ [h/Mpc]")
axs[1, 1].set_ylabel("Power Spectrum [h/Mpc]$^2$")
axs[1, 1].legend()

plt.tight_layout()
plt.savefig("cosmo_compression/results/workshop_figures/vae_comparison.pdf")
