import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import torch
import lightning
import Pk_library as PKL
import tqdm
import umap

from cosmo_compression.model import represent
from cosmo_compression.data import data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
plt.style.use("science")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({
    "font.size": 16,           # Base font size for labels, ticks, legend
    "axes.titlesize": 20,      # Font size for the title
    "axes.labelsize": 14,      # Font size for x/y labels
    "xtick.labelsize": 12,     # Font size for x‐tick labels
    "ytick.labelsize": 12,     # Font size for y‐tick labels
    "legend.fontsize": 14      # Font size for legend
})


# -----------------------------------------------------------------------------
# 1) Load the pretrained FM model
# -----------------------------------------------------------------------------
fm_model_path = "latent_ablation_workshop_outc/no_hierarchical_8/step=step=15500-val_loss=0.327.ckpt"
fm: lightning.LightningModule = represent.Represent.load_from_checkpoint(
    fm_model_path
).to("cuda")
fm.eval()

# -----------------------------------------------------------------------------
# 2) Define WDM and CDM CAMELS datasets
#
#    - For WDM, we need to include "WDM" as the last parameter so we can color-code by mass.
#    - For CDM, we simply load maps with the standard cosmological parameters (no "WDM" field).
# -----------------------------------------------------------------------------
wdm_dataset = data.CAMELS(
    parameters=["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "WDM"],
    suite="IllustrisTNG",
    dataset="WDM",
    map_type="Mcdm",
)

cdm_dataset = data.CAMELS(
    parameters=["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "Omega_b"],
    suite="IllustrisTNG",
    dataset="LH",
    map_type="Mcdm",
)

# -----------------------------------------------------------------------------
# 3) Choose how many samples to encode for UMAP (you can raise/lower these)
# -----------------------------------------------------------------------------
N_WDM = 1000  # Number of WDM maps to encode
N_CDM = 1000  # Number of CDM maps to encode

# Pre-allocate lists
latents_wdm = []
wdm_masses  = []  # to store the last "WDM" entry from each cosmo vector
latents_cdm = []

# -----------------------------------------------------------------------------
# 4) Encode the first N_WDM WDM samples, collecting their latent + mass
# -----------------------------------------------------------------------------
print("Encoding WDM latents:")
for idx in tqdm.tqdm(range(N_WDM), desc=" WDM → encoder"):
    y_map, cosmo = wdm_dataset[idx]
    # y_map: NumPy array of shape (1, H, W); cosmo: length-7 array, where cosmo[-1] = WDM mass
    y_tensor = torch.tensor(y_map).unsqueeze(0).to("cuda")  # shape (1, 1, H, W)
    with torch.no_grad():
        h = fm.encoder(y_tensor)                            # h: (1, C_lat, H_lat, W_lat)
    h_np = h.cpu().numpy().reshape(-1)                      # flatten to 1D feature vector
    latents_wdm.append(h_np)
    wdm_masses.append(float(cosmo[-1]))                     # last entry = WDM mass

latents_wdm = np.stack(latents_wdm, axis=0)  # shape (N_WDM, latent_dim)
wdm_masses  = np.array(wdm_masses)           # shape (N_WDM,)

# -----------------------------------------------------------------------------
# 5) Encode the first N_CDM CDM samples, collecting only their latents
# -----------------------------------------------------------------------------
print("Encoding CDM latents:")
for idx in tqdm.tqdm(range(N_CDM), desc=" CDM → encoder"):
    y_map, _ = cdm_dataset[idx]
    y_tensor = torch.tensor(y_map).unsqueeze(0).to("cuda")
    with torch.no_grad():
        h = fm.encoder(y_tensor)
    h_np = h.cpu().numpy().reshape(-1)
    latents_cdm.append(h_np)

latents_cdm = np.stack(latents_cdm, axis=0)  # shape (N_CDM, latent_dim)

# -----------------------------------------------------------------------------
# 6) Concatenate the two latent sets and run UMAP
# -----------------------------------------------------------------------------
all_latents = np.concatenate([latents_wdm, latents_cdm], axis=0)  # (N_WDM + N_CDM, latent_dim)

print("Running UMAP on combined latents (this may take a minute)...")
umap_embed = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean"
).fit_transform(all_latents)
# umap_embed: shape (N_WDM + N_CDM, 2)

# -----------------------------------------------------------------------------
# 7) Split the embedding back into WDM vs CDM
# -----------------------------------------------------------------------------
umap_wdm = umap_embed[:N_WDM]          # first N_WDM rows
umap_cdm = umap_embed[N_WDM : N_WDM + N_CDM]  # next N_CDM rows

# -----------------------------------------------------------------------------
# 8) Find the WDM sample with the maximum value of the last entry in cosmo (WDM mass)
#    by iterating over the ENTIRE WDM dataset
# -----------------------------------------------------------------------------
print("Finding WDM sample with maximum WDM mass across entire dataset:")
max_wdm_mass_value = -np.inf
max_wdm_mass_idx = None

for idx in tqdm.tqdm(range(len(wdm_dataset)), desc="Searching for max WDM mass"):
    _, cosmo = wdm_dataset[idx]
    wdm_mass = float(cosmo[-1])  # last entry = WDM mass
    
    if wdm_mass > max_wdm_mass_value:
        max_wdm_mass_value = wdm_mass
        max_wdm_mass_idx = idx

print(f"Selected WDM index (max WDM mass) = {max_wdm_mass_idx}, WDM mass = {max_wdm_mass_value:.4e}")

# -----------------------------------------------------------------------------
# 9) Perform multiple reconstructions of the selected WDM sample to analyze power spectrum scatter
# -----------------------------------------------------------------------------
N_RECONSTRUCTIONS = 1  # Number of reconstructions to perform

print(f"Performing {N_RECONSTRUCTIONS} reconstructions of selected WDM sample:")
y_sel, cosmo_sel = wdm_dataset[max_wdm_mass_idx]
y_sel_tensor = torch.tensor(y_sel).unsqueeze(0).to("cuda")

# Encode once
with torch.no_grad():
    h_sel = fm.encoder(y_sel_tensor)

# Compute original field statistics
y_denorm_sel = (y_sel_tensor.detach().cpu().numpy().squeeze() * wdm_dataset.std) + wdm_dataset.mean
orig_overdensity_sel = y_denorm_sel / np.mean(y_denorm_sel) - 1
orig_pk_sel = PKL.Pk_plane(orig_overdensity_sel, 25.0, "None", 1, verbose=False)

# Perform multiple reconstructions
reconstruction_pks = []
for i in tqdm.tqdm(range(N_RECONSTRUCTIONS), desc="Generating reconstructions"):
    with torch.no_grad():
        x0_sel = torch.randn_like(y_sel_tensor)
        x1_sel = fm.decoder.predict(x0_sel, h_sel.clone(), n_sampling_steps=50)
    
    x1_denorm_sel = (x1_sel.detach().cpu().numpy().squeeze() * wdm_dataset.std) + wdm_dataset.mean
    fm_overdensity_sel = x1_denorm_sel / np.mean(x1_denorm_sel) - 1
    fm_pk_sel = PKL.Pk_plane(fm_overdensity_sel, 25.0, "None", 1, verbose=False)
    reconstruction_pks.append(fm_pk_sel.Pk)

reconstruction_pks = np.array(reconstruction_pks)  # Shape: (N_RECONSTRUCTIONS, n_k_bins)

# Use the first reconstruction for the field visualization
with torch.no_grad():
    x0_sel = torch.randn_like(y_sel_tensor)
    x1_sel_display = fm.decoder.predict(x0_sel, h_sel, n_sampling_steps=50)
x1_denorm_sel_display = (x1_sel_display.detach().cpu().numpy().squeeze() * wdm_dataset.std) + wdm_dataset.mean

# Now plot all panels:

fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# --- Panel 0: Selected WDM Field ---
axs[0, 0].imshow(y_denorm_sel, cmap="viridis", origin="lower")
axs[0, 0].axis("off")
axs[0, 0].set_title(f"WDM Field")

# --- Panel 1: FM Reconstruction of that WDM Field ---
axs[0, 1].imshow(x1_denorm_sel_display, cmap="viridis", origin="lower")
axs[0, 1].axis("off")
axs[0, 1].set_title("FM Reconstruction")

# --- Panel 2: Power Spectra with Reconstruction Scatter ---
# Plot original power spectrum
axs[1, 0].plot(orig_pk_sel.k, orig_pk_sel.Pk, label="Original", linestyle="-", color="C0", linewidth=1)

# Plot all reconstruction power spectra as scatter/thin lines
for i, pk_recon in enumerate(reconstruction_pks):
    axs[1, 0].plot(orig_pk_sel.k, pk_recon, alpha=1, color="C1", linewidth=0.8)

# Add a single label for reconstructions
axs[1, 0].plot([], [], alpha=1, color="C1", linewidth=1, label=f"CosmoFlow")
axs[1, 0].set_ylim(5e-8, 1e-1)

axs[1, 0].set_xscale("log")
axs[1, 0].set_yscale("log")
axs[1, 0].set_xlabel("k [h/Mpc]")
axs[1, 0].set_ylabel("P(k) [h/Mpc]$^2$")
axs[1, 0].set_title("Power Spectra")
axs[1, 0].legend()

# --- Panel 3: UMAP Embedding of ALL latents (CDM vs. WDM) ---
#
#    - CDM points: gray (alpha=0.3)
#    - WDM points: colored by their WDM mass (as before)
#
sc_cdm = axs[1, 1].scatter(
    umap_cdm[:, 0],
    umap_cdm[:, 1],
    c="gray",
    s=8,
    alpha=0.2,
    label="CDM",
)

sc_wdm = axs[1, 1].scatter(
    umap_wdm[:, 0],
    umap_wdm[:, 1],
    c=wdm_masses,
    cmap="viridis",
    s=8,
    alpha=0.8,
    label="WDM",
)

axs[1, 1].set_title("UMAP of CDM vs. WDM\n Latents")
axs[1, 1].set_xticks([])
axs[1, 1].set_yticks([])
axs[1, 1].set_xlim(np.min(umap_cdm[:, 0]) - 1, np.max(umap_cdm[:, 0]) + 1)
axs[1, 1].set_ylim(np.min(umap_cdm[:, 1]) - 1, np.max(umap_cdm[:, 1]) + 1)

cbar = plt.colorbar(sc_wdm, ax=axs[1, 1], fraction=0.046, pad=0.04)
cbar.set_label("WDM Mass")
axs[1, 1].legend(fontsize="small")

plt.tight_layout()
plt.savefig("cosmo_compression/results/workshop_figures/wdm_cdm_umap_max_wdm_mass.pdf")
plt.close(fig)

print("Saved combined figure to:\n  cosmo_compression/results/workshop_figures/wdm_cdm_umap_max_wdm_mass.pdf")