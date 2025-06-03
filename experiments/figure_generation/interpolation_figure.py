import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL

import scienceplots

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


from cosmo_compression.data import data
from cosmo_compression.model import represent

torch.manual_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

MAP_TYPE = "Mcdm"
MAP_RESOLUTION = 256

# normalization dicts for “mean” and “std”
mean = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["mean"]
std = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["std"]

device = "cuda" if torch.cuda.is_available() else "cpu"

fm_model_path = "latent_ablation_workshop_outc/no_hierarchical_8/step=step=15500-val_loss=0.327.ckpt"

#-----------------------------------
# 1) Load dataset and pretrained model
#-----------------------------------
dataset = data.CAMELS(
    map_type=MAP_TYPE,
    dataset="1P",
    parameters=["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "Omega_b"],
)

fm = represent.Represent.load_from_checkpoint(
    fm_model_path
).to(device)
fm.eval()

#-----------------------------------
# 2) Grab two sample images (original and target)
#-----------------------------------
img0, _ = dataset[1]    # “original” 
img1, _ = dataset[61]   # “target/final”
img0 = torch.tensor(img0).unsqueeze(0).to(device)
img1 = torch.tensor(img1).unsqueeze(0).to(device)

# Encode them
h0 = fm.encoder(img0)
h1 = fm.encoder(img1)

# Convert to NumPy and un-normalize (for plotting & P(k))
img0_np = img0.cpu().numpy()[0, 0] * std + mean
img1_np = img1.cpu().numpy()[0, 0] * std + mean

# Compute delta = (map / mean) - 1, then P(k) for original & final
delta0 = img0_np / np.mean(img0_np) - 1
delta1 = img1_np / np.mean(img1_np) - 1

Pk0_2D = PKL.Pk_plane(delta0, 25.0, "None", 1, verbose=False)
Pk1_2D = PKL.Pk_plane(delta1, 25.0, "None", 1, verbose=False)

k0, Pk0 = Pk0_2D.k, Pk0_2D.Pk
k1, Pk1 = Pk1_2D.k, Pk1_2D.Pk

#-----------------------------------
# 3) Define modulation stages (same as before)
#-----------------------------------
modulation_ranges = {
    "High Frequency Interpolation": list(range(0, 2)),
    "Low Frequency Interpolation": list(range(2, 8)),
}

n_sampling_steps = 40
x0 = torch.randn((1, 1, 256, 256), device=device)

#-----------------------------------
# 4) Build interpolated results
#-----------------------------------
current_h = h0.clone()
intermediate_results = []  # we’ll keep (label, decoded_image, k, Pk) for each stage

for label, channels in modulation_ranges.items():
    interp_h = current_h.clone()
    interp_h[:, channels, :, :] = h1[:, channels, :, :]  # swap in those channels from h1
    
    with torch.no_grad():
        decoded = fm.decoder.predict(x0, h=interp_h.clone(), n_sampling_steps=n_sampling_steps)
    image = decoded.cpu().numpy()[0, 0] * std + mean

    # recompute P(k) for this interpolated image
    delta = image / np.mean(image) - 1
    Pk2D = PKL.Pk_plane(delta, 25.0, "None", 1, verbose=False)
    k_interp, Pk_interp = Pk2D.k, Pk2D.Pk

    intermediate_results.append((label, image, k_interp, Pk_interp))
    # current_h = interp_h.clone()

#-----------------------------------
# 5) Plot all columns: [Original] + [Stage 0, Stage 1, Stage 2, Stage 3] + [Final]
#-----------------------------------
n_interpolation = len(intermediate_results)              # 4
n_cols = n_interpolation + 2                              # +2 for original & final
fig, axs = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

# --- Column 0: “Original” ---
axs[0, 0].imshow(img0_np, cmap="viridis", origin="lower")
axs[0, 0].set_title("Source CDM Field")
axs[0, 0].axis("off")

ax_pk_orig = axs[1, 0]
# ax_pk_orig.plot(k0, Pk0,       label="Current (Original)", color="C0")           # current = Pk0
ax_pk_orig.plot(k0, Pk0,       label="Original P(k)", color="C1", linestyle="--")  # identical to current
ax_pk_orig.plot(k1, Pk1,       label="Target P(k)",  color="C2", linestyle=":")   # final Pk
ax_pk_orig.set_xscale("log")
ax_pk_orig.set_yscale("log")
ax_pk_orig.set_xlabel("k")
ax_pk_orig.set_ylabel("P(k)")
ax_pk_orig.legend()

# --- Middle columns: each interpolation “Stage i” ---
for i, (label, image, k_interp, Pk_interp) in enumerate(intermediate_results):
    col = i + 1  # because col 0 is “Original”
    # Top row: the decoded interpolation image
    axs[0, col].imshow(image, cmap="viridis", origin="lower")
    axs[0, col].set_title(f"{label}")
    axs[0, col].axis("off")
    
    # Bottom row: P(k) curves (Current vs Original vs Target)
    ax_pk = axs[1, col]
    ax_pk.plot(k_interp, Pk_interp, label="Reconstruction", color="C0")
    ax_pk.plot(k0,       Pk0,       label="Original", color="C1", linestyle="--")
    ax_pk.plot(k1,       Pk1,       label="Target",   color="C2", linestyle=":")
    ax_pk.set_xscale("log")
    ax_pk.set_yscale("log")
    ax_pk.set_xlabel("k")
    ax_pk.set_ylabel("P(k)")
    if i == 0:
        ax_pk.legend()

# --- Last column: “Final” (target) ---
final_col = n_cols - 1
axs[0, final_col].imshow(img1_np, cmap="viridis", origin="lower")
axs[0, final_col].set_title("Target CDM Field")
axs[0, final_col].axis("off")

ax_pk_final = axs[1, final_col]
# ax_pk_final.plot(k1, Pk1,       label="Current (Final)", color="C0")   # current = Pk1
ax_pk_final.plot(k0, Pk0,       label="Original P(k)", color="C1", linestyle="--")
ax_pk_final.plot(k1, Pk1,       label="Target P(k)",   color="C2", linestyle=":")
ax_pk_final.set_xscale("log")
ax_pk_final.set_yscale("log")
ax_pk_final.set_xlabel("k")
ax_pk_final.set_ylabel("P(k)")
ax_pk_final.legend()

plt.tight_layout()
plt.savefig("cosmo_compression/results/workshop_figures/interpolation.png")
plt.close()