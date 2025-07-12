import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from cosmo_compression.data import data
from cosmo_compression.model import represent
from cosmo_compression.downstream import param_est_model as pe
import Pk_library as PKL
import scienceplots

# ----------------------------
# Plotting style (SciencePlots)
# ----------------------------
plt.style.use("science")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({
    "font.size": 16,           # Base font size for labels, ticks, legend
    "axes.titlesize": 20,      # Font size for the title
    "axes.labelsize": 16,      # Font size for x/y labels
})

# ----------------------------------------------
# 0) Configuration: map type, resolution, seeds
# ----------------------------------------------
MAP_TYPE = "Mcdm"
MAP_RESOLUTION = 256

# normalization dicts for “mean” and “std”
mean = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["mean"]
std  = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["std"]

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# --------------------------------------------
# 1) Paths to pretrained Flow and Param models
# --------------------------------------------
fm_model_path    = "latent_ablation_workshop_outc/no_hierarchical_8/step=step=15500-val_loss=0.327.ckpt"
param_model_path = "latent_ablation_workshop_outc/best_model_latent_pe.pt"

# -----------------------------------
# 2) Load dataset, Flow‐Matching model
# -----------------------------------
dataset = data.CAMELS(
    map_type=MAP_TYPE,
    dataset="1P",
    parameters=["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "Omega_b"],
)

# --- Grab two samples *before* we convert to torch:
(orig_img0, true_params0) = dataset[1]   # source sample as NumPy array
(orig_img1, true_params1) = dataset[61]  # target sample

# We may want to un‐normalize these raw images for display.
# dataset[...] typically returns a *normalized* image (in [–1, 1] or similar).
# But in your decoding pipeline you un‐normalize by (×std + mean). We do the same here:
orig_img0 = orig_img0 * std + mean    # shape: (256, 256)
orig_img1 = orig_img1 * std + mean

# Convert to torch tensors (for encoder), but keep the NumPy originals above
img0 = torch.tensor(dataset[1][0]).unsqueeze(0).to(device)  # shape (1, 256,256)
img1 = torch.tensor(dataset[61][0]).unsqueeze(0).to(device)

# Load the pretrained Flow‐Matching model
fm = represent.Represent.load_from_checkpoint(fm_model_path).to(device)
fm.eval()

# -------------------------------------
# 3) Load the pretrained Param model
# -------------------------------------
param_estimator = pe.ParamEstVec(
    hidden_dim=1545, num_hiddens=1, in_dim=8, output_size=2
)
param_estimator.load_state_dict(torch.load(param_model_path))
param_estimator.eval().to(device)

# -------------------------------------------------------
# 4) Encode source & target to latent space
# -------------------------------------------------------
with torch.no_grad():
    h0 = fm.encoder(img0)  # shape: (1, C, H', W')
    h1 = fm.encoder(img1)

# -----------------------------------------------
# 5) Build a linear interpolation in latent space
# -----------------------------------------------
n_interpolations = 20
alphas = np.linspace(0.0, 1.0, n_interpolations)

decoded_images = []
estimated_params = []  # Will be shape (n_interpolations, 2)

# We’ll feed a fixed “x0 noise” into the decoder each time:
x0 = torch.randn((1, 1, MAP_RESOLUTION, MAP_RESOLUTION), device=device)

label_means = torch.tensor([0.2993, 0.7986], device=device)
label_stds  = torch.tensor([0.1151, 0.1152], device=device)

for alpha in alphas:
    # Linear interpolation between h0 and h1
    h_interp = (1.0 - alpha) * h0 + alpha * h1  # shape: (1, C, H', W')
    
    # Decode to pixel‐space image
    with torch.no_grad():
        decoded = fm.decoder.predict(x0, h=h_interp.clone(), n_sampling_steps=40)
    # decoded: shape (1, 1, 256, 256) in *normalized* units
    image_np = decoded.cpu().numpy()[0, 0] * std + mean  # un‐normalize back to real space
    
    decoded_images.append(image_np)
    
    # ------------------------------------------------------
    # Parameter estimate from the latent interpolation
    # ------------------------------------------------------
    with torch.no_grad():
        # The param_estimator expects the pooled latent; replicate your original pipeline:
        pooled = fm.decoder.velocity_model.pool(h_interp).squeeze()
        preds = param_estimator(pooled)
    preds_np = (preds.cpu() * label_stds.cpu() + label_means.cpu()).numpy().flatten()
    estimated_params.append(preds_np)

estimated_params = np.stack(estimated_params, axis=0)  # shape: (20, 2)

true_params0 = np.array(true_params0)  # e.g. [Ω_m0, σ₈₀, …]
true_params1 = np.array(true_params1)

# --------------------------------------------------------
# 6) Plotting:
#    1 × 8 subplots:
#      [0] = raw source (orig_img0)
#      [1–5] = selected decoded snapshots (including endpoints)
#      [6] = raw target (orig_img1)
#      [7] = parameter‐estimate vs α
# --------------------------------------------------------
snapshot_indices = np.linspace(0, n_interpolations - 1, 4, dtype=int)
snapshot_indices = snapshot_indices[1:-1]
n_decoded = len(snapshot_indices)  # =5

n_cols = n_decoded + 3  # 5 decoded + source + target + param‐plot = 8

fig, axs = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))

# ---------------------------------------------
# (1) Plot raw source image (before ANY decoding)
# ---------------------------------------------
ax_src = axs[0]
ax_src.imshow(orig_img0.squeeze(), cmap="viridis", origin="lower")
ax_src.set_title("Source") 
ax_src.axis("off")

# ---------------------------------------------
# (2) Plot the 5 decoded snapshots
# ---------------------------------------------
for i, idx in enumerate(snapshot_indices):
    ax_dec = axs[1 + i]
    alpha_val = alphas[idx]
    ax_dec.imshow(decoded_images[idx], cmap="viridis", origin="lower")
    ax_dec.set_title(r"$\alpha$ = " + f"{alpha_val:.2f}")
    ax_dec.axis("off")

# ---------------------------------------------
# (3) Plot raw target image
# ---------------------------------------------
ax_tgt = axs[1 + n_decoded]
ax_tgt.imshow(orig_img1.squeeze(), cmap="viridis", origin="lower")
ax_tgt.set_title("Target")
ax_tgt.axis("off")

# ---------------------------------------------
# (4) Plot parameter estimates vs α
# ---------------------------------------------
ax_params = axs[-1]
param_names = [r"$\Omega_m$"]  # only plotting the first parameter here

# Plot the estimated Ω_m from each decoded interpolation
ax_params.plot(
    alphas,
    estimated_params[:, 0],
    label=param_names[0],
    linewidth=1.5
)

# Also overlay the “true” linear‐in‐α trajectory (for Ω_m)
true_line = (1 - alphas) * true_params0[0] + alphas * true_params1[0]
ax_params.plot(
    alphas,
    true_line,
    linestyle="--",
    linewidth=1.0,
    color="k",
    alpha=0.5,
    label="True (linear)"
)

ax_params.set_xlabel(r"$\alpha$ (interpolation)")
ax_params.set_ylabel(r"$\Omega_m$")
ax_params.set_title(r"Parameter Estimate vs.\ $\alpha$")
ax_params.legend(ncol=1, loc="best", fontsize=12)
ax_params.grid(True)

plt.tight_layout()
plt.savefig("cosmo_compression/results/latent_interp_with_raw_endpoints_and_params.pdf")
plt.close()
