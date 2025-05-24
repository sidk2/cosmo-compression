import os
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import lightning
from cosmo_compression.data import data
from cosmo_compression.model import represent

np.random.seed(41)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# — load data + model —
val_data = data.CAMELS(
    parameters=['Omega_m','sigma_8','A_SN1','A_SN2','A_AGN1','WDM'],
    suite="IllustrisTNG", dataset="WDM", map_type="Mcdm"
)
y, cosmo = val_data[np.random.randint(len(val_data))]
y = torch.tensor(y).to(device).unsqueeze(0)

fm = represent.Represent.load_from_checkpoint(
    "latent_ablation_non_hierarchical_splitting/no_hierarchical_64/step=step=25100-val_loss=0.258.ckpt"
).to(device)
fm.eval()

# — run encoder once —
normed = y
spatial = fm.encoder(normed)
# spatial shape: [1, C_feat, H, W]
spatial = spatial[0].detach().cpu().numpy()
n_feats = spatial.shape[0]

orig_img = (y[0,0].detach().cpu().numpy() * val_data.std + val_data.mean)
fig_orig, ax_orig = plt.subplots(figsize=(6,6))
ax_orig.imshow(orig_img, cmap='viridis')
ax_orig.set_title("Original Map")
ax_orig.axis('off')
fig_orig.tight_layout()
fig_orig.savefig("cosmo_compression/results/original_map.png", dpi=200)
plt.close(fig_orig)


n_feats = spatial.shape[0]
ncols  = min(8, n_feats)
nrows  = math.ceil(n_feats / ncols)

fig_latent, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
axes = axes.flatten()

for i in range(n_feats):
    ch = spatial[i]
    print(np.mean(ch), np.std(ch))
    vmin, vmax = ch.min(), ch.max()
    ax = axes[i]
    im = ax.imshow(ch, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f"Feat {i}", fontsize=8)
    ax.axis('off')

# Turn off any extra axes
for ax in axes[n_feats:]:
    ax.axis('off')

fig_latent.tight_layout()
fig_latent.savefig("cosmo_compression/results/latent_features.png", dpi=200)
plt.close(fig_latent)
