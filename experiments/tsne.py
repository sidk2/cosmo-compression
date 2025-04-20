import os
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.ndimage import gaussian_filter

import optuna
from torch.utils.data import DataLoader, TensorDataset, random_split

from cosmo_compression.data import data
from cosmo_compression.model import represent
from cosmo_compression.downstream import anomaly_det_model as ad

# Set random seed for reproducibility
torch.manual_seed(90)

# Define file paths for caching latent representations (if needed)
save_dir = "../../../monolith/global_data/astro_compression/CAMELS/images"
os.makedirs(save_dir, exist_ok=True)
cdm_latents_path = os.path.join(save_dir, "cdm_latents.npy")
cdm_params_path = os.path.join(save_dir, "cdm_params.npy")
wdm_latents_path = os.path.join(save_dir, "wdm_latents.npy")
wdm_params_path = os.path.join(save_dir, "wdm_params.npy")

use_latents = True

# Set the CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the pre-trained model checkpoint
fm = represent.Represent.load_from_checkpoint("reversion_2_126lat/step=step=60600-val_loss=0.232.ckpt")
fm.encoder = fm.encoder.cuda()
for p in fm.encoder.parameters():
    p.requires_grad = False
fm.eval()

# Set number of sampling steps for the decoder (if needed elsewhere)
n_sampling_steps = 30

###############################################################################
# Helper function: Extract images and parameters from a dataset
###############################################################################
def get_images_and_params(dataset):
    images = []
    params = []
    for i in range(len(dataset)):
        img, par = dataset[i]
        images.append(img)
        params.append(par)
    return np.stack(images), np.array(params)

###############################################################################
# Load CDM and WDM datasets
###############################################################################
cdm_data = data.CAMELS(
    idx_list=range(15000),
    parameters=['Omega_m', 'sigma_8', "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"],
    suite="IllustrisTNG", dataset="LH", map_type="Mcdm"
)
wdm_data = data.CAMELS(
    idx_list=range(15000),
    parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_AGN2', 'WDM'],
    suite="IllustrisTNG", dataset="WDM", map_type="Mcdm"
)

###############################################################################
# Extract images and parameters for CDM and WDM datasets
###############################################################################
print("Extracting CDM images and parameters...")
cdm_imgs, cdm_params = get_images_and_params(cdm_data)
print("Extracting WDM images and parameters...")
wdm_imgs, wdm_params = get_images_and_params(wdm_data)

###############################################################################
# Filter WDM samples by the threshold on the last parameter ("WDM" value)
###############################################################################
threshold = 0.5
# Assuming that the last column of wdm_params corresponds to the "WDM" parameter:
wdm_filter = wdm_params[:, -1] > threshold
wdm_imgs_filtered = wdm_imgs[wdm_filter]
wdm_params_filtered = wdm_params[wdm_filter]
print(f"Filtered WDM samples: {len(wdm_imgs_filtered)} / {len(wdm_imgs)} remain (WDM > {threshold})")

###############################################################################
# Compute Latents using the Encoder
###############################################################################
def encode_latents(images):
    latent_list = []
    for img in tqdm.tqdm(images, desc="Encoding latents"):
        with torch.no_grad():
            # If images are 2D, add a channel dimension.
            # Here we assume images are in shape (H, W); after unsqueeze, shape becomes (1, H, W)
            # Adjust if your model expects a different input shape.
            img_tensor = torch.tensor(img).unsqueeze(0).float().cuda()
            spatial, vec = fm.encoder(img_tensor)
            latent_list.append(vec.cpu().numpy())
    return np.array(latent_list)

print("Encoding latents for CDM images...")
latents_cdm = encode_latents(cdm_imgs)
print("Encoding latents for filtered WDM images...")
latents_wdm = encode_latents(wdm_imgs_filtered)

###############################################################################
# Combine the latent representations and run T-SNE
###############################################################################
# Concatenate along the batch dimension
all_latents = np.concatenate([latents_cdm, latents_wdm], axis=0)
tsne = TSNE(n_components=2, random_state=42)
tsne_latents = tsne.fit_transform(all_latents)

###############################################################################
# Plot the T-SNE embeddings
###############################################################################
plt.figure(figsize=(10, 8))
n_cdm = latents_cdm.shape[0]
# Scatter plot for CDM (first group)
plt.scatter(tsne_latents[:n_cdm, 0], tsne_latents[:n_cdm, 1], label='CDM', alpha=0.6)
# Scatter plot for filtered WDM (second group)
plt.scatter(tsne_latents[n_cdm:, 0], tsne_latents[n_cdm:, 1], label='WDM (filtered)', alpha=0.6)
plt.legend()
plt.title("T-SNE of Latents: CDM vs. WDM (WDM > 0.5)")
plt.xlabel("T-SNE Dim 1")
plt.ylabel("T-SNE Dim 2")
plt.grid(True)
plt.tight_layout()

# Save the plot
save_path = "cosmo_compression/results/tsne_cdm_vs_wdm.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
print("T-SNE plot saved to:", save_path)
plt.show()
