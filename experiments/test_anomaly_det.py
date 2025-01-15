from cosmo_compression.data import data
from cosmo_compression.model import represent

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data
wdm_data = data.CAMELS(
    idx_list=range(10000),
        parameters=['Omega_m', 'sigma_8', 'WDM'],
        suite="IllustrisTNG",
        dataset="WDM",
        map_type="Mcdm",
    )

cdm_data = data.CAMELS(
    idx_list=range(10000),
        parameters=['Omega_m', 'sigma_8'],
        suite="Astrid",
        dataset="LH",
        map_type="Mcdm",
    )

fm = represent.Represent.load_from_checkpoint("img-lat-128ch/step=step=31000-val_loss=0.348.ckpt").cuda()
fm.eval()

# Compute CDM latents
cdm_latents = []
cdm_params = []

print("Computing CDM latents")
for i, (cosmo, data) in enumerate(cdm_data):
    with torch.no_grad():
        data = torch.tensor(data)
        data = data.cuda()
        latent, img = fm.encoder(data.unsqueeze(0))
        data = data.cpu()
        latent = latent.cpu()

        cdm_latents.append(latent)
        cdm_params.append(cosmo)

cdm_latents = np.array(cdm_latents).squeeze()
cdm_params = np.array(cdm_params).squeeze()

# Compute total matter latents
print("Computing total matter latents")
wdm_latents = []
wdm_params = []

for i, (cosmo, data) in enumerate(wdm_data):
    with torch.no_grad():
        data = torch.tensor(data)
        data = data.cuda()
        latent, img = fm.encoder(data.unsqueeze(0))
        data = data.cpu()
        latent = latent.cpu()

        wdm_latents.append(latent)
        wdm_params.append(cosmo)

wdm_latents = np.array(wdm_latents).squeeze()
wdm_params = np.array(wdm_params).squeeze()

# Combine latents and labels
all_latents = np.vstack([cdm_latents, wdm_latents])
labels = np.array(["CDM"] * len(cdm_latents) + ["WDM"] * len(wdm_latents))

# Perform T-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=10, verbose=True)
latents_2d = tsne.fit_transform(all_latents)

# Plot T-SNE
plt.figure(figsize=(10, 8))

# Plot CDM latents
cdm_idx = labels == "CDM"
plt.scatter(latents_2d[cdm_idx, 0], latents_2d[cdm_idx, 1], c="red", label="CDM", alpha=0.7)

# Plot WDM latents colored by the WDM parameter
wdm_idx = labels == "WDM"
wdm_wdm_values = wdm_params[:, 2]  # Assuming the WDM parameter is the 3rd column
scatter = plt.scatter(latents_2d[wdm_idx, 0], latents_2d[wdm_idx, 1], c=wdm_wdm_values, cmap="viridis", label="WDM", alpha=0.7)

plt.colorbar(scatter, label="WDM Parameter Value")
plt.title("T-SNE of Latents")
plt.xlabel("TSNE Component 1")
plt.ylabel("TSNE Component 2")
plt.legend()

plt.savefig('cosmo_compression/results/wdm_cdm_tsne.png')