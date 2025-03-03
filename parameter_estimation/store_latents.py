import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

from torch.utils.data import DataLoader

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

from cosmo_compression.data.data import CAMELS
from cosmo_compression.model.represent import Represent 

import torch 

cdm_data = CAMELS(
        idx_list=range(15000),
        map_type='Mcdm',
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2'],
    )
mtot_data = CAMELS(
        idx_list=range(15000),
        map_type='Mtot',
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2'],
    )

cdm_data_loader = DataLoader(
    cdm_data,
    batch_size=1,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

mtot_data_loader = DataLoader(
    mtot_data,
    batch_size=1,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

fm = Represent.load_from_checkpoint("img-lat-128ch/step=step=31000-val_loss=0.348.ckpt").cuda()
fm.eval()

cdm_latents = []
cdm_params = []

print("Computing CDM latents")
for cosmo, data in cdm_data_loader:
    with torch.no_grad():
        data = data.cuda()
        latent, img = fm.encoder(data)
        data = data.cpu()
        latent = latent.cpu()
        cosmo = cosmo.cpu()
        
        cdm_latents.append(latent)
        cdm_params.append(cosmo)
                
cdm_latents = np.array(cdm_latents).squeeze()
cdm_params = np.array(cdm_params).squeeze()

print("Computing total matter latents")

mtot_latents = []
mtot_params = []

for cosmo, data in mtot_data_loader:
    with torch.no_grad():
        data = data.cuda()
        latent, img = fm.encoder(data)
        data = data.cpu()
        latent = latent.cpu()
        cosmo = cosmo.cpu()
        
        mtot_latents.append(latent)
        mtot_params.append(cosmo)
                
mtot_latents = np.array(mtot_latents).squeeze()
mtot_params = np.array(mtot_params).squeeze()


# # Flatten the latents arrays for t-SNE
# # cdm_latents_flat = cdm_latents.reshape(len(cdm_latents), -1)
# # mtot_latents_flat = mtot_latents.reshape(len(mtot_latents), -1)

# # Perform t-SNE on combined data
# # latents_combined = np.concatenate([cdm_latents_flat, mtot_latents_flat], axis=0)

# # print("Performing T-SNE on the combined latents")

# # Split t-SNE results back into separate datasets
# # cdm_tsne = latents_2d[:len(cdm_latents)]
# # mtot_tsne = latents_2d[len(cdm_latents):]

# # Compute opacities from the first element of params
# # cdm_opacities = mtot_params[:, 1] / np.max(mtot_params[:, 1]) / 2
# # mtot_opacities = mtot_params[:, 0] / np.max(mtot_params[:, 0])

# # Bin the values in cdm_params[:, 1] into 6 groups
# param_values = mtot_params[:, 0]
# bins = np.linspace(np.min(param_values), np.max(param_values), 5)  # 6 bins = 7 edges
# bin_indices = np.digitize(param_values, bins) - 1  # Bin indices (0 to 5)
# print(np.min(bin_indices), np.max(bin_indices))
# tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# cdm_tsne = tsne.fit_transform(mtot_latents)
# # Define a color map with 6 distinct colors
# colors = ['salmon', 'palegoldenrod', 'olivedrab', 'steelblue', 'plum']
# cdm_colors = [colors[i] for i in bin_indices]

# # Plot t-SNE visualization
# plt.figure(figsize=(10, 8))

# scatter = plt.scatter(
#     cdm_tsne[:, 0],
#     cdm_tsne[:, 1],
#     c=cdm_colors,
#     alpha=0.7,
#     edgecolor='none'
# )

# # Create a legend for the bins
# handles = [plt.Line2D([0], [0], marker='o', color=color, markersize=10, linestyle='') for color in colors]
# labels = [f'Bin {i+1}: {bins[i]:.2f} to {bins[i+1]:.2f}' for i in range(len(bins)-1)]
# plt.legend(handles, labels, title='CDM Parameter Bins', loc='best', fontsize='small')

# plt.title("t-SNE of Latent Space of $M_{tot}$ (Color-Coded by $\Omega_m$)")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
# plt.grid(True)

# # Save the plot

# plt.savefig('cosmo_compression/results/t_sne_mtot_cdm.png')
np.save('cosmo_compression/parameter_estimation/data/mtot_latents.npy', mtot_latents)
np.save('cosmo_compression/parameter_estimation/data/mtot_params.npy', mtot_params)
np.save('cosmo_compression/parameter_estimation/data/cdm_latents.npy',cdm_latents)
np.save('cosmo_compression/parameter_estimation/data/cdm_params.npy', cdm_params)