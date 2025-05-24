from typing import List
import os
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from cosmo_compression.data import data
from cosmo_compression.model import represent
import Pk_library as PKL  # if still needed

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load normalization constants
mean = data.NORM_DICT["Mcdm"][256]["mean"]
std = data.NORM_DICT["Mcdm"][256]["std"]

diffs_cdm = np.load("../../../monolith/global_data/astro_compression/CAMELS/images/64/diffs_cdm.npy")
diffs_wdm = np.load("../../../monolith/global_data/astro_compression/CAMELS/images/64/diffs_wdm.npy")

# Compute T-SNE of diffs
X_cdm = diffs_cdm.reshape(diffs_cdm.shape[0], -1)
X_wdm = diffs_wdm.reshape(diffs_wdm.shape[0], -1)

# 3) Stack and optionally reduce dimensionality with PCA
X = np.vstack([X_cdm, X_wdm])
labels = np.array([0]*len(X_cdm) + [1]*len(X_wdm))  # 0=CDM, 1=WDM

# PCA to speed up t-SNE (helps when original dim >> samples)
pca = PCA(n_components=50, whiten=True, random_state=42)
X_pca = pca.fit_transform(X)

# 4) Run t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='random',
    random_state=42,
    n_iter=1000,
    verbose=1
)
X_tsne = tsne.fit_transform(X_pca)

# 5) Plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    X_tsne[:,0], X_tsne[:,1],
    c=labels,
    cmap='viridis',
    alpha=0.7,
    s=10
)
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=['CDM','WDM'],
    title='Type'
)
plt.title("t-SNE of diffs_cdm vs diffs_wdm")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.tight_layout()
plt.savefig("cosmo_compression/results/tsne.png")

mae_cdm = np.mean(np.abs(diffs_cdm), axis=(1, 2, 3))
mae_wdm = np.mean(np.abs(diffs_wdm), axis=(1, 2, 3))

# Plot histograms
plt.figure(figsize=(8, 6))
plt.hist(mae_cdm, bins=50, alpha=0.5, label='CDM')
plt.hist(mae_wdm, bins=50, alpha=0.5, label='WDM')
plt.xlabel('Mean Squared Error per Image')
plt.ylabel('Frequency')
plt.title('Histogram of Image-wise MSE for CDM vs WDM')
plt.legend()
plt.tight_layout()
plt.savefig("cosmo_compression/results/mse_histogram.png")

# Print mean and median
print(f"Mean MSE for CDM: {np.mean(mae_cdm):.4f}")
print(f"Median MSE for CDM: {np.median(mae_cdm):.4f}")
print(f"Mean MSE for WDM: {np.mean(mae_wdm):.4f}")
print(f"Median MSE for WDM: {np.median(mae_wdm):.4f}")