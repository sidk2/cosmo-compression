from typing import List
import os

import torch
import torch.utils
import torch.nn as nn
from torch.utils import data as torchdata

import Pk_library as PKL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import lightning
import tqdm

from cosmo_compression.data import data
from cosmo_compression.model import represent
from cosmo_compression.parameter_estimation import inference

torch.manual_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

MAP_TYPE = "Mcdm"
MAP_RESOLUTION = 256

mean = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["mean"]
std = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["std"]

device: str = "cuda" if torch.cuda.is_available() else "cpu"

dataset: torchdata.Dataset = data.CAMELS(
    map_type=MAP_TYPE,
    dataset="LH",
    parameters=["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "Omega_b"],
)

loader = torchdata.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

# Load models
fm: lightning.LightningModule = represent.Represent.load_from_checkpoint(
    "cosmo_notimeslice_64ch/step=step=9900-val_loss=0.312.ckpt"
).to(device)
fm.eval()

gts = []

cosmo, img = dataset[120]
img = torch.tensor(img).unsqueeze(0).cuda()

start_latent = fm.encoder(img)

y_orig = img.cpu().numpy().squeeze()

cosmo, img = dataset[8205]
img = torch.tensor(img).unsqueeze(0).cuda()

end_latent = fm.encoder(img)

y_fin = img.cpu().numpy().squeeze()

# Initialize a list for the interpolated latents
h_linear = []

num_samples = 4

for i in range(num_samples):
    t = i / (num_samples - 1)  # Interpolation factor
    latent = (
        (1 - t) * start_latent
        + t * end_latent
    )
    # Combine the latent vector (unchanged) with the interpolated image
    h_linear.append(latent)

x0 = torch.randn((1, 1, 256, 256), device="cuda")

results = []
for latent in h_linear:
    pred = fm.decoder.predict(x0.cuda(), h=latent, n_sampling_steps=50)
    results.append(pred.detach().cpu().numpy()[0, 0, :, :])
    
def compute_power_spectrum(image, std, mean):
    """
    Compute the power spectrum of a 2D image using the provided normalization.

    Args:
        image (ndarray): Input 2D image.
        std (float): Standard deviation for normalization.
        mean (float): Mean for normalization.

    Returns:
        tuple: Wavenumbers (k) and power spectrum (Pk).
    """
    y = image * std + mean
    delta_fields = y / np.mean(y) - 1
    Pk2D = PKL.Pk_plane(delta_fields, 25.0, "None", 1, verbose=False)
    return Pk2D.k, Pk2D.Pk

# Function to plot interpolation results with power spectrum
def plot_interpolated_images(start_image, end_image, results, num_samples, std, mean):
    """
    Plot the start and end images along with interpolated images and a cross dissolve comparison,
    including their power spectra.

    Args:
        start_image (ndarray): Ground truth start image.
        end_image (ndarray): Ground truth end image.
        results (list of ndarray): List of interpolated images.
        num_samples (int): Number of interpolation steps.
        std (float): Standard deviation for normalization.
        mean (float): Mean for normalization.
    """
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 15))  # 3 rows: images, cross dissolve, power spectrum

    # Compute power spectra for start and end images
    k_start, Pk_start = compute_power_spectrum(start_image, std, mean)
    k_end, Pk_end = compute_power_spectrum(end_image, std, mean)

    # # Plot start image (row 1: images)
    # axes[0, 0].imshow(start_image, cmap="viridis")
    # axes[0, 0].set_title("Start")
    # axes[0, 0].axis("off")
    # axes[1, 0].imshow(start_image, cmap="viridis")
    # axes[1, 0].set_title("Start")
    # axes[1, 0].axis("off")

    # # Plot start power spectrum (row 3: power spectrum)
    # axes[2, 0].loglog(k_start, Pk_start, label="Start")
    # axes[2, 0].set_title("Power Start")
    # axes[2, 0].legend()

    # # Plot end image (row 1: images)
    # axes[0, -1].imshow(end_image, cmap="viridis")
    # axes[0, -1].set_title("End")
    # axes[0, -1].axis("off")

    # axes[1, -1].imshow(end_image, cmap="viridis")
    # axes[1, -1].set_title("End")
    # axes[1, -1].axis("off")
    
    # # Plot end power spectrum (row 3: power spectrum)
    # axes[2, -1].loglog(k_end, Pk_end, label="End")
    # axes[2, -1].set_title("Power End")
    # axes[2, -1].legend()

    # Generate and plot cross dissolve images and power spectra
    for i, image in enumerate(results):
        t = i / (num_samples - 1)  # Interpolation factor
        cross_dissolve = (1 - t) * start_image + t * end_image
        k_cross, Pk_cross = compute_power_spectrum(cross_dissolve, std, mean)
        k_img, Pk_img = compute_power_spectrum(image, std, mean)
        
        print(f"PkCross min, max at step {i} is {Pk_cross.min()}, {Pk_cross.max()}")

        # Row 1: interpolated images
        axes[0, i].imshow(image, cmap="viridis")
        axes[0, i].set_title(f"Step {i+1}")
        axes[0, i].axis("off")

        # Row 2: cross dissolve
        axes[1, i].imshow(cross_dissolve, cmap="viridis")
        dissolve_title = None
        if i == 0:
            dissolve_title = "Start Image"
        elif i == num_samples - 1:
            dissolve_title = "End Image"
        else:
            dissolve_title = f"Dissolve {i+1}"
        axes[1, i].set_title(dissolve_title)
        axes[1, i].axis("off")

        # Row 3: Power spectrum
        axes[2, i].loglog(k_cross, Pk_cross, label="Cross Dissolve")
        axes[2, i].loglog(k_img, Pk_img, label="Interpolated Image")
        axes[2, i].loglog(k_end, Pk_end, label="End")
        axes[2, i].loglog(k_start, Pk_start, label="Start")
        axes[2, i].set_ylim(1e-7, 4e-2)
        axes[2, i].set_title(f"Power {i+1}")
        axes[2, i].legend()

    plt.tight_layout()
    plt.savefig('cosmo_compression/results/interpolation_with_power_spectrum.png')
    
plot_interpolated_images(y_orig, y_fin, results, num_samples, std, mean)