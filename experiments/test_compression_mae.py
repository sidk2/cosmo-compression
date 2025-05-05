from typing import List
import os
import time

import torch
import torch.utils
from torch.utils import data as torchdata

import Pk_library as PKL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torchvision.transforms as T


import lightning
from torch.utils.data import DataLoader

import lzma

from cosmo_compression.data import data
from cosmo_compression.model import represent

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device: str = "cuda" if torch.cuda.is_available() else "cpu"

dataset: torchdata.Dataset = data.CAMELS(
    idx_list=range(14_600, 15_000),
    map_type="Mcdm",
    dataset="LH",
    parameters=["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "Omega_b"],
)

data_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

batch_size = 32
mean = data.NORM_DICT["Mcdm"][256]["mean"]
std = data.NORM_DICT["Mcdm"][256]["std"]

# Load models
# ckpt_path = "/monolith/global_data/astro_compression/train_compression_model/lossless_latent_full_training/last.ckpt"
ckpt_path = "/monolith/global_data/astro_compression/train_compression_model/lmb_0.0001_full/step=step=6900-val_total_loss=0.330.ckpt"
fm = represent.Represent.load_from_checkpoint(ckpt_path).to(device)
fm.eval()

img, cosmo = dataset[0]
img = torch.tensor(img).unsqueeze(0).cuda()


n_sampling_steps = 30
t = torch.linspace(0, 1, n_sampling_steps).cuda()
        
h = fm.encoder(img)

# lossy compression of h
h_hat, h_likelihoods = fm.entropy_bottleneck(h) 
print(fm.entropy_bottleneck.training)
fm.entropy_bottleneck.update()
compressed_bytes = fm.entropy_bottleneck.compress(h)
bpp = len(compressed_bytes[0]) * 8 /(256*256)
print('lossy cai bpp: ', bpp)

# lossless compression of h
compress_input = h[0,:,:,:].detach().cpu().numpy()
print('compress_input shape: ', compress_input.shape)
compressed_bytes = lzma.compress(compress_input)

bpp = len(compressed_bytes) * 8 /(256*256)
print('lossless lzma bpp: ', bpp)


x0 = torch.randn((1, 1, 256, 256), device="cuda")

pred = fm.decoder.predict(x0.cuda(), h=h_hat, n_sampling_steps=n_sampling_steps)

# Pre-compute the original power spectrum
y = img.cpu().numpy()[0,0,:,:] * std + mean

denorm = img[0,:,:,:] * std + mean
low = T.GaussianBlur(11, 5)(denorm)
mid = denorm - low

print(std, mean, torch.mean(mid), torch.mean(pred), torch.std(mid), torch.std(pred))

delta_fields_orig_1 = denorm.cpu().numpy()[0, : , :] / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, "None", 1, verbose=False)
k_orig = Pk2D.k
Pk_orig = Pk2D.Pk

# pred = pred.detach().cpu().numpy()[0,0,:,:]*std+mean
print(pred.dtype)
pred = (pred).cpu().numpy()[0, 0, :, :] * std + mean
delta_fields_pred_1 = pred / np.mean(pred) - 1
Pk2D = PKL.Pk_plane(delta_fields_pred_1, 25.0, "None", 1, verbose=False)
k_pred = Pk2D.k
Pk_pred = Pk2D.Pk

# Initialize the figure and axes
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Original Image
ax1 = axs[0, 0]
img1 = ax1.imshow(y, cmap="viridis", origin="lower")
ax1.set_title("Original Image")
ax1.axis("off")
plt.colorbar(img1, ax=ax1, label="Density")

# Predicted Image
ax2 = axs[0, 1]
img2 = ax2.imshow(pred, cmap="viridis", origin="lower")
ax2.set_title("Reconstructed Image")
ax2.axis("off")
plt.colorbar(img2, ax=ax2, label="Density")

# # Power Spectra
ax3 = axs[1, 0]
line_orig, = ax3.plot(k_orig, Pk_orig, label="Original")
line_pred, = ax3.plot(k_pred, Pk_pred, label="Reconstructed")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_title("Power Spectra")
ax3.set_xlabel("Wavenumber $k\,[h/Mpc]$")
ax3.set_ylabel("$P(k)\,[(Mpc/h)^2]$")
ax3.legend()

# Difference Image
ax4 = axs[1, 1]
img3 = ax4.imshow(y-pred, cmap="seismic", origin="lower", vmin=-1, vmax=1)
ax4.set_title("Error Map")
ax4.axis("off")
plt.colorbar(img3, ax=ax4, label="Difference")

# Adjust layout
plt.tight_layout()


# compute image MSE
field_MSE = np.mean((y - pred) ** 2)
# compute spectrum MSE
spectrum_MSE = np.mean((Pk_orig - Pk_pred) ** 2)
log_spectrum_MSE = np.mean((np.log10(Pk_orig) - np.log10(Pk_pred)) ** 2)

print(f"field_MSE: {field_MSE:.3f}")
print(f"spectrum_MSE: {spectrum_MSE:.3f}")
print(f"log_spectrum_MSE: {log_spectrum_MSE:.3f}")

# Create the animation
# anim = FuncAnimation(fig, update_plot, frames=len(pred), blit=True)

# # Save or display the animation
plt.imsave('cosmo_compression/results/image.png', img.detach().cpu().numpy()[0,0,:,:], cmap='viridis')
plt.savefig("cosmo_compression/results/mae_decoded.png")
# # Alternatively, display in a Jupyter Notebook
# # from IPython.display import HTML
# # HTML(anim.to_jshtml())
