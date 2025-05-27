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

torch.manual_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MAP_TYPE = "Mcdm"
MAP_RESOLUTION = 256

mean = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["mean"]
std = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["std"]

device: str = "cuda" if torch.cuda.is_available() else "cpu"

dataset: torchdata.Dataset = data.CAMELS(
    map_type=MAP_TYPE,
    dataset="1P",
    parameters=["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "Omega_b"],
)

loader = torchdata.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

fm = represent.Represent.load_from_checkpoint("16x16_diti_setup_non_hier/16x16_diti_no_latent_splitting/step=step=21800-val_loss=0.335.ckpt").cuda()
fm.eval()

gts = []
Pk_orig = np.zeros(181)
Pk_fin = np.zeros(181)

img, cosmo = dataset[0]
img = torch.tensor(img).unsqueeze(0).cuda()

n_sampling_steps = 20
        
h = fm.encoder(img)

gts.append((cosmo, img * std + mean, h))
y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, "None", 1, verbose=False)
k_orig = Pk2D.k
Pk_orig = Pk2D.Pk

img, cosmo = dataset[60]
img = torch.tensor(img).unsqueeze(0).cuda()

h = fm.encoder(img) 

gts.append((cosmo, img * std + mean, h))

y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, "None", 1, verbose=False)
k_fin = Pk2D.k
Pk_fin = Pk2D.Pk

# Define latent vectors for gradient-based interpolation
h_linear = []

starting_img = gts[0][2]
target_img = gts[-1][2]

# Initialize a list for the interpolated latents
h_linear = []
# Define latent interpolation ranges and labels
modulation_ranges = {
    "Stage 0" : list(range(0, 4)),
    "Stage 1" : list(range(4, 8)),
    "Stage 2" : list(range(8, 12)),
    "Stage 3" : list(range(12, 16)),
}

num_samples_per_stage = 10
all_interpolations = []
labels = []

current_image = starting_img.clone()

for stage, (label, specified_channels) in enumerate(modulation_ranges.items()):
    # Perform interpolation for the specified channels
    for i in range(num_samples_per_stage):
        t = i / (num_samples_per_stage - 1)  # Interpolation factor
        interpolated_image = current_image.clone()  # Clone the current image
        interpolated_image[:, specified_channels] = (
            (1 - t) * starting_img[:, specified_channels]
            + t * target_img[:, specified_channels]
        )
        
        # Combine the latent vector (unchanged) with the interpolated image
        all_interpolations.append(interpolated_image)
        labels.append(label)

    # Update current latent and image to the end of this stage
    current_image = interpolated_image.clone()

# Generate multiple x0s and use the same set for all steps
num_samples = 1
x0_samples = [torch.randn((1, 1, 256, 256), device="cuda") for _ in range(num_samples)]
Pk_interpolations = np.zeros((len(all_interpolations), 181))
images_interpolations = []

for i, latent_interpolation in tqdm.tqdm(enumerate(all_interpolations)):
    preds = []
    Pk_samples = []  # To store power spectra for each x0

    for x0 in x0_samples:
        pred = fm.decoder.predict(x0.cuda(), h=latent_interpolation, n_sampling_steps=n_sampling_steps)

        preds.append(pred.cpu().numpy()[0, 0, :, :])
        
        # Compute power spectrum for this sample
        sample_pred = pred.cpu().numpy()[0, 0, :, :] * std + mean
        delta = sample_pred / np.mean(sample_pred) - 1
        Pk2D = PKL.Pk_plane(delta, 25.0, "None", 1, verbose=False)
        Pk_samples.append(Pk2D.Pk)
    
    # Average power spectra across all samples
    Pk_interpolations[i, :] = np.mean(Pk_samples, axis=0)
    
    # Use the first sample for visualization
    images_interpolations.append(preds[0] * std + mean)
    if i == (len(all_interpolations) - 1):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].imshow(gts[-1][1].cpu().numpy()[0, 0, :, :], cmap='viridis')
        ax[0].set_title("Source Image")
        ax[1].imshow(preds[0], cmap='viridis')
        ax[1].set_title("Reconstruction")
        plt.savefig(f"cosmo_compression/results/first_interp.png")
        plt.close()

def create_combined_animation_with_pauses(Pk_data, images, labels, filename, pause_frames=5):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    global_vmin = min(np.min(img) for img in images)
    global_vmax = max(np.max(img) for img in images)

    axs[0, 0].imshow(
        gts[0][1].cpu().squeeze(),
        cmap="viridis",
        origin="lower",
        vmin=global_vmin,
        vmax=global_vmax,
    )
    axs[0, 0].set_title("Source Image")
    axs[0, 0].axis("off")

    axs[1, 0].imshow(
        gts[-1][1].cpu().squeeze(),
        cmap="viridis",
        origin="lower",
        vmin=global_vmin,
        vmax=global_vmax,
    )
    axs[1, 0].set_title("Destination Image")
    axs[1, 0].axis("off")

    linear_img_ax = axs[0, 1]
    linear_img_plot = linear_img_ax.imshow(
        images[0].squeeze(),
        cmap="viridis",
        origin="lower",
        vmin=global_vmin,
        vmax=global_vmax,
    )
    linear_img_ax.set_title("Interpolation: Images")
    linear_img_ax.axis("off")

    linear_power_ax = axs[1, 1]
    linear_power_ax.set_xscale("log")
    linear_power_ax.set_yscale("log")
    linear_power_ax.plot(k_orig, Pk_orig, label="Start Point", color="blue")
    linear_power_ax.plot(k_orig, Pk_fin, label="End Point", color="red")
    (linear_power_line,) = linear_power_ax.plot([], [], lw=2, color="green")
    linear_power_ax.set_title("Interpolation: Power Spectrum")
    linear_power_ax.legend()

    # Extend frames to include pauses
    extended_frames = []
    extended_labels = []
    for i, (img, label) in enumerate(zip(images, labels)):
        extended_frames.append(i)
        extended_labels.append(label)
        if i == len(images) - 1 or labels[i] != labels[i + 1]:
            # Repeat the last frame of a phase
            extended_frames.extend([i] * pause_frames)
            extended_labels.extend([label] * pause_frames)

    def update(frame):
        current_frame = extended_frames[frame]
        linear_img_plot.set_data(images[current_frame].squeeze())
        linear_power_line.set_data(k_orig, Pk_data[current_frame])
        axs[0, 1].set_title(extended_labels[frame])

        return (
            linear_img_plot,
            linear_power_line,
        )

    ani = FuncAnimation(fig, update, frames=len(extended_frames), blit=True)
    ani.save(filename, writer=PillowWriter(fps=5))
    plt.close(fig)

# Create the combined animation with pauses after each modulation phase
create_combined_animation_with_pauses(
    Pk_interpolations,
    images_interpolations,
    labels,
    "cosmo_compression/results/Combined_Modulations.gif",
)
