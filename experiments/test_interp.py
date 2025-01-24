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

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

MAP_TYPE = "Mcdm"
MAP_RESOLUTION = 256

mean = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["mean"]
std = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["std"]


def input_grad(
    model: nn.Module,
    input: torch.Tensor,
    output_index: int,
    other_output_inds: List[int] | None = None,
):
    """
    Computes the gradient of the input with respect to a target output,
    and makes it orthogonal to the gradient w.r.t. a specified set of outputs

    Args:
        model: The parameter estimation model.
        x: The input tensor, requires gradient.
        target_output_index: The index of the target output.
        other_output_indices: Indices of the outputs to which the gradient
                                            should be orthogonal.

    Returns:
        torch.Tensor: The gradient of the input with respect to the target output, orthogonalized
                      to the gradients of other specified outputs.
    """
    # Ensure input requires gradient
    x = input.requires_grad_(True)
    outputs = model(x)

    if not other_output_inds:
        other_output_inds = [
            i for i in range(len(outputs.shape[-1])) if i != output_index
        ]

    # Compute gradient w.r.t the target output
    grad_target = torch.autograd.grad(outputs[:, output_index], x, retain_graph=True)[0]

    # Initialize orthogonal gradient as the target gradient
    grad_orthogonal = grad_target.clone()

    # Compute orthogonal component for each other output
    for idx in other_output_inds:
        grad_other = torch.autograd.grad(outputs[:, idx], x, retain_graph=True)[0]
        dot_product = torch.sum(grad_orthogonal * grad_other)
        norm_grad_other = torch.sum(grad_other**2)
        grad_orthogonal = (
            grad_orthogonal - (dot_product / (norm_grad_other + 1e-8)) * grad_other
        )

    return grad_orthogonal


param_est_path: str = "cosmo_compression/parameter_estimation/data/best_model_64ch.pth"
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

# Load models
fm: lightning.LightningModule = represent.Represent.load_from_checkpoint(
    "cosmo_segm_latent_64ch2wind/step=step=9200-val_loss=0.358.ckpt"
).to(device)
fm.eval()

param_est: nn.Module = inference.ParamMLP(
    input_dim=2304, hidden_widths=[1000, 1000, 256], output_dim=2
).to(device)
param_est.load_state_dict(torch.load(param_est_path))

gts = []
Pk_orig = np.zeros(181)
Pk_fin = np.zeros(181)

cosmo, img = dataset[0]
img = torch.tensor(img).unsqueeze(0).cuda()

latent = fm.encoder(img)

gts.append((cosmo, img * std + mean, latent))
y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, "None", 1, verbose=False)
k_orig = Pk2D.k
Pk_orig = Pk2D.Pk

cosmo, img = dataset[60]
img = torch.tensor(img).unsqueeze(0).cuda()

latent = fm.encoder(img)

gts.append((cosmo, img * std + mean, latent))

y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, "None", 1, verbose=False)
k_fin = Pk2D.k
Pk_fin = Pk2D.Pk

# Define latent vectors for gradient-based interpolation
h_grad = [gts[0][2][0]]
h_linear = []

starting_latent = gts[0][2]
starting_img = starting_latent[1]
target_latent = gts[-1][2]
target_img = target_latent[1]


# Linear interpolation: Create latent representations with custom rules
specified_channels = list(range(0, 8))  # Example channels
specified_lat_dims = list(range(0, 238))

num_samples = 10

# Initialize a list for the interpolated latents
h_linear = []

# Perform interpolation
for i in range(num_samples):
    t = i / (num_samples - 1)  # Interpolation factor
    interpolated_image = starting_img.clone()  # Clone the starting image
    interpolated_latent = starting_latent[0].clone()
    interpolated_latent[specified_lat_dims] = (
        (1 - t) * starting_latent[0][specified_lat_dims]
        + t * target_latent[0][ specified_lat_dims]
    )

    # Interpolate only the specified channels
    interpolated_image[:,specified_channels] = (
        (1 - t) * starting_img[:, specified_channels]
        + t * target_img[:, specified_channels]
    )

    # Combine the first latent vector (unchanged) with the interpolated image
    h_linear.append((interpolated_latent.unsqueeze(0), interpolated_image))

starting_omega_m = gts[0][0][0]
starting_sigma_8 = gts[0][0][1]

target_omega_m = gts[-1][0][0]
target_sigma_8 = gts[-1][0][1]

x0 = torch.randn((1, 1, 256, 256), device="cuda")
Pk_linear = np.zeros((num_samples, 181))

# images_grad = []
images_linear = []

# Generate maps, power spectra, and images
for i, latent_linear in tqdm.tqdm(enumerate(h_linear)):

    # Linear
    pred_linear = fm.decoder.predict(x0.cuda(), h=latent_linear, n_sampling_steps=50)
    pred_linear = pred_linear.cpu().numpy()[0, 0, :, :] * std + mean
    delta_linear = pred_linear / np.mean(pred_linear) - 1
    Pk2D_linear = PKL.Pk_plane(delta_linear, 25.0, "None", 1, verbose=False)
    Pk_linear[i, :] = Pk2D_linear.Pk
    images_linear.append(pred_linear)


def create_combined_animation_with_params(
    linear_Pk_data, linear_images, linear_params, filename
):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Calculate global vmin and vmax for images
    global_vmin = min(
        min(np.min(img) for img in linear_images),
        torch.min(gts[0][1].cpu()),
        torch.min(gts[-1][1].cpu()),
    )
    global_vmax = max(
        max(np.max(img) for img in linear_images),
        torch.max(gts[0][1].cpu()),
        torch.max(gts[-1][1].cpu()),
    )

    # Source Image Plot
    axs[0, 0].imshow(
        gts[0][1].cpu().squeeze(),
        cmap="viridis",
        origin="lower",
        vmin=global_vmin,
        vmax=global_vmax,
    )
    axs[0, 0].set_title("Source Image")
    axs[0, 0].axis("off")

    # Destination Image Plot
    axs[1, 0].imshow(
        gts[-1][1].cpu().squeeze(),
        cmap="viridis",
        origin="lower",
        vmin=global_vmin,
        vmax=global_vmax,
    )
    axs[1, 0].set_title("Destination Image")
    axs[1, 0].axis("off")

    # Initialize plots for linear interpolation
    linear_img_ax = axs[0, 1]
    linear_img_plot = linear_img_ax.imshow(
        linear_images[0].squeeze(),
        cmap="viridis",
        origin="lower",
        vmin=global_vmin,
        vmax=global_vmax,
    )
    linear_img_ax.set_title("Linear Interpolation: Images")
    linear_img_ax.axis("off")

    linear_power_ax = axs[1, 1]
    linear_power_ax.set_xscale("log")
    linear_power_ax.set_yscale("log")
    linear_power_ax.plot(k_orig, Pk_orig, label="Start Point", color="blue")
    linear_power_ax.plot(k_orig, Pk_fin, label="End Point", color="red")
    (linear_power_line,) = linear_power_ax.plot([], [], lw=2, color="green")
    linear_power_ax.set_title("Linear Interpolation: Power Spectrum")
    linear_power_ax.legend()

    # Animation update function
    def update(frame):
        # Adjust frame index to account for extended pauses at the start and end
        if frame < 5:  # Stay on the first frame for 5 frames
            current_frame = 0
        elif frame >= len(linear_images) + 5:  # Stay on the last frame for 5 frames
            current_frame = len(linear_images) - 1
        else:
            current_frame = frame - 5

        # Update linear interpolation plots
        linear_img_plot.set_data(linear_images[current_frame].squeeze())
        linear_power_line.set_data(k_orig, linear_Pk_data[current_frame])

        return (
            linear_img_plot,
            linear_power_line,
        )

    # Total frames include the extra 5 frames at the start and end
    num_frames = len(linear_images) + 10
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)

    # Save animation as GIF
    ani.save(filename, writer=PillowWriter(fps=5))
    plt.close(fig)


# Compute parameter values for interpolation steps
linear_params = np.array(
    [0 for h in h_linear]
)

# Create the combined animation
create_combined_animation_with_params(
    Pk_linear, images_linear, linear_params, "cosmo_compression/results/Combined.gif"
)
