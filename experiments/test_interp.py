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

os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

MAP_TYPE = 'Mcdm'
MAP_RESOLUTION = 256

mean = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["mean"]
std = data.NORM_DICT[MAP_TYPE][MAP_RESOLUTION]["std"]

def input_grad(model: nn.Module, 
               input: torch.Tensor, 
               output_index: int, 
               other_output_inds: List[int] | None = None):
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
        other_output_inds = [i for i in range(len(outputs.shape[-1])) if i != output_index]

    # Compute gradient w.r.t the target output
    grad_target = torch.autograd.grad(outputs[:, output_index], x, retain_graph=True)[0]

    # Initialize orthogonal gradient as the target gradient
    grad_orthogonal = grad_target.clone()

    # Compute orthogonal component for each other output
    for idx in other_output_inds:
        grad_other = torch.autograd.grad(outputs[:, idx], x, retain_graph=True)[0]
        dot_product = torch.sum(grad_orthogonal * grad_other)
        norm_grad_other = torch.sum(grad_other ** 2)
        grad_orthogonal = grad_orthogonal - (dot_product / (norm_grad_other + 1e-8)) * grad_other

    return grad_orthogonal

param_est_path: str = "cosmo_compression/parameter_estimation/data/best_model.pth"
device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset: torchdata.Dataset = data.CAMELS(
        map_type=MAP_TYPE,
        dataset='1P',
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2','Omega_b'],
    )

loader = torchdata.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

# Load models
fm: lightning.LightningModule = represent.Represent.load_from_checkpoint("soda-comp/step=step=3500-val_loss=0.445.ckpt").to(device)
fm.eval()

param_est: nn.Module = inference.ParamMLP(input_dim=2304, hidden_widths=[1000,1000,256], output_dim=2).to(device)
param_est.load_state_dict(torch.load(param_est_path))

gts = []
Pk_orig = np.zeros(181)
Pk_fin = np.zeros(181)

cosmo, img = dataset[0]
img = torch.tensor(img).unsqueeze(0).cuda()

latent = fm.encoder(img).unsqueeze(0)

gts.append((cosmo, img, latent))
y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, 'None', 1, verbose=False)
k_orig = Pk2D.k
Pk_orig = Pk2D.Pk

cosmo, img = dataset[60]
img = torch.tensor(img).unsqueeze(0).cuda()

latent = fm.encoder(img).unsqueeze(0)

gts.append((cosmo, img, latent))
y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, 'None', 1, verbose=False)
k_fin = Pk2D.k
Pk_fin = Pk2D.Pk

# Define latent vectors for gradient-based interpolation
h_grad = [gts[0][2]]
h_linear = []

# Define starting and target points
starting_latent = gts[0][2]
target_latent = gts[-1][2]

# Linear interpolation: Create latent representations directly
num_samples = 20
h_linear = [
    starting_latent + t * (target_latent - starting_latent)
    for t in np.linspace(0, 1, num_samples)
]

starting_omega_m = gts[0][0][0]
starting_sigma_8 = gts[0][0][1]

target_omega_m = gts[-1][0][0]
target_sigma_8 = gts[-1][0][1]

print(f"Interpolation between {starting_omega_m, starting_sigma_8} and {target_omega_m, target_sigma_8}")

# Learning rate for updates
learning_rate = 0.01
# Number of steps
max_steps = 50000
# Gradient-based interpolation (already implemented)
x0 = torch.randn((1, 1,256,256), device='cuda')
for step in tqdm.tqdm(range(max_steps)):
    output = param_est(h_grad[-1].cuda())[:, 0].item()
    grad = input_grad(model=param_est, input=h_grad[-1].cuda(), output_index=0, other_output_inds=[1])
    
    if output > target_omega_m:
        h_new = h_grad[-1] - learning_rate * grad
    else:
        h_new = h_grad[-1] + learning_rate * grad
    
    outputs = param_est(h_new.cuda())
    target_output = outputs[:, 0].item()
    h_grad.append(h_new)
    
    if abs(target_output - target_omega_m) < 0.001:
        break

h_grad = h_grad[::len(h_grad) // num_samples] + [h_grad[-1]]
Pk_grad = np.zeros((num_samples, 181))
Pk_linear = np.zeros((num_samples, 181))

images_grad = []
images_linear = []

# Generate maps, power spectra, and images
for i, (latent_grad, latent_linear) in tqdm.tqdm(enumerate(zip(h_grad, h_linear))):
    # Gradient-based
    pred_grad = fm.decoder.predict(x0.cuda(), h=latent_grad.cuda(), n_sampling_steps=50)
    pred_grad = pred_grad.cpu().numpy()[0, 0, :, :] * std + mean
    delta_grad = pred_grad / np.mean(pred_grad) - 1
    Pk2D_grad = PKL.Pk_plane(delta_grad, 25.0, 'None', 1, verbose=False)
    Pk_grad[i, :] = Pk2D_grad.Pk
    images_grad.append(pred_grad)
    
    # Linear
    pred_linear = fm.decoder.predict(x0.cuda(), h=latent_linear.cuda(), n_sampling_steps=50)
    pred_linear = pred_linear.cpu().numpy()[0, 0, :, :] * std + mean
    delta_linear = pred_linear / np.mean(pred_linear) - 1
    Pk2D_linear = PKL.Pk_plane(delta_linear, 25.0, 'None', 1, verbose=False)
    Pk_linear[i, :] = Pk2D_linear.Pk
    images_linear.append(pred_linear)

def create_combined_animation(title, Pk_data, images, filename):
    fig, axs = plt.subplots(1, 2, figsize=(24, 12))

    # Power Spectrum Plot
    power_ax = axs[0]
    power_ax.set_xscale("log")
    power_ax.set_yscale("log")
    power_ax.set_title(f"{title}: Power Spectrum")
    power_ax.set_xlabel("Wavenumber $k\,[h/Mpc]$")
    power_ax.set_ylabel("$P(k)\,[(Mpc/h)^2]$")
    power_ax.plot(k_orig, Pk_orig, label="Start Point", color="blue")
    power_ax.plot(k_orig, Pk_fin, label="End Point", color="red")
    power_ax.legend()
    power_ax.set_xlim(k_orig.min(), k_orig.max())
    power_ax.set_ylim(Pk_data.min(), Pk_data.max())

    # Line for power spectrum animation
    power_line, = power_ax.plot([], [], lw=2, color="green")

    # Image Plot
    img_ax = axs[1]
    img_ax.set_title(f"{title}: Images")
    img_ax.axis("off")
    img_plot = img_ax.imshow(images[0], cmap="viridis", origin="lower")

    # Update function for animation
    def update(frame):
        # Update power spectrum
        power_line.set_data(k_orig, Pk_data[frame])
        power_ax.set_title(f"{title}: Power Spectrum - Step {frame}")

        # Update image
        img_plot.set_data(images[frame])
        img_ax.set_title(f"{title}: Image - Step {frame}")

        return [power_line, img_plot]

    # Create animation
    ani = FuncAnimation(fig, update, frames=range(num_samples), blit=True)

    # Save animation as a GIF
    ani.save(filename, writer=PillowWriter(fps=10))
    plt.close(fig)

# Create combined animations
create_combined_animation(
    "Gradient-Based Interpolation", 
    Pk_grad, 
    images_grad, 
    "cosmo_compression/results/Gradient_Combined.gif"
)
create_combined_animation(
    "Linear Interpolation", 
    Pk_linear, 
    images_linear, 
    "cosmo_compression/results/Linear_Combined.gif"
)