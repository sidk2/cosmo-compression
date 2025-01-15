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

param_est_path: str = "cosmo_compression/parameter_estimation/data/best_model_64ch.pth"
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
fm: lightning.LightningModule = represent.Represent.load_from_checkpoint("img-lat-64ch/step=step=44500-val_loss=0.282.ckpt").to(device)
fm.eval()

param_est: nn.Module = inference.ParamMLP(input_dim=2304, hidden_widths=[1000,1000,256], output_dim=2).to(device)
param_est.load_state_dict(torch.load(param_est_path))

gts = []
Pk_orig = np.zeros(181)
Pk_fin = np.zeros(181)

cosmo, img = dataset[0]
img = torch.tensor(img).unsqueeze(0).cuda()

latent = fm.encoder(img)
foo, bar = latent
foo = foo.unsqueeze(0)
latent = foo, bar

gts.append((cosmo, img*std+mean, latent))
y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, 'None', 1, verbose=False)
k_orig = Pk2D.k
Pk_orig = Pk2D.Pk

cosmo, img = dataset[60]
img = torch.tensor(img).unsqueeze(0).cuda()

latent = fm.encoder(img)
foo, bar = latent
foo = foo.unsqueeze(0)
latent = foo, bar

gts.append((cosmo, img*std+mean, latent))
y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, 'None', 1, verbose=False)
k_fin = Pk2D.k
Pk_fin = Pk2D.Pk

# Define latent vectors for gradient-based interpolation
h_grad = [gts[0][2][0]]
h_linear = []

starting_latent = (gts[0][2][0], gts[0][2][1])
target_latent = (gts[-1][2][0], gts[-1][2][1])

# Linear interpolation: Create latent representations with custom rules
num_samples = 20
h_linear = []

# h_linear = [
#     tuple(
#         start + t * (end - start)
#         for start, end in zip(starting_latent, target_latent)
#     )
#     for t in np.linspace(0, 1, num_samples)
# ]

for t in np.linspace(0, 1, num_samples):
    interpolated_vector = (
        starting_latent[0][:256] + t * (target_latent[0][:256] - starting_latent[0][:256]),
        starting_latent[0][-256:] + t * (target_latent[0][-256:] - starting_latent[0][-256:]),
    )
    interpolated_vector = torch.cat([
        interpolated_vector[0][:, :256].reshape(-1),
        starting_latent[0][:, 256:-256].reshape(-1),
        interpolated_vector[1][:, -256:].reshape(-1)
    ])
    interpolated_image = starting_latent[1] + t * (target_latent[1] - starting_latent[1])
    h_linear.append((interpolated_vector.unsqueeze(0), interpolated_image))

starting_omega_m = gts[0][0][0]
starting_sigma_8 = gts[0][0][1]

target_omega_m = gts[-1][0][0]
target_sigma_8 = gts[-1][0][1]

# print(f"Interpolation between {starting_omega_m, starting_sigma_8} and {target_omega_m, target_sigma_8}")

# # Learning rate for updates
# learning_rate = 0.01
# # Number of steps
# max_steps = 50000
# # Gradient-based interpolation (already implemented)
x0 = torch.randn((1, 1,256,256), device='cuda')
# for step in tqdm.tqdm(range(max_steps)):
#     output = param_est(h_grad[-1].cuda())[:, 0].item()
#     grad = input_grad(model=param_est, input=h_grad[-1].cuda(), output_index=0, other_output_inds=[1])
    
#     if output > target_omega_m:
#         h_new = h_grad[-1] - learning_rate * grad
#     else:
#         h_new = h_grad[-1] + learning_rate * grad
    
#     outputs = param_est(h_new.cuda())
#     target_output = outputs[:, 0].item()
#     h_grad.append(h_new)
    
#     if abs(target_output - target_omega_m) < 0.001:
#         break

# h_grad = h_grad[::len(h_grad) // num_samples] + [h_grad[-1]]
# Pk_grad = np.zeros((num_samples, 181))
Pk_linear = np.zeros((num_samples, 181))

# images_grad = []
images_linear = []

# Generate maps, power spectra, and images
for i, latent_linear in tqdm.tqdm(enumerate(h_linear)):
    # Gradient-based
    # pred_grad = fm.decoder.predict(x0.cuda(), h=(latent_grad.cuda(), gts[0][2][1]), n_sampling_steps=50)
    # pred_grad = pred_grad.cpu().numpy()[0, 0, :, :] * std + mean
    # delta_grad = pred_grad / np.mean(pred_grad) - 1
    # Pk2D_grad = PKL.Pk_plane(delta_grad, 25.0, 'None', 1, verbose=False)
    # Pk_grad[i, :] = Pk2D_grad.Pk
    # images_grad.append(pred_grad)
    
    # Linear
    pred_linear = fm.decoder.predict(x0.cuda(), h=latent_linear, n_sampling_steps=50)
    pred_linear = pred_linear.cpu().numpy()[0, 0, :, :] * std + mean
    delta_linear = pred_linear / np.mean(pred_linear) - 1
    Pk2D_linear = PKL.Pk_plane(delta_linear, 25.0, 'None', 1, verbose=False)
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
    linear_power_line, = linear_power_ax.plot([], [], lw=2, color="green")
    linear_power_ax.set_title("Linear Interpolation: Power Spectrum")
    linear_power_ax.legend()

    # linear_param_ax = axs[1, 2]
    # linear_param_ax.set_title("Linear Interpolation: Parameter Values")
    # linear_param_ax.plot(linear_params[:, 0], label="$\Omega_m$")
    # linear_param_ax.plot(linear_params[:, 1], label="$\sigma_8$")
    # linear_param_ax.legend()

    # linear_param_ax.set_xlabel("Interpolation Step")
    # linear_param_ax.set_ylabel("Parameter Value")

    # # Initialize plots for gradient-based interpolation
    # grad_img_ax = axs[1, 1]x
    # grad_img_plot = grad_img_ax.imshow(
    #     grad_images[0].squeeze(),
    #     cmap="viridis",
    #     origin="lower",
    #     vmin=global_vmin,
    #     vmax=global_vmax,
    # )
    # grad_img_ax.set_title("Gradient-Based Interpolation: Images")
    # grad_img_ax.axis("off")

    # grad_power_ax = axs[1, 2]
    # grad_power_ax.set_xscale("log")
    # grad_power_ax.set_yscale("log")
    # grad_power_ax.plot(k_orig, Pk_orig, label="Start Point", color="blue")
    # grad_power_ax.plot(k_orig, Pk_fin, label="End Point", color="red")
    # grad_power_line, = grad_power_ax.plot([], [], lw=2, color="green")
    # grad_power_ax.set_title("Gradient-Based Interpolation: Power Spectrum")
    # grad_power_ax.legend()

    # grad_param_ax = axs[1, 3]
    # grad_param_ax.set_title("Gradient-Based Interpolation: Parameter Values")
    # grad_param_ax.plot(grad_params[:,0], label="$\Omega_m$")
    # grad_param_ax.plot(grad_params[:,1], label="$\sigma_8$")
    # grad_param_ax.legend()
    # grad_param_ax.set_xlabel("Interpolation Step")
    # grad_param_ax.set_ylabel("Parameter Value")

    # Animation update function
    def update(frame):
        # Linear interpolation updates
        linear_img_plot.set_data(linear_images[frame].squeeze())
        linear_power_line.set_data(k_orig, linear_Pk_data[frame])

        # Gradient-based interpolation updates
        # grad_img_plot.set_data(grad_images[frame].squeeze())
        # grad_power_line.set_data(k_orig, grad_Pk_data[frame])

        return (
            linear_img_plot,
            linear_power_line,
            # grad_img_plot,
            # grad_power_line,
        )

    # Create animation
    num_frames = len(linear_images)
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)

    # Save animation as GIF
    ani.save(filename, writer=PillowWriter(fps=5))
    plt.close(fig)

# Compute parameter values for interpolation steps
linear_params = np.array([param_est(h[0].cuda()).cpu().detach().numpy()[0] for h in h_linear])
# grad_params = np.array([param_est(h.cuda()).cpu().detach().numpy()[0] for h in h_grad])

# Create the combined animation
create_combined_animation_with_params(
    Pk_linear, images_linear, linear_params, "cosmo_compression/results/Combined.gif"
)