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

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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

fm = represent.Represent.load_from_checkpoint("64_hier/step=step=4600-val_loss=0.328.ckpt").to(device)
fm.eval()

<<<<<<< HEAD
param_est: nn.Module = inference.ParamMLP(input_dim=2304, hidden_widths=[1000,1000,256], output_dim=2).to(device)
param_est.load_state_dict(torch.load(param_est_path))

=======
>>>>>>> curriculum-learning
gts = []
Pk_orig = np.zeros(181)
Pk_fin = np.zeros(181)

img, cosmo = dataset[60]
img = torch.tensor(img).unsqueeze(0).cuda()
<<<<<<< HEAD
latent = fm.encoder(img).unsqueeze(0)

gts.append((cosmo, img*std+mean, latent))
=======

n_sampling_steps = 50
t = torch.linspace(0, 1, n_sampling_steps).cuda()
        
h = fm.encoder(img)

gts.append((cosmo, img * std + mean, h))
>>>>>>> curriculum-learning
y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, "None", 1, verbose=False)
k_orig = Pk2D.k
Pk_orig = Pk2D.Pk

img, cosmo = dataset[0]
img = torch.tensor(img).unsqueeze(0).cuda()

h = fm.encoder(img) 

gts.append((cosmo, img * std + mean, h))

<<<<<<< HEAD
gts.append((cosmo, img*std+mean, latent))
=======
>>>>>>> curriculum-learning
y = img.cpu().numpy().squeeze() * std + mean
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, "None", 1, verbose=False)
k_fin = Pk2D.k
Pk_fin = Pk2D.Pk

# Define latent vectors for gradient-based interpolation
h_grad = [gts[0][2][0]]
h_linear = []

starting_img = gts[0][2]
target_img = gts[-1][2]

<<<<<<< HEAD
# Linear interpolation: Create latent representations directly
num_samples = 20
h_linear = [
    starting_latent + t * (target_latent - starting_latent)
    for t in np.linspace(0, 1, num_samples)
]
=======
# Initialize a list for the interpolated latents
h_linear = []
# Define latent interpolation ranges and labels
modulation_ranges = {
    f"Stage 3" : range(48, 64),
    f"Stage 2" : range(32, 48),
    f"Stage 1" : range(16, 32),
    f"Stage 0" : range(0, 16),
}
>>>>>>> curriculum-learning

num_samples_per_stage = 5
all_interpolations = []
labels = []

current_image = starting_img.clone()

for label, specified_channels in modulation_ranges.items():
    # Perform interpolation for the specified channels
    for i in range(num_samples_per_stage):
        t = i / (num_samples_per_stage - 1)  # Interpolation factor
        interpolated_image = current_image.clone()  # Clone the current image
        interpolated_image[:, specified_channels] = (
            (1 - t) * starting_img[:, specified_channels]
            + t * target_img[:, specified_channels]
        )
        # Combine the latent vector (unchanged) with the interpolated image
        all_interpolations.append( interpolated_image)
        labels.append(label)

<<<<<<< HEAD
# Learning rate for updates
learning_rate = 0.01
# Number of steps
max_steps = 50000
# Gradient-based interpolation (already implemented)
x0 = torch.randn((1, 1,256,256), device='cuda')
for step in tqdm.tqdm(range(max_steps)):
    output = param_est(h_grad[-1].cuda())[:, 0].item()
    grad = input_grad(model=param_est, input=h_grad[-1].cuda(), output_index=0, other_output_inds=[1])
=======
    # Update current latent and image to the end of this stage
    current_image = interpolated_image.clone()

# Reverse interpolation
for label, specified_channels in modulation_ranges.items():
    for i in range(num_samples_per_stage):
        t = i / (num_samples_per_stage - 1)  # Interpolation factor
        interpolated_image = current_image.clone()  # Clone the current image
        interpolated_image[:, specified_channels] = (
            (1 - t) * target_img[:, specified_channels]
            + t * starting_img[:, specified_channels]
        )
        # Combine the latent vector (unchanged) with the interpolated image
        all_interpolations.append(interpolated_image)
        labels.append(f"Reverse: {label}")

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
>>>>>>> curriculum-learning
    
    # Average power spectra across all samples
    Pk_interpolations[i, :] = np.mean(Pk_samples, axis=0)
    
    # Use the first sample for visualization
    images_interpolations.append(preds[0] * std + mean)

def create_combined_animation_with_pauses(Pk_data, images, labels, filename, pause_frames=10):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    global_vmin = min(np.min(img) for img in images)
    global_vmax = max(np.max(img) for img in images)

<<<<<<< HEAD
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
def create_combined_animation_with_params(
    linear_Pk_data, grad_Pk_data, linear_images, grad_images, linear_params, grad_params, filename
):
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))

    # Calculate global vmin and vmax for images
    global_vmin = min(
        min(np.min(img) for img in linear_images + grad_images),
        torch.min(gts[0][1].cpu()),
        torch.min(gts[-1][1].cpu()),
    )
    global_vmax = max(
        max(np.max(img) for img in linear_images + grad_images),
        torch.max(gts[0][1].cpu()),
        torch.max(gts[-1][1].cpu()),
    )

    # Source Image Plot
    axs[0, 0].imshow(
        gts[0][1].cpu().squeeze(),
=======
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
>>>>>>> curriculum-learning
        cmap="viridis",
        origin="lower",
        vmin=global_vmin,
        vmax=global_vmax,
    )
<<<<<<< HEAD
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

    linear_power_ax = axs[0, 2]
    linear_power_ax.set_xscale("log")
    linear_power_ax.set_yscale("log")
    linear_power_ax.plot(k_orig, Pk_orig, label="Start Point", color="blue")
    linear_power_ax.plot(k_orig, Pk_fin, label="End Point", color="red")
    linear_power_line, = linear_power_ax.plot([], [], lw=2, color="green")
    linear_power_ax.set_title("Linear Interpolation: Power Spectrum")
    linear_power_ax.legend()

    linear_param_ax = axs[0, 3]
    linear_param_ax.set_title("Linear Interpolation: $\Omega_m$")
    linear_param_ax.plot(linear_params, label="$\Omega_m$")
    linear_param_ax.set_xlabel("Interpolation Step")
    linear_param_ax.set_ylabel("Parameter Value")

    # Initialize plots for gradient-based interpolation
    grad_img_ax = axs[1, 1]
    grad_img_plot = grad_img_ax.imshow(
        grad_images[0].squeeze(),
        cmap="viridis",
        origin="lower",
        vmin=global_vmin,
        vmax=global_vmax,
    )
    grad_img_ax.set_title("Gradient-Based Interpolation: Images")
    grad_img_ax.axis("off")

    grad_power_ax = axs[1, 2]
    grad_power_ax.set_xscale("log")
    grad_power_ax.set_yscale("log")
    grad_power_ax.plot(k_orig, Pk_orig, label="Start Point", color="blue")
    grad_power_ax.plot(k_orig, Pk_fin, label="End Point", color="red")
    grad_power_line, = grad_power_ax.plot([], [], lw=2, color="green")
    grad_power_ax.set_title("Gradient-Based Interpolation: Power Spectrum")
    grad_power_ax.legend()

    grad_param_ax = axs[1, 3]
    grad_param_ax.set_title("Gradient-Based Interpolation: $\Omega_m$")
    grad_param_ax.plot(grad_params, label="$\Omega_m$")
    grad_param_ax.plot(grad_params, label="$\sigma_8$")
    grad_param_ax.legend()
    grad_param_ax.set_xlabel("Interpolation Step")
    grad_param_ax.set_ylabel("Parameter Value")

    # Animation update function
    def update(frame):
        # Linear interpolation updates
        linear_img_plot.set_data(linear_images[frame].squeeze())
        linear_power_line.set_data(k_orig, linear_Pk_data[frame])

        # Gradient-based interpolation updates
        grad_img_plot.set_data(grad_images[frame].squeeze())
        grad_power_line.set_data(k_orig, grad_Pk_data[frame])

        return (
            linear_img_plot,
            linear_power_line,
            grad_img_plot,
            grad_power_line,
        )

    # Create animation
    num_frames = len(linear_images)
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)

    # Save animation as GIF
    ani.save(filename, writer=PillowWriter(fps=5))
    plt.close(fig)

# Compute parameter values for interpolation steps
print(param_est(h_linear[0].cuda()))
linear_params = np.array([param_est(h.cuda()).cpu().detach().numpy()[0,0] for h in h_linear])
grad_params = np.array([param_est(h.cuda()).cpu().detach().numpy()[0,0] for h in h_grad])

# Create the combined animation
create_combined_animation_with_params(
    Pk_linear, Pk_grad, images_linear, images_grad, linear_params, grad_params, "cosmo_compression/results/Combined.gif"
)
=======
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
>>>>>>> curriculum-learning
