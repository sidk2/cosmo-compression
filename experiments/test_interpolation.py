import os

import Pk_library as PKL
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from cosmo_compression.data.data import CAMELS
from cosmo_compression.model.represent import Represent 

data = CAMELS(
        # idx_list=range(15000),
        map_type='Mtot',
        dataset='1P',
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2','Omega_b'],
    )

loader = DataLoader(
    data,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

fm = Represent.load_from_checkpoint("soda-comp/step=step=3500-val_loss=0.445.ckpt").cuda()
fm.eval()

gts = []
for cosmology, y in loader:
    if (cosmology.cpu() == torch.Tensor([[0.3, 0.6, 1.00000, 1.00000, 1.00000, 1.00000, 0.04900]])).all() and len(gts) == 0:
        h = fm.encoder(y.cuda()).cuda().unsqueeze(0)
        gts.append((cosmology, y, h))
    elif (cosmology.cpu() == torch.Tensor([[0.3, 0.7, 1.00000, 1.00000, 1.00000, 1.00000, 0.04900]])).all() and len(gts) == 1:
        h = fm.encoder(y.cuda()).cuda().unsqueeze(0)
        gts.append((cosmology, y, h))
    elif (cosmology.cpu() == torch.Tensor([[0.3, 0.8, 1.00000, 1.00000, 1.00000, 1.00000, 0.04900]])).all() and len(gts) == 2:
        h = fm.encoder(y.cuda()).cuda().unsqueeze(0)
        gts.append((cosmology, y, h))
    elif (cosmology.cpu() == torch.Tensor([[0.3, 0.9, 1.00000, 1.00000, 1.00000, 1.00000, 0.04900]])).all() and len(gts) == 3:
        h = fm.encoder(y.cuda()).cuda().unsqueeze(0)
        gts.append((cosmology, y, h))
    elif (cosmology.cpu() == torch.Tensor([[0.3, 1.0, 1.00000, 1.00000, 1.00000, 1.00000, 0.04900]])).all() and len(gts) == 4:
        h = fm.encoder(y.cuda()).cuda().unsqueeze(0)
        gts.append((cosmology, y, h))
     
def distance_to_line(v1, v2, p):
    """
    Calculate the distance from a point to a line in n-dimensional space.
    
    Parameters:
        v1 (array-like): Endpoint 1 of the line.
        v2 (array-like): Endpoint 2 of the line.
        p (array-like): The point to measure the distance to the line.
        
    Returns:
        float: The distance from the point to the line.
    """
    v1, v2, p  = v1.reshape(-1).detach(), v2.reshape(-1).detach(), p.reshape(-1).detach()
    d = v2 - v1  # Vector along the line
    w = p - v1   # Vector from v1 to the point
    proj_d_w = (torch.dot(w, d) / torch.dot(d, d)) * d  # Projection of w onto d
    perpendicular = w - proj_d_w  # Perpendicular vector
    return torch.linalg.norm(perpendicular).item(), torch.linalg.norm(d).item()

for i in range(5):
    print(torch.mean(np.abs(gts[0][1]-gts[i][1])), distance_to_line(gts[0][2], gts[-1][2], gts[i][2]))


def lerp(v1, v2, t):
    return (1 - t) * v1 + t * v2

h_s = []
for t in torch.linspace(0, 1, 2):
    h_s.append(lerp(gts[0][2].cpu(), gts[-1][2].cpu(), t))


# Plot ground truth, prediction, and power spectra in the same figure
fig, axs = plt.subplots(3, 2, figsize=(10, 15))

num_trials = 1
Pk_avg = np.zeros((2, 181))

x0 = torch.randn((1,1,256,256)).cuda() 
for i, h in enumerate(h_s):
    y = gts[i][1].cuda()
    pred = fm.decoder.predict(
        x0.cuda(),
        h=h_s[i].cuda(),
        n_sampling_steps=fm.hparams.n_sampling_steps,
    )
    
    # Convert tensors to numpy arrays for processing
    y = y.cpu().numpy()[0, 0, :, :]
    pred = pred.cpu().numpy()[0, 0, :, :]
    
    # Compute the power spectrum for the ground truth
    delta_fields_orig = y / np.mean(y) - 1
    Pk2D = PKL.Pk_plane(delta_fields_orig, 25.0, 'None', 1, verbose=False)
    k_orig = Pk2D.k
    Pk_orig = Pk2D.Pk
            
    # Compute the power spectrum for the prediction
    delta_fields_pred = pred / np.mean(pred) - 1
    Pk2D = PKL.Pk_plane(delta_fields_pred, 25.0, 'None', 1, verbose=False)
    k_pred = Pk2D.k
    Pk_pred = Pk2D.Pk
    Pk_avg[i, :] += Pk_pred
    
    # Ground truth image
    axs[0, i].imshow(y, cmap='viridis')
    axs[0, i].set_title("Example Ground Truth")
    axs[0, i].axis("off")

    title = "Interpolated" if (i != 0 and i != 4) else "Decompressed"
    # Predicted image
    axs[1, i].imshow(pred, cmap='viridis')
    axs[1, i].set_title(title)
    axs[1, i].axis("off")

    # Power spectrum plot
    axs[2, i].plot(k_orig, Pk_orig, label='Original')
    axs[2, i].plot(k_pred, Pk_pred, label='Prediction')
    axs[2, i].set_xscale("log")
    axs[2, i].set_yscale("log")
    axs[2, i].set_title("Power Spectra")
    axs[2, i].set_xlabel("Wavenumber $k\,[h/Mpc]$")
    axs[2, i].set_ylabel("$P(k)\,[(Mpc/h)^2]$")

# for i in range(5):
#     axs[2, i].plot(k_pred, Pk_avg[i, :]/num_trials, label='Average')
#     axs[2, i].legend()

# Save the combined figure
plt.tight_layout()
plt.savefig("cosmo_compression/results/interpolation.png")
plt.close()
