from torch.utils.data import DataLoader

import Pk_library as PKL
import time
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

from cosmo_compression.data.data import CAMELS
from cosmo_compression.model.represent import Represent 

data = CAMELS(
        # idx_list=range(15000),
        map_type='Mcdm',
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
        h = fm.encoder(y.cuda()).cuda().unsqueeze(dim=0)
        gts.append((cosmology, y, h))
    elif (cosmology.cpu() == torch.Tensor([[0.3, 1.0, 1.00000, 1.00000, 1.00000, 1.00000, 0.04900]])).all() and len(gts) == 1:
        h = fm.encoder(y.cuda()).cuda().unsqueeze(dim=0)
        gts.append((cosmology, y, h))
        
def lerp(v1, v2, t):
    return (1 - t) * v1 + t * v2

h_s = []
num_samples = 20
averaging_steps = 20
for t in torch.linspace(0, 1, num_samples):
    h_s.append(lerp(gts[0][2].cpu(), gts[-1][2].cpu(), t))

Pk_test = np.zeros((num_samples, 181))
x0 = torch.randn((1,1,256,256)).cuda() 
for t in range(averaging_steps):
    for i, latent in enumerate(h_s):
        print(t, i)
        pred = fm.decoder.predict(
            x0.cuda(),
            h=latent.cuda(),
            n_sampling_steps=fm.hparams.n_sampling_steps,
        )
        
        # Convert tensors to numpy arrays for processing
        pred = pred.cpu().numpy()[0, 0, :, :]
                
        # Compute the power spectrum for the prediction
        delta_fields_pred = pred / np.mean(pred) - 1
        Pk2D = PKL.Pk_plane(delta_fields_pred, 25.0, 'None', 1, verbose=False)
        k_pred = Pk2D.k
        Pk_pred = Pk2D.Pk
        Pk_test[i, :] += Pk_pred

Pk_test = Pk_test / averaging_steps
fig, ax = plt.subplots()

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Power Spectra")
ax.set_xlabel("Wavenumber $k\,[h/Mpc]$")
ax.set_ylabel("$P(k)\,[(Mpc/h)^2]$")

  # Set x-axis range based on the number of columns
line, = ax.plot([], [], lw=2)

y = gts[0][1]
y = y.cpu().numpy()[0, 0, :, :]
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, 'None', 1, verbose=False)
k_orig = Pk2D.k
Pk_orig = Pk2D.Pk

ax.plot(k_orig, Pk_orig, label='Start Point')

y = gts[-1][1]
y = y.cpu().numpy()[0, 0, :, :]
delta_fields_orig_1 = y / np.mean(y) - 1
Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, 'None', 1, verbose=False)
k_orig = Pk2D.k
Pk_orig = Pk2D.Pk
ax.plot(k_orig, Pk_orig, label='End Point')
ax.set_xlim(k_orig.min(), k_orig.max())
ax.set_ylim(Pk_test.min()/10, max(Pk_test.max(), Pk_orig.max())*10)  # Set y-axis range based on data
ax.legend()

# Update function for the animation
def update(frame):
    line.set_data(k_orig, Pk_test[frame])  # x: indices, y: values
    ax.set_title(f"Interpolation step: {frame}")
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=range(Pk_test.shape[0]), blit=True)

# Save animation as a GIF
ani.save("cosmo_compression/results/Pk_test_animation.gif", writer=PillowWriter(fps=10))
plt.close(fig)