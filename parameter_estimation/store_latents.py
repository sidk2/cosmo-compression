import numpy as np

from torch.utils.data import DataLoader

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

from cosmo_compression.data.data import CAMELS
from cosmo_compression.model.represent import Represent 

import torch 

train_data = CAMELS(
        idx_list=range(15000),
        map_type='Mtot',
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2'],
    )
train_loader = DataLoader(
    train_data,
    batch_size=1,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

fm = Represent.load_from_checkpoint("soda-comp/step=step=3500-val_loss=0.445.ckpt").cuda()
fm.eval()

latents = []
params = []
i = 0
for cosmo, data in train_loader:
    with torch.no_grad():
        print(i)
        data = data.cuda()
        latent = fm.encoder(data)
        data = data.cpu()
        latent = latent.cpu()
        cosmo = cosmo.cpu()
        
        latents.append(latent)
        params.append(cosmo)
        
        torch.cuda.empty_cache()
        i+=1
latents = np.array(latents)
params = np.array(params)

np.save('cosmo_compression/parameter_estimation/data/latents.npy', latents)
np.save('cosmo_compression/parameter_estimation/data/params.npy', params)