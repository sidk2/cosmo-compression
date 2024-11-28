from pathlib import Path
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning import Trainer
from lightning.pytorch import seed_everything

import Pk_library as PKL
import wandb
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 


from .data.data import CAMELS
from .model.represent import Represent 

data = CAMELS(
        idx_list=range(15000),
        map_type='T',
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2'],
    )
loader = DataLoader(
    data,
    batch_size=1,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

# fm = Represent.load_from_checkpoint("results/noada-compression/step=step=6000-val_loss=0.479.ckpt").cuda()
fm.eval()

for cosmology, y in loader:
    h = fm.encoder(y.cuda()).cuda()
    x0 = torch.randn_like(y.cuda()).cuda() 
    pred = fm.decoder.predict(
        x0.cuda(),
        h=h,
        n_sampling_steps=fm.hparams.n_sampling_steps,
        )
    y = y.cpu().numpy()[0,0,:,:]
    delta_fields_orig = y / np.mean(y) - 1
    Pk2D = PKL.Pk_plane(delta_fields_orig, 25.0, 'None', 1, verbose=False)
    k_orig      = Pk2D.k      #k in h/Mpc
    Pk_orig     = Pk2D.Pk
    
    pred = pred.cpu().numpy()[0,0,:,:]
    delta_fields_pred = pred / np.mean(pred) - 1
    Pk2D = PKL.Pk_plane(delta_fields_pred, 25.0, 'None', 1, verbose=False)
    k_pred      = Pk2D.k      #k in h/Mpc
    Pk_pred     = Pk2D.Pk
    
    
    plt.figure()
    plt.plot(k_orig, Pk_orig, label='original')
    plt.plot(k_pred, Pk_pred, label='pred')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Power Spectra of Images")
    plt.legend()
    plt.xlabel("Wavenumber $k\,[h/Mpc]$")
    plt.ylabel("$P(k)\,[(Mpc/h)^2]$")
    plt.savefig("pks.png")
    plt.close()
    break