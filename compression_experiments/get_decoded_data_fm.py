import os
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import json
import argparse
from compressai.zoo import image_models
from compressai.utils.eval_model import __main__ as cai_eval
from compressai.datasets.ndarray import NdArrayDataset
from collections import defaultdict
from timm.utils import AverageMeter
import numpy as np
import matplotlib.pyplot as plt
import Pk_library as PKL
import sys
from pathlib import Path

from cosmo_compression.data import data
from cosmo_compression.model import represent


device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = NdArrayDataset(np.load("/monolith/global_data/astro_compression/CAMELS/images/test/test_data.npy"), single=True)
lmbs = ["1.5", "1", "0.5", "0.1", "0.01", "0.005", "0.001", "0.0001"]
ckpt_root = "/monolith/global_data/astro_compression/train_compression_model"
ckpt_dict = {"1.5":Path(ckpt_root) /  "lmb_1.5_full" / 'step=step=6900-val_total_loss=0.514.ckpt',
             "1":Path(ckpt_root) /  "lmb_1_full" / 'step=step=6900-val_total_loss=0.488.ckpt',
             "0.5":Path(ckpt_root) /  "lmb_0.5_full" / 'step=step=6900-val_total_loss=0.448.ckpt',
             "0.1":Path(ckpt_root) /  "lmb_0.1_full" / 'step=step=6100-val_total_loss=0.383.ckpt',
             "0.01":Path(ckpt_root) /  "lmb_0.01_full" / 'step=step=7700-val_total_loss=0.330.ckpt',
             "0.005":Path(ckpt_root) /  "lmb_0.005_full" / 'step=step=6100-val_total_loss=0.332.ckpt',
             "0.001":Path(ckpt_root) /  "lmb_0.001_full" / 'step=step=6300-val_total_loss=0.339.ckpt',
             "0.0001":Path(ckpt_root) /  "lmb_0.0001_full" / 'step=step=6900-val_total_loss=0.330.ckpt',
             }
ckpt_root = "/monolith/global_data/astro_compression/train_compression_model"

for lmb in lmbs:
    ckpt_path = ckpt_dict[lmb]
    fm = represent.Represent.load_from_checkpoint(ckpt_path).to(device)
    fm.eval()
    fm.entropy_bottleneck.update(force=True, update_quantiles=True)
    output_dataset = np.zeros((30, 1, 256, 256), dtype=np.float32)
    for image_index in tqdm(range(30)):
        input_array = test_dataset[image_index, :, :, :]
        img = torch.from_numpy(input_array).unsqueeze(0).cuda()
        n_sampling_steps = 30
        t = torch.linspace(0, 1, n_sampling_steps).cuda()
                
        h = fm.encoder(img)
        h_hat, h_likelihoods = fm.entropy_bottleneck(h) 

        x0 = torch.randn((1, 1, 256, 256), device="cuda")

        pred = fm.decoder.predict(x0.cuda(), h=h_hat, n_sampling_steps=n_sampling_steps)
    np.save(f'/monolith/global_data/astro_compression/fm_output/{lmb}_output_dataset.npy', output_dataset)
    

    mean = data.NORM_DICT["Mcdm"][256]["mean"]
    std = data.NORM_DICT["Mcdm"][256]["std"]

    y = img.cpu().numpy()[0,0,:,:] * std + mean
    pred = (pred).cpu().numpy()[0, 0, :, :] * std + mean

    # Initialize the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Original Image
    ax1 = axs[0]
    img1 = ax1.imshow(y, cmap="viridis", origin="lower")
    ax1.set_title("Original Image")
    ax1.axis("off")
    plt.colorbar(img1, ax=ax1, label="Density")

    # Predicted Image
    ax2 = axs[1]
    img2 = ax2.imshow(pred, cmap="viridis", origin="lower")
    ax2.set_title("Reconstructed Image")
    ax2.axis("off")
    plt.colorbar(img2, ax=ax2, label="Density")

    plt.savefig(f"cosmo_compression/compression_experiments/check_mae{lmb}.png")