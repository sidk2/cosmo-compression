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
import matplotlib.pyplot 
from pathlib import Path
import Pk_library as PKL
import pickle

from cosmo_compression.model import represent
from cosmo_compression.data import data

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

def get_bpps(model, image_tensor):
    h = model.encoder(image_tensor.unsqueeze(0))
    strings = model.entropy_bottleneck.compress(h)
    compress_bpp = len(strings[0]) * 8 / (256 * 256)
    stats = {
        'compress_bpp': float(compress_bpp)
    }
    return stats


def get_single_image_stats(model, input_array, output_array):
    # input array is [1, 256, 256]
    input_tensor = torch.from_numpy(input_array).cuda()
    bpp_stats = get_bpps(model, input_tensor)

    mean = data.NORM_DICT["Mcdm"][256]["mean"]
    std = data.NORM_DICT["Mcdm"][256]["std"]

    input_denorm = input_array * std + mean
    output_denorm = output_array * std + mean

    delta_fields_orig_1 = input_denorm[0, : , :] / np.mean(input_array) - 1
    Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, "None", 1, verbose=False)
    k_orig = Pk2D.k
    Pk_orig = Pk2D.Pk

    delta_fields_pred_1 = output_denorm[0, : , :] / np.mean(input_array) - 1
    Pk2D = PKL.Pk_plane(delta_fields_pred_1, 25.0, "None", 1, verbose=False)
    
    k_pred = Pk2D.k
    Pk_pred = Pk2D.Pk

    spectrum_MSE = np.mean((Pk_orig - Pk_pred) ** 2)
    log_spectrum_MSE = np.mean((np.log10(Pk_orig) - np.log10(Pk_pred)) ** 2)

    field_MSE = np.mean((input_array[0,:,:] - output_array[:,:]) ** 2)

    return {"compress_bpp":bpp_stats["compress_bpp"],
            "spectrum_MSE": spectrum_MSE,
            "log_spectrum_MSE": log_spectrum_MSE,
            "field_MSE": field_MSE
            }

all_results = []
for lmb in lmbs:
    ckpt_path = ckpt_dict[lmb]
    fm = represent.Represent.load_from_checkpoint(ckpt_path).to(device)
    fm.eval()
    fm.entropy_bottleneck.update(force=True, update_quantiles=True)
    output_dataset = np.load(f'/monolith/global_data/astro_compression/fm_output/{lmb}_output_dataset.npy')
    all_image_stats = defaultdict(AverageMeter)
    for image_index in tqdm(range(30)):
        input_array = test_dataset[image_index, :, :, :]
        output_array = output_dataset[image_index, :, :, :]
        single_image_stats = get_single_image_stats(fm, input_array, output_array)
        for key, value in single_image_stats.items():
            all_image_stats[key].update(value)
    results = {k: meter.avg for k,meter in all_image_stats.items()}
    all_results.append(results)


with open("cosmo_compression/compression_experiments/fm_results.pkl", "wb") as f:
    pickle.dump(all_results, f)