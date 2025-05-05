import os
import torch
from compressai.zoo import image_models
from compressai.utils.eval_model import __main__ as cai_eval
from compressai.datasets.ndarray import NdArrayDataset
import numpy as np
from pathlib import Path
import Pk_library as PKL
import matplotlib.pyplot as plt
from torch.utils import data as torchdata
import matplotlib as mpl

from cosmo_compression.model import represent
from cosmo_compression.data import data

from compressai.zoo import image_models
from compressai.utils.eval_model import __main__ as cai_eval
from compressai.datasets.ndarray import NdArrayDataset

mpl.style.use('seaborn-v0_8-colorblind')
plt.rcParams["font.family"] = "serif"

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# load dataset
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

# didn't norm
test_dataset_hp = NdArrayDataset(np.load("/monolith/global_data/astro_compression/CAMELS/images/test/test_data.npy"), single=True)

# normed
test_dataset_fm: torchdata.Dataset = data.CAMELS(
    idx_list=range(14_000, 15_000),
    map_type="Mcdm",
    dataset="LH",
    parameters=["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "Omega_b"],
)


# load hp model
hp_lmbs = ["0.0009", "0.0018", "0.0067", "0.013", "0.025", "0.0932", "0.18", "0.36"]
hp_ckpt_root = "/monolith/global_data/astro_compression/compressai_ckpts"
hp_lmb = "0.0009"
ckpt_path = os.path.join(hp_ckpt_root, f"lambda_{hp_lmb.replace(".", "_")}_best_loss.pth.tar")
model = image_models["bmshj2018-hyperprior"](quality=3)
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.update(force=True)

# load fm model
# low is higher rate
fm_lmbs = ["1.5", "1", "0.5", "0.1", "0.01", "0.005", "0.001", "0.0001"]
fm_ckpt_root = "/monolith/global_data/astro_compression/train_compression_model"
fm_ckpt_dict = {"1.5":Path(fm_ckpt_root) /  "lmb_1.5_full" / 'step=step=6900-val_total_loss=0.514.ckpt',
             "1":Path(fm_ckpt_root) /  "lmb_1_full" / 'step=step=6900-val_total_loss=0.488.ckpt',
             "0.5":Path(fm_ckpt_root) /  "lmb_0.5_full" / 'step=step=6900-val_total_loss=0.448.ckpt',
             "0.1":Path(fm_ckpt_root) /  "lmb_0.1_full" / 'step=step=6100-val_total_loss=0.383.ckpt',
             "0.01":Path(fm_ckpt_root) /  "lmb_0.01_full" / 'step=step=7700-val_total_loss=0.330.ckpt',
             "0.005":Path(fm_ckpt_root) /  "lmb_0.005_full" / 'step=step=6100-val_total_loss=0.332.ckpt',
             "0.001":Path(fm_ckpt_root) /  "lmb_0.001_full" / 'step=step=6300-val_total_loss=0.339.ckpt',
             "0.0001":Path(fm_ckpt_root) /  "lmb_0.0001_full" / 'step=step=6900-val_total_loss=0.330.ckpt',
             "no_compression":Path(fm_ckpt_root) /  "lossless_latent_full_training" / 'step=step=42300-val_loss=0.254.ckpt',
             }
ckpt_root = "/monolith/global_data/astro_compression/train_compression_model"

fm_lmb="0.0001"
fm_ckpt_path = fm_ckpt_dict[fm_lmb]
fm = represent.Represent.load_from_checkpoint(fm_ckpt_path).to(device)
fm.eval()
fm.entropy_bottleneck.update(force=True, update_quantiles=True)

def get_bpps(model_type, model, image_tensor):
    if model_type == "hp":
        compress_stats = cai_eval.inference(model, image_tensor)
        stats = {
            'compress_bpp': float(compress_stats['bpp'])
        }
    elif model_type == "fm":
        image_tensor=image_tensor.cuda()
        h = model.encoder(image_tensor.unsqueeze(0))
        strings = model.entropy_bottleneck.compress(h)
        compress_bpp = len(strings[0]) * 8 / (256 * 256)
        stats = {
            'compress_bpp': float(compress_bpp)
        }
    return stats


def get_decoding(model_type, model, input_array):
    if model_type == "hp":
        input_tensor = torch.from_numpy(input_array)
        with torch.no_grad():
            out_net = model.forward(input_tensor.unsqueeze(0))
        out_crop = out_net['x_hat']
        output_array = out_crop.cpu().numpy()[0,:,:,:]
    elif model_type == "fm":
        input_tensor = torch.from_numpy(input_array)
        img = torch.from_numpy(input_array).unsqueeze(0).cuda()
        n_sampling_steps = 30
        t = torch.linspace(0, 1, n_sampling_steps).cuda()     
        h = fm.encoder(img)
        h_hat, h_likelihoods = fm.entropy_bottleneck(h) 
        x0 = torch.randn((1, 1, 256, 256), device="cuda")
        pred = fm.decoder.predict(x0.cuda(), h=h_hat, n_sampling_steps=n_sampling_steps)
        print(pred.shape)
        output_array = pred.cpu().numpy()[0, :, :, :]
    # shape is [1, 256, 256]
    return output_array

# get hp decoding and spectrum
image_index = 0
input_array_hp  = test_dataset_hp[0, :, :, :]
input_array_fm, _ = test_dataset_fm[0]

output_array_hp = get_decoding("hp", model, input_array_hp)
output_array_fm = get_decoding("fm", fm, input_array_fm)


def get_single_image_stats(model_type, model, input_array, output_array):
    # input array is [1, 256, 256]
    input_tensor = torch.from_numpy(input_array)
    bpp_stats = get_bpps(model_type, model, input_tensor)

    mean = data.NORM_DICT["Mcdm"][256]["mean"]
    std = data.NORM_DICT["Mcdm"][256]["std"]

    if model_type == "fm":
        input_denorm = input_array * std + mean
        output_denorm = output_array * std + mean
    elif model_type == "hp":
        input_denorm = input_array
        output_denorm = output_array


    delta_fields_orig_1 = input_denorm[0, : , :] / np.mean(input_denorm) - 1
    Pk2D = PKL.Pk_plane(delta_fields_orig_1, 25.0, "None", 1, verbose=False)
    k_orig = Pk2D.k
    Pk_orig = Pk2D.Pk

    delta_fields_pred_1 = output_denorm[0, : , :] / np.mean(output_denorm) - 1
    Pk2D = PKL.Pk_plane(delta_fields_pred_1, 25.0, "None", 1, verbose=False)
    k_pred = Pk2D.k
    Pk_pred = Pk2D.Pk

    spectrum_MSE = np.mean((Pk_orig - Pk_pred) ** 2)
    log_spectrum_MSE = np.mean((np.log10(Pk_orig) - np.log10(Pk_pred)) ** 2)

    field_MSE = np.mean((input_array[0,:,:] - output_array[:,:]) ** 2)
    result_dict = {"compress_bpp":bpp_stats["compress_bpp"],
            "spectrum_MSE": spectrum_MSE,
            "log_spectrum_MSE": log_spectrum_MSE,
            "field_MSE": field_MSE,
            "k_orig": k_orig,
            "k_pred": k_pred,  
            "Pk_orig":Pk_orig,
            "Pk_pred":Pk_pred,
            "input_denorm":input_denorm,
            "output_denorm":output_denorm
            }

    return result_dict



hp_results = get_single_image_stats('hp', model, input_array_hp, output_array_hp)
fm_results = get_single_image_stats('fm', fm, input_array_fm, output_array_fm)


# plotting
fig, axes = plt.subplots(1, 4, figsize=(12,3))

axes[0].imshow(hp_results["input_denorm"][0,:,:])
axes[0].set_title("Original Density Field")
axes[0].set_xlabel("32 bits per pixel", labelpad=20)
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].imshow(hp_results["output_denorm"][0,:,:])
axes[1].set_title("After VAE Compression")
axes[1].set_xlabel(f"{hp_results['compress_bpp']:.2f} bits per pixel", labelpad=20)
axes[1].set_xticks([])
axes[1].set_yticks([])

axes[2].imshow(fm_results["output_denorm"][0,:,:])
axes[2].set_title("After Flow Matching Compression")
axes[2].set_xlabel(f"{fm_results['compress_bpp']:.2f} bits per pixel", labelpad=20)
axes[2].set_xticks([])
axes[2].set_yticks([])

axes[3].plot(hp_results["k_orig"][:-2], hp_results["Pk_orig"][:-2], label = "Original")
axes[3].plot(hp_results["k_pred"][:-2], hp_results["Pk_pred"][:-2], label = "VAE")
axes[3].plot(fm_results["k_pred"][:-2], fm_results["Pk_pred"][:-2], label = "Flow Matching")

axes[3].set_xscale("log")
axes[3].set_yscale("log")
axes[3].set_title("Power Spectra")
axes[3].set_xlabel("Wavenumber $k\,[h/Mpc]$")
axes[3].set_ylabel("$P(k)\,[(Mpc/h)^2]$")
axes[3].legend(fontsize=8)

plt.tight_layout()

plt.savefig("cosmo_compression/compression_experiments/comparison.png", dpi=300)