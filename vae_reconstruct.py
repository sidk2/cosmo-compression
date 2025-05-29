import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "../../../monolith/global_data/astro_compression/"

from pathlib import Path
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torchvision import transforms
import random

from torchvision.datasets import CelebA
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning import Trainer
from lightning.pytorch import seed_everything
import matplotlib.pyplot as plt
import numpy as np
# from datasets import load_dataset

import torch.nn as nn
import torch


torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

from cosmo_compression.model import represent
from cosmo_compression.model import google
from cosmo_compression.data import data

def get_celeba_dataloaders(root, batch_size, num_workers, transforms_cfg, train_size=None, val_size=None):
    # Load the Hugging Face mirror
    ds = load_dataset("flwrlabs/celeba", cache_dir="../../../monolith/global_data/astro_compression/")
    # Optionally subset the splits
    if train_size:
        ds["train"] = ds["train"].select(range(min(train_size, len(ds["train"]))))
    if val_size:
        ds["valid"] = ds["valid"].select(range(min(val_size, len(ds["valid"]))))

    # Preprocess transforms
    def preprocess(batch):
        images = [transforms.Compose(transforms_cfg)(img.convert("RGB")) for img in batch["image"]]
        return {"pixel_values": images}

    ds = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names,
        batch_size=32,
    )
    ds.set_format(type="torch", columns=["pixel_values"])

    train_loader = DataLoader(
        ds["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=lambda batch: (torch.stack([x["pixel_values"] for x in batch]), torch.zeros(len(batch), dtype=torch.long))
    )
    valid_loader = DataLoader(
        ds["valid"], batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=lambda batch: (torch.stack([x["pixel_values"] for x in batch]), torch.zeros(len(batch), dtype=torch.long))
    )

    return train_loader, valid_loader, len(ds["train"]), len(ds["valid"])

def get_camels_dataloaders(batch_size, num_workers, idx_train, idx_val, map_type, parameters):
    print(f"Using {len(idx_train)} training points and {len(idx_val)} validation points.")
    train_data = data.CAMELS(
        idx_list=idx_train,
        map_type=map_type,
        parameters=parameters,
    )
    val_data = data.CAMELS(
        idx_list=idx_val,
        map_type=map_type,
        parameters=parameters,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_data), len(val_data)

def main(args):
    seed_everything(137, workers=True)

    logger = None
    if args.use_wandb:
        logger = WandbLogger(project="cosmo_vae", name=args.run_name, log_model=False)
        logger.log_hyperparams(vars(args))
    else:
        print("Running without Weights and Biases logging.")

    if args.dataset == 'celeba':
        celeba_transforms = [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ]
        train_loader, val_loader, n_train, n_val = get_celeba_dataloaders(
            root=args.celeba_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            transforms_cfg=celeba_transforms,
            train_size=args.train_size,
            val_size=args.val_size,
        )
    else:
        # Build index lists based on requested sizes
        train_end = args.train_size if args.train_size else 14000
        val_end = (args.train_size or 14000) + (args.val_size or 1000)
        idx_train = range(train_end)
        idx_val = range(train_end, val_end)
        train_loader, val_loader, n_train, n_val = get_camels_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            idx_train=idx_train,
            idx_val=idx_val,
            map_type='Mcdm',
            parameters=['Omega_m', 'sigma_8'],
        )



    model = google.FactorizedPrior.load_from_checkpoint(args.ckpt_path)
    model.eval()
    print(model)
    x, _ = next(iter(val_loader))
    x = x.cuda()
    with torch.no_grad():
        output, _, _ = model(x)
    # plot field reconstruction
    image_index = 0
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].imshow(x[image_index, :, : , :].detach().cpu().permute(1, 2, 0).numpy())
    ax[1].imshow(output[image_index, :, : , :].detach().cpu().permute(1, 2, 0).numpy())
    ax[0].set_title("x")
    ax[1].set_title("Reconstructed x")
    plt.savefig("cosmo_compression/vae_results/vae_field_reconstruction_0529.png")
    np.save("cosmo_compression/vae_results/vae_field_reconstruction_0529.npy", output[image_index, :, : , :].detach().cpu().permute(1, 2, 0).numpy())
    plt.close()
    return



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=['camels', 'celeba'], default='camels', help="Which dataset to use")
    parser.add_argument('--N_channels', type=int, default=128, help="number of earlier channels")
    parser.add_argument('--M_channels', type=int, default=64, help="number of final layer channels")
    parser.add_argument('--kl_weight', type=float, default=1.0, help="KL loss weight")
    parser.add_argument('--train_size', type=int, default=14_000, help="Number of training samples to use")
    parser.add_argument('--val_size', type=int, default=1000, help="Number of validation samples to use")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--ckpt_path", default="", help="checkpoint filepath")
    #/home/tianqiu/astro/output/128_64_4x4_0.001kl_no_gelu/last.ckpt
    args = parser.parse_args()
    main(args)