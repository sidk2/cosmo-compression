import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2, 3"
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
from datasets import load_dataset

import torch.nn as nn
import torch


torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

from cosmo_compression.model import represent
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


def train(args):
    seed_everything(137, workers=True)

    logger = None
    if args.use_wandb:
        logger = WandbLogger(project="hierarchical_representations", name=args.run_name, log_model=False)
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

    print(f'Using {n_train} training samples and {n_val} validation samples from {args.dataset}.')

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.output_dir) / f'{args.run_name}',
        filename='step={step}-{val_loss:.3f}',
        save_top_k=1,
        monitor='val_loss',
        save_last=True,
        every_n_train_steps=args.save_every,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            m.bias.data.fill_(0.01)

    fm = represent.Represent(
        log_wandb=args.use_wandb,
        latent_img_channels=args.latent_img_channels,
    )
    # fm.apply(init_weights)
    fm.train()

    trainer = Trainer(
        max_steps=args.max_steps,
        gradient_clip_val=args.grad_clip,
        logger=logger,
        log_every_n_steps=50,
        accumulate_grad_batches=args.accumulate_gradients or 1,
        callbacks=[checkpoint_callback, lr_monitor],
        devices=args.gpus,
        check_val_every_n_epoch=None,
        val_check_interval=args.eval_every,
        max_epochs=200,
        profiler="simple" if args.profile else None,
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
    )
    trainer.fit(model=fm, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", default="", help="output_directory")
    parser.add_argument("--run_name", default="identity-comp", type=str, help="WandB run name")
    parser.add_argument("--dataset", choices=['camels', 'celeba'], default='camels', help="Which dataset to use")
    parser.add_argument("--celeba_root", type=str, default='../../../monolith/global_data/astro_compression/', help="Root directory for CelebA data")
    parser.add_argument('--unconditional', action='store_true', default=False)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--total_steps", default=2_000, type=int)
    parser.add_argument("--warmup", default=5000, type=int)
    parser.add_argument("--max_steps", default=10_000_000, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--accumulate_gradients", default=None, type=int)
    parser.add_argument("--num_workers", default=31, type=int)
    parser.add_argument("--save_every", default=100, type=int)
    parser.add_argument("--eval_every", default=50, type=int)
    parser.add_argument("--latent_dim", default=256, type=int)
    parser.add_argument('--train_size', type=int, default=14_000, help="Number of training samples to use")
    parser.add_argument('--val_size', type=int, default=1000, help="Number of validation samples to use")
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--gpus', type=int, default=5, help="Number of GPUs to use")
    parser.add_argument('--latent_img_channels', type=int, default=16, help="Number of latent image channels")

    args = parser.parse_args()
    train(args)
