import os

import numpy as np
from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from cosmo_compression.model import represent
from cosmo_compression.data import data

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

def get_camels_dataloaders(
    batch_size,
    num_workers,
    idx_train,
    idx_val,
    map_type,
    parameters,
    suite,
    camels_data,
    root,
):
    """
    Now we explicitly forward `suite` and `camels_data` (i.e. 'WDM') into data.CAMELS.
    """
    print(f"Using {len(idx_train)} training points and {len(idx_val)} validation points.")
    train_data = data.CAMELS(
        root = root,
        idx_list=idx_train,
        map_type=map_type,         # e.g. 'Mcdm' or 'WDM'
        parameters=parameters,     # e.g. ['Omega_m', 'sigma_8']
        suite=suite,               # e.g. 'IllustrisTNG'
        dataset=camels_data,          # e.g. 'WDM'
    )
    val_data = data.CAMELS(
        root=root,
        idx_list=idx_val,
        map_type=map_type,
        parameters=parameters,
        suite=suite,
        dataset=camels_data,
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
    # ------------------------
    # 1) Set random seeds
    # ------------------------
    seed_everything(137, workers=True)

    # ------------------------
    # 2) (Optional) initialize WandB
    # ------------------------
    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            project="hierarchical_representations",
            name=args.run_name,
            log_model=False,
        )
        logger.log_hyperparams(vars(args))
    else:
        print("🔸 Running without Weights & Biases logger.")


    train_end = args.train_size
    val_end = args.train_size + args.val_size
    idx_train = range(train_end)
    idx_val = range(train_end, val_end)
    
    # Now forward suite and data into our loader util:
    train_loader, val_loader, n_train, n_val = get_camels_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        idx_train=idx_train,
        idx_val=idx_val,
        map_type="Mcdm",                # you could also switch this to "WDM" if that’s what you want
        parameters=["Omega_m", "sigma_8"],
        suite=args.camels_suite,
        camels_data=args.camels_data,
        root=args.root,
    )

    print(
        f"Using {n_train} training samples and {n_val} validation samples (suite={getattr(args, 'camels_suite', 'N/A')}, data={getattr(args, 'camels_data', 'N/A')})."
    )

    fm = represent.Represent(
        log_wandb=args.use_wandb,
        latent_img_channels=args.latent_img_channels,
    )
    # re‐initialize (re‐apply) weight init if you want—optional:
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            m.bias.data.fill_(0.01)
    if not args.pretrained_ckpt:
        fm.apply(init_weights)
    fm.train()

    # Checkpoints: keep the best on val_loss and always save last,
    # saving every N training steps (so you can resume if you crash)
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.output_dir) / f"{args.run_name}",
        filename="step={step}-val_loss={val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        save_last=True,
        every_n_train_steps=args.save_every,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ------------------------
    # 5) Trainer
    # ------------------------
    trainer = Trainer(
        max_steps=args.max_steps,
        gradient_clip_val=args.grad_clip,
        logger=logger,
        log_every_n_steps=50,
        accumulate_grad_batches=args.accumulate_gradients or 1,
        callbacks=[checkpoint_callback, lr_monitor],
        devices=args.gpus,
        check_val_every_n_epoch=1,
        # val_check_interval=args.eval_every,  # you can re-enable if you want more frequent val
        max_epochs=args.max_epochs,
        profiler="simple" if args.profile else None,
        # strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
    )

    # ------------------------
    # 6) Start (fine‐)tuning
    # ------------------------
    trainer.fit(
        model=fm,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    # output + logging
    parser.add_argument(
        "--output_dir", default="", help="output directory for checkpoints etc."
    )
    parser.add_argument("--root", type=str, required=True, help="dataset directory")
    parser.add_argument(
        "--run_name",
        default="finetuning",
        type=str,
        help="WandB run name (if using wandb).",
    )

    # ── CAMELS‐specific arguments ────────────────────────────
    parser.add_argument(
        "--camels_suite",
        type=str,
        default="Astrid",
        help="Which CAMELS suite to use (e.g. IllustrisTNG, SIMBA, etc.)",
    )
    parser.add_argument(
        "--camels_data",
        type=str,
        default="LH",
        help="Which CAMELS data/map type to load (e.g. WDM, Mcdm, etc.)",
    )

    # ── Optimization hyperparameters ────────────────────────
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--accumulate_gradients", default=None, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--save_every", default=50, type=int)
    parser.add_argument("--eval_every", default=50, type=int)
    parser.add_argument("--latent_dim", default=256, type=int)
    parser.add_argument("--latent_img_channels", type=int, default=16)

    # ── Data subset sizes (for small‐scale fine‐tuning) ───────
    parser.add_argument(
        "--train_size",
        type=int,
        required=True,
        help="Number of training samples to use (e.g. 500 for quick fine‐tuning).",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        required=True,
        help="Number of validation samples to use.",
    )

    # ── Trainer settings ────────────────────────────────────
    parser.add_argument("--max_steps", default=2_000, type=int)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--gpus", type=int, default=3, help="How many GPUs to use")
    parser.add_argument("--use_wandb", action="store_true", default=False)

    args = parser.parse_args()
    train(args)
