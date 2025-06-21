import os
# make sure to pick whichever GPUs you actually have available
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
# your HF cache remains the same
os.environ["HF_HOME"] = "../../../monolith/global_data/astro_compression/"

import numpy as np
from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import EarlyStopping

from cosmo_compression.model import represent
from cosmo_compression.data import data

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')


def get_celeba_dataloaders(
    root, batch_size, num_workers, transforms_cfg, train_size=None, val_size=None
):
    # (unchanged from before)
    ds = load_dataset("flwrlabs/celeba", cache_dir=root)
    if train_size:
        ds["train"] = ds["train"].select(range(min(train_size, len(ds["train"]))))
    if val_size:
        ds["valid"] = ds["valid"].select(range(min(val_size, len(ds["valid"]))))
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
        ds["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: (
            torch.stack([x["pixel_values"] for x in batch]),
            torch.zeros(len(batch), dtype=torch.long),
        ),
    )
    valid_loader = DataLoader(
        ds["valid"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: (
            torch.stack([x["pixel_values"] for x in batch]),
            torch.zeros(len(batch), dtype=torch.long),
        ),
    )
    return train_loader, valid_loader, len(ds["train"]), len(ds["valid"])


def get_camels_dataloaders(
    batch_size,
    num_workers,
    idx_train,
    idx_val,
    map_type,
    parameters,
    suite,
    camels_data,
):
    """
    Now we explicitly forward `suite` and `camels_data` (i.e. 'WDM') into data.CAMELS.
    """
    print(f"Using {len(idx_train)} training points and {len(idx_val)} validation points.")
    train_data = data.CAMELS(
        idx_list=idx_train,
        map_type=map_type,         # e.g. 'Mcdm' or 'WDM'
        parameters=parameters,     # e.g. ['Omega_m', 'sigma_8']
        suite=suite,               # e.g. 'IllustrisTNG'
        dataset=camels_data,          # e.g. 'WDM'
    )
    val_data = data.CAMELS(
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

    # ------------------------
    # 3) Decide which dataset we’re using
    # ------------------------
    if args.dataset == "celeba":
        celeba_transforms = [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
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
        # Build index lists for a small subset:
        split_list = np.random.permutation(14000)
        idx_train = split_list[: (args.train_size or 14000)]
        idx_val = split_list[(args.train_size or 14000) : (args.train_size or 14000) + (args.val_size or 1000)]
        # train_end = args.train_size if args.train_size else 14000
        # val_end = (args.train_size or 14000) + (args.val_size or 1000)
        # idx_train = range(train_end)
        # idx_val = range(train_end, val_end)
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
        )

    print(
        f"▶️ Using {n_train} training samples and {n_val} validation samples from {args.dataset} (suite={getattr(args, 'camels_suite', 'N/A')}, data={getattr(args, 'camels_data', 'N/A')})."
    )

    # ------------------------
    # 4) Model + checkpointing
    # ------------------------
    # If a pretrained checkpoint is given, load it via Lightning’s load_from_checkpoint.
    if args.pretrained_ckpt:
        print(f"🔄 Loading pretrained weights from {args.pretrained_ckpt} …")
        fm = represent.Represent.load_from_checkpoint(args.pretrained_ckpt,
                                                     log_wandb=args.use_wandb,
                                                     latent_img_channels=args.latent_img_channels)
        
        # Freeze the encoder, and most of the decoder
        # for param in fm.encoder.parameters():
        #     param.requires_grad = False
        for param in fm.decoder.parameters():
            param.requires_grad = False
        for param in fm.decoder.velocity_model.parameters():
            param.requires_grad = False
            
        # for param in fm.encoder.resnet_list[0].resnet_layers[-1].parameters():
        #     param.requires_grad = True
        # for param in fm.encoder.resnet_list[0].out_conv.parameters():
        #     param.requires_grad = True
        for param in fm.decoder.velocity_model.pool.parameters():
            param.requires_grad = True
            
    else:
        # fresh new model if no checkpoint provided
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
    parser.add_argument(
        "--run_name",
        default="finetuning",
        type=str,
        help="WandB run name (if using wandb).",
    )

    # ── Which dataset? ───────────────────────────────────────
    parser.add_argument(
        "--dataset",
        choices=["camels", "celeba"],
        default="camels",
        help="Which dataset to use: 'celeba' or 'camels'",
    )
    parser.add_argument(
        "--celeba_root",
        type=str,
        default="../../../monolith/global_data/astro_compression/",
        help="Root directory for CelebA HF mirror",
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

    # ── Fine‐tuning arguments ───────────────────────────────
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=None,
        help="If set, path to a .ckpt file to load for fine‐tuning",
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
        default=500,
        help="Number of training samples to use (e.g. 500 for quick fine‐tuning).",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=100,
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
