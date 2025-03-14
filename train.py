from pathlib import Path
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import random

from torchvision.datasets import CelebA
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning import Trainer
from lightning.pytorch import seed_everything

import torch.nn as nn

import wandb
import os

from cosmo_compression.model import represent
from cosmo_compression.data import data


import torch
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

parser = ArgumentParser()
parser.add_argument(
    "--output_dir",
    default="",
    help="output_directory",
    required=False,
)
parser.add_argument(
    "--run_name",
    default="identity-comp",
    type=str,
    help="weights and biases run name",
    required=False,
)

# Models
parser.add_argument('--unconditional', action='store_true', default=False,)

# Training
parser.add_argument(
    "--learning_rate",
    default=2e-4,
    type=float,
    help="learning rate",
    required=False,
)
parser.add_argument(
    "--grad_clip",
    default=1.0,
    help="gradient norm clipping",
    required=False,
)
parser.add_argument(
    "--total_steps",
    default=2_000 ,
    help="total training steps",
    required=False,
)
parser.add_argument(
    "--warmup",
    default=5000,
    help="learning rate warmup",
    required=False,
)
parser.add_argument(
    "--max_steps",
    default=10_000_000,
    type=int,
    help="steps to run for",
    required=False,
)
parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="batch size",
    required=False,
)
parser.add_argument(
    "--accumulate_gradients",
    default=None,
    type=int,
    help="steps to accumulate grads",
    required=False,
)
parser.add_argument(
    "--num_workers",
    default=2,
    type=int,
    help="workers of Dataloader",
    required=False,
)
parser.add_argument(
    "--save_every",
    default=100,
    type=int,
    required=False,
    help="frequency of saving checkpoints, 0 to disable during training",
)
parser.add_argument(
    "--eval_every",
    default=200,
    type=int,
    required=False,
    help="frequency of evaluating model, 0 to disable during training",
)

parser.add_argument(
    "--latent_dim",
    default=256,
    type=int,
    required=False,
)

# Extras
parser.add_argument('--use_wandb', action='store_true', default=False, help='Set this flag to use Weights and Biases for logging')
parser.add_argument('--profile', action='store_true', default=False, help='Set this flag to use a profiler')

def train(args):
    # fix training seed
    seed_everything(137, workers=True)
    dataset = 'CAMELS' # Hard coded for now, make a command line arg

    logger = None
    if args.use_wandb:
        run_name = args.run_name
        logger = WandbLogger(project="hierarchical_representations", name=run_name, log_model=False)
        logger.log_hyperparams(vars(args))
    else:
        run_name = "test_run"  # You can set a default name for non-logging runs
        print(f"Running without Weights and Biases logging.")

    dataset == 'CAMELS'
    train_data = data.CAMELS(
    idx_list=range(14_000),
    map_type='Mcdm',
    parameters=['Omega_m', 'sigma_8',],
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_data = data.CAMELS(
        idx_list=range(14_000, 15_000),
        map_type='Mcdm',
        parameters=['Omega_m', 'sigma_8',],
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
        
    print(f'Using {len(train_data)} training samples and {len(val_data)} validation samples.')

    checkpoint_callback_phase_0 = ModelCheckpoint(
        dirpath=Path(args.output_dir) / f'{run_name}',
        filename='step={step}-{val_loss:.3f}',
        save_top_k=1,
        monitor='val_loss',
        save_last=True,
        every_n_train_steps=args.save_every,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu', generator=None)
            m.bias.data.fill_(0.01)
    
    fm = represent.Represent(
        log_wandb=args.use_wandb,
        unconditional=args.unconditional,
        latent_img_channels = 32,
    )
        
    fm.apply(init_weights)
    fm.train()
    
    trainer = Trainer(
        max_steps=args.max_steps,
        gradient_clip_val=1.0,
        logger=logger,
        log_every_n_steps=50,
        accumulate_grad_batches=args.accumulate_gradients if args.accumulate_gradients is not None else 1,
        callbacks=[checkpoint_callback_phase_0, lr_monitor],
        devices=4,
        check_val_every_n_epoch=None,
        val_check_interval=args.eval_every,
        max_epochs=300,
        profiler="simple" if args.profile else None,
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
    )
    trainer.fit(model=fm, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
if __name__ == "__main__":
    args = parser.parse_args()
    train(args)