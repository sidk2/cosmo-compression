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

import wandb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

from cosmo_compression.model.represent import Represent

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
    default=1_000_000,
    type=int,
    help="steps to run for",
    required=False,
)
parser.add_argument(
    "--batch_size",
    default=8,
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
    default=100,
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
    seed_everything(42, workers=True)
    dataset = 'CelebA' # Hard coded for now, make a command line arg

    logger = None
    if args.use_wandb:
        run_name = args.run_name
        logger = WandbLogger(project="hierarchical_representations", name=run_name, log_model=False)
        logger.log_hyperparams(vars(args))
    else:
        run_name = "test_run"  # You can set a default name for non-logging runs
        print(f"Running without Weights and Biases logging.")

    if dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Pad(
                padding=(39, 19),  # Pad 39 pixels on the left/right and 19 pixels on the top/bottom
                fill=0  # Fill with 0s (black padding)
            ),
            transforms.CenterCrop((256, 256)),  # Crop if necessary to get an exact 256x256 resolution
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

        train_data = CelebA(
            root="./data",
            split="train",
            download=True,
            transform=transform
        )

        subset_size = 12000
        subset_indices = random.sample(range(len(train_data)), subset_size)
        subset_dataset = Subset(train_data, subset_indices)

        # Use the subset dataset for training
        train_data = subset_dataset
        
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        val_data = CelebA(
            root="./data",
            split="valid",
            download=True,
            transform=transform
        )
        
        subset_size = 3000
        subset_indices = random.sample(range(len(val_data)), subset_size)
        subset_dataset = Subset(val_data, subset_indices)

        # Use the subset dataset for training
        val_data = subset_dataset
        
        val_loader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
    elif dataset == 'CAMELS':
        train_data = CAMELS(
        idx_list=range(14_600),
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
        val_data = CAMELS(
            idx_list=range(14_600, 15_000),
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

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.output_dir) / f'{run_name}',
        filename='step={step}-{val_loss:.3f}',
        save_top_k=1,
        monitor='val_loss',
        save_last=False,
        every_n_train_steps=args.save_every,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    fm = Represent(
        latent_dim=args.latent_dim,
        log_wandb=args.use_wandb,
        unconditional=args.unconditional,
    )

    trainer = Trainer(
        max_steps=args.max_steps,
        gradient_clip_val=1.0,
        logger=logger,
        log_every_n_steps=100,
        accumulate_grad_batches=args.accumulate_gradients if args.accumulate_gradients is not None else 1,
        callbacks=[checkpoint_callback, lr_monitor],
        devices=4,
        check_val_every_n_epoch=None,
        val_check_interval=args.eval_every,
        max_epochs=100,
        profiler="simple" if args.profile else None,
        accelerator="gpu",
        strategy='ddp_find_unused_parameters_true',
    )
    trainer.fit(model=fm, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)