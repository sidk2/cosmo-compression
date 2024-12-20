from pathlib import Path
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning import Trainer
from lightning.pytorch import seed_everything

import wandb
import os

from cosmo_compression.data.data import CAMELS
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
    default=100_000, 
    type=int,
    help="steps to run for",
    required=False,
)
parser.add_argument(
    "--batch_size",
    default = 8,
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
    default=500,
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
# Architecture 
parser.add_argument(
    "--encoder",
    default='resnet18',
    type=str,
    required=False,
)

parser.add_argument(
    "--latent_dim",
    default=256*256,
    type=int,
    required=False,
)


# Extras
parser.add_argument('--use_wandb', action='store_true', default=False, help='Set this flag to use Weights and Biases for logging')
parser.add_argument('--profile', action='store_true', default=False, help='Set this flag to use a profiler')


def train(args,):
    # fix training seed
    seed_everything(42, workers=True)

    logger = None
    if args.use_wandb:
        run_name = args.run_name
        logger = WandbLogger(project="hierarchical_representations", name=run_name, log_model=False)
        logger.log_hyperparams(vars(args))
    else:
        run_name = "test_run"  # You can set a default name for non-logging runs
        print(f"Running without Weights and Biases logging.")

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
        save_top_k=3,
        monitor='val_loss',
        save_last=True,
        every_n_train_steps=args.save_every,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    fm = Represent(
        encoder=args.encoder,
        latent_dim=args.latent_dim,
        log_wandb=args.use_wandb,
        unconditional=args.unconditional,
    )
    # fm = Represent.load_from_checkpoint("soda-comp/last-v4.ckpt").cuda()

    trainer = Trainer(
        max_steps=args.max_steps, 
        gradient_clip_val=1.0,
        logger=logger,
        log_every_n_steps=50,
        accumulate_grad_batches=args.accumulate_gradients if args.accumulate_gradients is not None else 1,
        callbacks=[checkpoint_callback,lr_monitor,],
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
    print(args)
    train(args)
