from typing import Optional, Tuple, Dict, Any
import math
import io

from lightning.pytorch import utilities
from lightning import LightningModule
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import Pk_library as PKL
from sklearn.manifold import TSNE
import torch
from torch import nn
import wandb

from cosmo_compression.model import train_utils


from compressai.layers import GDN
from compressai.models.utils import conv, deconv

class FactorizedPrior(LightningModule):
    def __init__(
        self,
        N,
        M,
        log_wandb: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.log_wandb = log_wandb
        self.validation_step_outputs = []
        self.encoder = nn.Sequential(
            conv(1, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )
        self.decoder = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 1),
        )
        self.N = N
        self.M = M

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10, min_lr=1e-8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
    
    def training_step(
        self,
        batch: Tuple[np.array, np.array],
    ) -> torch.Tensor | int:
        self.optimizers().step()
        loss_dict = self.get_loss(batch=batch)
        loss = loss_dict['loss']
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(
        self,
        batch: Tuple[np.array, np.array],
    ) -> None:
        loss = self.get_loss(batch=batch)['loss']
        self.validation_step_outputs.append({"val_loss": loss, "batch": batch})
    
    def on_validation_epoch_end(self) -> None:
        mean_val_loss = torch.stack(
            [output["val_loss"] for output in self.validation_step_outputs]
        ).mean()
        self.log("val_loss", mean_val_loss, prog_bar=True, sync_dist=True)
        batch = self.validation_step_outputs[0]["batch"]
        self._log_figures(batch, log=self.log_wandb)
        self.validation_step_outputs.clear()
        
        self.optimizers().step()

    def get_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        y, cosmo = batch
                
        # Train representation
        # the size list is for debugging
        out = self.forward(y)
        recon_loss = nn.functional.mse_loss(out, y)
        loss = recon_loss
        return {'loss': loss}

    @utilities.rank_zero_only
    def _log_figures(
        self,
        batch: Tuple[np.array, np.array],
        log=True,
    ) -> None:
        y, cosmo = batch
        out = self.forward(y)
        if log:
            print("Logging")
            # plot field reconstruction
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].imshow(y[0, :, : , :].detach().cpu().permute(1, 2, 0).numpy())
            ax[1].imshow(out[0, :, : , :].detach().cpu().permute(1, 2, 0).numpy())
            ax[0].set_title("x")
            ax[1].set_title("Reconstructed x")
            plt.savefig("cosmo_compression/vae_results/field_reconstruction.png")
            train_utils.log_matplotlib_figure("field_reconstruction")
            plt.close()