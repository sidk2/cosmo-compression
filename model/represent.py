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

from torchvision import transforms as T

from cosmo_compression.model import flow_matching as fm
from cosmo_compression.model import unet
from cosmo_compression.model import resnet
from cosmo_compression.model.autoenc import VariationalAutoEncoder  # Import the AutoEncoder

def compute_pk(
    mesh: np.array,
    mesh_2: Optional[np.array] = None,
    box_size: Optional[float] = 400.0,
    MAS: Optional[str] = None,
    MAS_2: Optional[str] = None,
) -> Tuple[np.array, np.array]:
    """Compute the power spectrum of a mesh, or cross correlations of two meshes if mesh_2 given

    Args:
        mesh (np.array): overdensity mesh
        mesh_2 (Optional[np.array], optional): overdensity mesh to cross correlate.
            Defaults to None.
        box_size (Optional[float], optional): size of the periodic box.
            Defaults to 500.0.

    Returns:
        Tuple[np.array, np.array]: k values and power spectrum / cross correlation coefficient
    """
    if mesh_2 is not None:
        cross_pk = PKL.XPk([mesh, mesh_2], box_size, MAS=[MAS, MAS_2], threads=1)
        return cross_pk.k3D, cross_pk.XPk[:, 0, 0] / np.sqrt(
            cross_pk.Pk[:, 0, 0] * cross_pk.Pk[:, 0, 1]
        )
    else:
        pk = PKL.Pk(mesh, box_size, MAS=MAS)
        return pk.k3D, pk.Pk[:, 0]


def get_2d_embeddings(
    embeddings,
):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(embeddings)

def log_matplotlib_figure(figure_label: str):
    """log a matplotlib figure to wandb, avoiding plotly

    Args:
        figure_label (str): label for figure
    """
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=300,
    )
    buf.seek(0)
    image = Image.open(buf)
    wandb.log({f"{figure_label}": wandb.Image(image)})


class Represent(LightningModule):
    def __init__(
        self,
        unconditional: bool = False,
        log_wandb: bool = True,
        reverse: bool = False,
        latent_img_channels: int = 64,
        latent_dim: int = 256,  # Latent dimension for the AE
    ):
        super().__init__()
        self.save_hyperparameters()
        self.unconditional = unconditional
        self.log_wandb = log_wandb
        self.latent_img_channels = latent_img_channels
        self.latent_dim = latent_dim
        
        # Initialize AutoEncoder and Flow Matching Decoder
        self.encoder = VariationalAutoEncoder(in_channels=1, latent_dim=self.latent_dim)
        velocity_model = self.initialize_velocity()
        self.decoder = fm.FlowMatching(velocity_model, reverse=reverse)
        
        self.validation_step_outputs = []
        
        self.total_steps = 374
        self.training_steps = 500

    def initialize_velocity(self) -> nn.Module:
        return unet.UNet(
            n_channels=1,
            time_dim=256,
            latent_img_channels=4*self.latent_img_channels,
        )

    def get_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        y, cosmo = batch
        t = torch.rand((y.shape[0]), device=y.device)
        out = self.encoder(y)
        recon = out['reconstruction']
        latent = out['mu']

        # Flow matching phase
        x0 = recon
        encoder_loss = 0
        if self.total_steps < self.training_steps:
            noise = torch.randn_like(x0, device=x0.device)
            x0 = self.total_steps / self.training_steps * x0 + (self.training_steps - self.total_steps) / self.training_steps * noise
            encoder_loss = self.encoder.loss_function(out, x=y)['loss']
        return self.decoder.compute_loss(
            x0=x0,
            x1=y,
            h=(latent, recon),
            t=t,
        ) + encoder_loss

    def training_step(
        self,
        batch: Tuple[np.array, np.array],
    ) -> torch.Tensor:
        loss = self.get_loss(batch=batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[np.array, np.array],
    ) -> None:
        loss = self.get_loss(batch=batch)
        self.validation_step_outputs.append({"val_loss": loss, "batch": batch})

    def on_validation_epoch_end(self) -> None:
        mean_val_loss = torch.stack(
            [o["val_loss"] for o in self.validation_step_outputs]
        ).mean()
        self.log("val_loss", mean_val_loss, prog_bar=True, sync_dist=True)
        batch = self.validation_step_outputs[0]["batch"]
        self._log_figures(batch, log=self.log_wandb)
        self.validation_step_outputs.clear()

        # Increment step counter
        self.total_steps += 1
        self.log("total_steps", self.total_steps, prog_bar=True, sync_dist=True)

    @utilities.rank_zero_only
    def _log_figures(
        self,
        batch: Tuple[np.array, np.array],
        log=True,
    ) -> None:
        y, _ = batch
        out = self.encoder(y)
        recon = out['reconstruction']
        latent = out['mu']
        x0 = recon
        
        if self.total_steps < self.training_steps:
            noise = torch.randn_like(x0, device=x0.device)
            x0 = self.total_steps / self.training_steps * x0 + (self.training_steps - self.total_steps) / self.training_steps * noise
        
        pred = self.decoder.predict(
            x0=x0,
            h=(latent, recon), 
            n_sampling_steps=30,
        )
        if log:
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].imshow(y[0].permute(1, 2, 0).cpu().numpy())
            ax[1].imshow(x0[0].permute(1, 2, 0).cpu().numpy())
            ax[2].imshow(pred[0].permute(1, 2, 0).cpu().numpy())
            for i, title in enumerate(["x", "Initial cond", "Reconstructed x"]):
                ax[i].set_title(title)
            log_matplotlib_figure("field_reconstruction")
            plt.close()

            fig, ax = plt.subplots()
            ax.imshow(recon[0].permute(1, 2, 0).cpu().numpy())
            ax.set_title("AE output")
            log_matplotlib_figure("ae_output")
            plt.close()

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': 5e-5},
            {'params': self.decoder.parameters(), 'lr': 5e-5},
        ])
        return {"optimizer": optimizer, "monitor": "val_loss"}
