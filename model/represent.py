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
    # Save plot to a buffer, otherwise wandb does ugly plotly
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=300,
    )
    buf.seek(0)
    image = Image.open(buf)
    # Log the plot to wandb
    wandb.log({f"{figure_label}": wandb.Image(image)})


class Represent(LightningModule):
    def __init__(
        self,
        unconditional: bool = False,
        log_wandb: bool = True,
        reverse: bool = False,
        latent_img_channels: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.unconditional = unconditional
        self.log_wandb = log_wandb
        self.latent_img_channels = latent_img_channels
        self.encoder = self.initialize_encoder(in_channels=1)
        velocity_model = self.initialize_velocity()
        self.decoder = fm.FlowMatching(velocity_model, reverse=reverse)
        self.validation_step_outputs = []
        

    def initialize_velocity(self) -> nn.Module:
        return unet.UNet(
            n_channels=1,
            time_dim=256,
            latent_img_channels = self.latent_img_channels,
        )

    def initialize_encoder(self, in_channels: int) -> nn.Module:
        return resnet.ResNetEncoder(in_channels=in_channels, latent_img_channels = self.latent_img_channels)

    def get_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        y, cosmo = batch
                
        # Train representation
        t = torch.rand((y.shape[0]), device = y.device)
        h = self.encoder(y) if not self.unconditional else None
        x0 = torch.randn_like(y)
        decoder_loss = self.decoder.compute_loss(
            x0=x0,
            x1=y,
            h=h,
            t=t,
        )
        return decoder_loss

    def training_step(
        self,
        batch: Tuple[np.array, np.array],
    ) -> torch.Tensor | int:
        self.optimizers().step()
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
            [output["val_loss"] for output in self.validation_step_outputs]
        ).mean()
        self.log("val_loss", mean_val_loss, prog_bar=True, sync_dist=True)
        batch = self.validation_step_outputs[0]["batch"]
        self._log_figures(batch, log=self.log_wandb)
        self.validation_step_outputs.clear()
        
        self.optimizers().step()
        
    @utilities.rank_zero_only
    def _log_figures(
        self,
        batch: Tuple[np.array, np.array],
        log=True,
    ) -> None:
        y, cosmo = batch
        h = self.encoder(y)
        x0 = torch.randn_like(y)
        pred = self.decoder.predict(
            x0,
            h=h,
            n_sampling_steps=30,
        )
        if log:
            print("Logging")
            # plot field reconstruction
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].imshow(y[0, :, : , :].detach().cpu().permute(1, 2, 0).numpy())
            ax[1].imshow(pred[0, :, : , :].detach().cpu().permute(1, 2, 0).numpy())
            ax[0].set_title("x")
            ax[1].set_title("Reconstructed x")
            plt.savefig("cosmo_compression/results/field_reconstruction.png")
            log_matplotlib_figure("field_reconstruction")
            plt.close()
            
            fig, ax = plt.subplots(2, 2, figsize=(8, 8))
            ax[0, 0].imshow(h[0, 0, : , :].detach().unsqueeze(-1).cpu().numpy())
            ax[0, 1].imshow(h[0, 1, : , :].detach().unsqueeze(-1).cpu().numpy())
            ax[1, 0].imshow(h[0, 2, : , :].detach().unsqueeze(-1).cpu().numpy())
            ax[1, 1].imshow(h[0, 3, : , :].detach().unsqueeze(-1).cpu().numpy())
            ax[0, 0].set_title("Encoder 1")
            ax[0, 1].set_title("Encoder 2")
            ax[1, 0].set_title("Encoder 3")
            ax[1, 1].set_title("Encoder 4")
            plt.savefig("cosmo_compression/results/latents.png")
            log_matplotlib_figure("latents")
            plt.close()

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
