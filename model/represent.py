import torch
from torch import nn
import timm
from diffusers import UNet2DModel
from diffusers.models.embeddings import TimestepEmbedding

from lightning.pytorch.utilities import rank_zero_only
from lightning import LightningModule

from typing import Optional, Tuple
import numpy as np
from .flow_matching import FlowMatching

import wandb
import matplotlib.pyplot as plt
from PIL import Image
import io
#import Pk_library as PKL
from sklearn.manifold import TSNE

from .unet import UNet
from .resnet import ResNet


# plt.style.use("science")



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

def get_2d_embeddings(embeddings,): 
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
        encoder: str = 'resnet18',
        latent_dim: int = 32,
        learning_rate: float = 3.e-4,
        n_sampling_steps: int = 50,
        unconditional: bool = False,
        log_wandb: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.unconditional = unconditional
        self.log_wandb = log_wandb
        self.encoder = self.initialize_encoder(encoder, latent_dim*9, 1)
        velocity_model = self.initialize_velocity(latent_dim=latent_dim) 
        self.decoder = FlowMatching(velocity_model)
        self.validation_step_outputs = []

    def initialize_velocity(self, latent_dim):
        return UNet(
            n_channels = 1,
            time_dim = 256,
            latent_dim = latent_dim,
        )

    def initialize_encoder(self, encoder_type, latent_dim, in_channels):
        encoder = timm.create_model(
                'resnet18', 
                pretrained=False, 
                in_chans=in_channels, 
                num_classes=latent_dim,
            )
        return encoder


    def get_loss(self, batch, ):
        cosmology, y = batch
        # Train representation
        h = self.encoder(y) if not self.unconditional else None
        # if h is not None:
        #     h = self.h_embedding(h)
        x0 = torch.randn_like(y) 
        decoder_loss = self.decoder.compute_loss(
            x0 = x0,
            x1 =y,
            h = h,
        )
        return decoder_loss 

           
    def training_step(self, batch, batch_idx, ):
        loss = self.get_loss(batch=batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch=batch)
        self.validation_step_outputs.append({"val_loss": loss, "batch": batch})

    def on_validation_epoch_end(self):
        mean_val_loss = torch.stack(
            [output["val_loss"] for output in self.validation_step_outputs]
        ).mean()
        self.log("val_loss", mean_val_loss, prog_bar=True, sync_dist=True)
        batch = self.validation_step_outputs[0]["batch"]
        self._log_figures(batch, log=self.log_wandb)
        self.validation_step_outputs.clear()

    @rank_zero_only
    def _log_figures(self, batch, log=True,):
        cosmology, y = batch
        h = self.encoder(y) if not self.unconditional else None
        # if h is not None:
        #     h = self.h_embedding(h)
        x0 = torch.randn_like(y) 
        pred = self.decoder.predict(
            x0,
            h=h,
            n_sampling_steps=self.hparams.n_sampling_steps,
        )
        if log:
            print("Logging")
            # plot field reconstruction 
            fig, ax = plt.subplots(1,2, figsize=(8,4 ))
            ax[0].imshow(y[0,0].detach().cpu().numpy(), cmap='viridis')    
            ax[1].imshow(pred[0,0].detach().cpu().numpy(), cmap='viridis')    
            ax[0].set_title('x')
            ax[1].set_title('Reconstructed x')
            plt.savefig("field_construction.png")
            log_matplotlib_figure("field_reconstruction")
            plt.close()

            # add Pk reconstruction
            #TODO: might need to undo standarization
            # k, true_power = compute_pk(y.cpu().numpy(), MAS=None, box_size=25.,)
            # k, pred_power = compute_pk(pred.cpu().numpy(), MAS=None,box_size=25.,)
            # fig, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}, sharex=True)
            # k_nyquist = np.pi / (25./ 256)
            # axs[0].loglog(k[k < k_nyquist], true_power[k < k_nyquist], label="True")
            # axs[0].loglog(k[k < k_nyquist], pred_power[k < k_nyquist], label="Reconstructed")
            # axs[1].set_ylim(0.8,1.2)
            # axs[0].legend()
            # axs[1].semilogx(k[k < k_nyquist], np.sqrt(pred_power[k < k_nyquist] / true_power[k < k_nyquist]))
            # axs[1].set_xlabel('k')
            # log_matplotlib_figure("pks")
            # plt.close()

            # low_dim_h = get_2d_embeddings(h.detach().cpu().numpy())

            # fig, ax = plt.subplots(ncols=2, figsize=(16,6))
            # for p in range(2):

            #     _ = ax[0].scatter(low_dim_h[:, 0], low_dim_h[:, 1], c=cosmology[:,0], cmap='viridis', s=10)
            #     plt.clorbar()
            #     _ = ax[1].scatter(low_dim_h[:, 0], low_dim_h[:, 1], c=cosmology[:,1], cmap='viridis', s=10)
            #     plt.clorbar()
            #     fig.suptitle('2D t-SNE Representation of Embeddings')
            #     for axis in ax:
            #         axis.set_xlabel('t-SNE Dimension 1')
            #         axis.set_ylabel('t-SNE Dimension 2')

            
            #log_matplotlib_figure("embeddings")
            #plt.close()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

