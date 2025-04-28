import torch
from torch import nn
from typing import List, Tuple, Dict

class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        hidden_dims: List[int] = None,
    ) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        self.hidden_dims = hidden_dims
        self.in_channels = in_channels

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(inplace=True)
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        # Flatten and two linear layers for mu and log_var
        self.flatten_dim = hidden_dims[-1] * 16 * 16
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Build Decoder
        hidden_dims.reverse()
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 16 * 16)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, padding=1),
                    nn.LeakyReLU(inplace=True)
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=3, padding=1)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[0], 16, 16)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar
        }

    def loss_function(self, outputs: Dict[str, torch.Tensor], x: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon = outputs['reconstruction']
        mu = outputs['mu']
        logvar = outputs['logvar']
        recons_loss = nn.functional.mse_loss(recon, x, reduction='mean')
        # KL divergence term
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recons_loss + 0.1*kl_loss
        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss,
            'KL_Divergence': kl_loss
        }

    def sample(self, num_samples: int, current_device: int) -> torch.Tensor:
        
        """
        Sample from the latent space and return the corresponding decoded samples.

        Args:
            num_samples: The number of samples to generate.
            current_device: The device to put the samples on.

        Returns:
            The generated samples, shape (num_samples, 1, 256, 256).
        """
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)['reconstruction']
