import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')


class AnnealedVAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta_start: float = 1e-6,  # Starting beta value
                 beta_end: float = 0.01,     # Final beta value
                 num_range: int = 800,    # Steps to reach final beta
                 num_end: int = 200,      # Steps to stay at final beta
                 **kwargs
                ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_range = num_range
        self.num_end = num_end
        self.cycle_length = num_range + num_end
        self.current_step = 0
        self.beta = beta_start

        modules = []
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]

        in_dim = input_dim
        # Build Encoder
        for h_dim in hidden_dims:
            # if kwargs['batch_norm']:
            #     modules.append(
            #         nn.Sequential(
            #             nn.Linear(in_dim, h_dim),
            #             nn.BatchNorm1d(h_dim),
            #             nn.LeakyReLU())
            #     )
            # else:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.LeakyReLU())
            )
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-1], input_dim),
                            nn.Tanh())

    def update_beta(self):
        """Update beta according to cyclic annealing schedule"""
        # Get position within current cycle
        cycle_step = self.current_step % self.cycle_length
        
        if cycle_step < self.num_range:
            # Linear increase from beta_start to beta_end
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * (cycle_step / self.num_range)
        else:
            # Hold at beta_end for num_end steps
            self.beta = self.beta_end
            
        self.current_step += 1
    
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var]

    def loss(self, x, x_recon, mu, log_var):
        self.update_beta()  # Update beta value
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + self.beta * kl_loss
        losses = {
            'loss': total_loss,
            'Reconstruction_Loss': recon_loss.detach(),
            'KLD': kl_loss.detach(),
            'beta': self.beta
        }
        return losses
    
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        w2w space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input weight x, returns the reconstructed weight
        :param x: (Tensor) [B x C]
        :return: (Tensor) [B x C]
        """
        return self.forward(x)[0]
