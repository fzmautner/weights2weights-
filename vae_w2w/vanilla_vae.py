import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')


class VanillaVAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        self.activation_func = kwargs.get('activation_func', nn.LeakyReLU)

        modules = []
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]

        in_dim = input_dim
        # Build Encoder
        for h_dim in hidden_dims:
            if kwargs['batch_norm']:
                modules.append(
                    nn.Sequential(
                        nn.Linear(in_dim, h_dim),
                        nn.BatchNorm1d(h_dim),
                        self.activation_func())
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Linear(in_dim, h_dim),
                        self.activation_func())
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
            if kwargs['batch_norm']:
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                        nn.BatchNorm1d(hidden_dims[i + 1]),
                        self.activation_func())
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                        self.activation_func())
                )

        self.decoder = nn.Sequential(*modules)

        # if kwargs['batch_norm']:
        #     self.final_layer = nn.Sequential(
        #                     nn.Linear(hidden_dims[-1], input_dim),
        #                     nn.BatchNorm1d(hidden_dims[-1]),
        #                     nn.Tanh())
        # else:
        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-1], input_dim),
                            nn.Tanh())

    
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
        return  [self.decode(z), mu, log_var]

    def loss(self, x, x_recon, mu, log_var, recon_weight=1, kl_weight=1):
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss * recon_weight + kl_loss * kl_weight, \
            recon_loss.detach(), kl_loss.detach()
    
    # def loss_function(self,
    #                   *args,
    #                   **kwargs) -> dict:
    #     """
    #     Computes the VAE loss function.
    #     KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    #     :param args:
    #     :param kwargs:
    #     :return:
    #     """
    #     recons = args[0]
    #     input = args[1]
    #     mu = args[2]
    #     log_var = args[3]

    #     kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
    #     recons_loss =F.mse_loss(recons, input)


    #     kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    #     loss = recons_loss + kld_weight * kld_loss
    #     return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    # def sample(self, num_samples):
    #     z = torch.randn(num_samples, self.latent_dim)
    #     return self.decode(z)
    
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
