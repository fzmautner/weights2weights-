import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')

class ResidualBlock(nn.Module):
    
    # this is a simple resnet block use only MLP and necessary normalization
    
    def __init__(self, in_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(), 
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, in_dim),
            nn.LayerNorm(in_dim)
        )
        
    def forward(self, x):
        return x + self.block(x)

class YuchenDiscriminator(nn.Module):
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 1024,
                 blocks: int = 1,
    ):
        super(YuchenDiscriminator, self).__init__()
        
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(), 
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim // 2) for _ in range(blocks)]
        )
        
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(), 
            nn.Linear(hidden_dim * 2, 1),
        )
    
    def predict(self, x):
        x = self.first_layer(x)
        x = self.blocks(x)
        x = self.final_layer(x)
        return x
    
    def loss(self, x, y):
        return F.binary_cross_entropy_with_logits(x, y)
    
    def forward(self, x, is_real):
        if is_real:
            labels = torch.ones(x.size(0), 1).to(x.device)
        else:
            labels = torch.zeros(x.size(0), 1).to(x.device)
            
        preds = self.predict(x)
        loss = self.loss(preds, labels)
        return loss, preds
                      
    
class YuchenVAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dim: int = 1024,
                 blocks: int = 1,
                 **kwargs) -> None:
        super(YuchenVAE, self).__init__()

        self.latent_dim = latent_dim
        
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(), 
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self.encoder = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim // 2) for _ in range(blocks)]
        )
        
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim // 2) for _ in range(blocks)]
        )

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 2),
                            nn.ReLU(), 
                            nn.Linear(hidden_dim * 2, input_dim),
        )

    
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C]
        :return: (Tensor) List of latent codes
        """
        
        input = self.first_layer(input)
        
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

    def loss(self, x, x_recon, mu, log_var, beta=0.1):
        recon_loss = F.mse_loss(x_recon, x)
        # recon_loss = F.l1_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + beta * kl_loss
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
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
