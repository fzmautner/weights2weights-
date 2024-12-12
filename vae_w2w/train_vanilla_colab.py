import sys
import os 
sys.path.append(os.path.abspath(os.path.join("", "..")))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
from vanilla_vae import VanillaVAE
import wandb
import gc

def train_vae(
    latent_dim: int,
    hidden_dims: list,
    normalize_by_10: bool = True,
    batch_norm: bool = False,
    activation=nn.LeakyReLU,
    run_name: str = "vanilla-vae",
    wandb_flag: bool = True,
    beta_schedule: dict = None,
    weights_path: str = '../weights_datasets/identities/all_weights.pt',
    epochs: int = 15,
    learning_rate: float = 1e-4,
    batch_size: int = 64,
    device: str = "cuda:0"
):
    """Train a Vanilla VAE model with specified parameters.
    
    Args:
        latent_dim: Dimension of the latent space
        hidden_dims: List of hidden dimensions for the encoder/decoder
        normalize_by_10: Whether to normalize weights by 10
        batch_norm: Whether to use batch normalization
        activation: Activation function to use
        run_name: Name for the training run
        wandb_flag: Whether to log to wandb
        beta_schedule: Dictionary containing beta schedule parameters
        weights_path: Path to the weights dataset
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        batch_size: Batch size for training
        device: Device to train on
    
    Returns:
        trained_vae: Trained VanillaVAE model
    """
    
    # Default beta schedule if none provided
    if beta_schedule is None:
        beta_schedule = {
            'start_epoch': 5,
            'end_epoch': 10,
            'pre_start_beta': 0,
            'start_beta': 0.001,
            'end_beta': 1
        }
    
    # Load weights
    all_weights = torch.load(weights_path, map_location=torch.device("cpu"))
    print(f"all_weights shape: {all_weights.shape}")

    # Print memory usage info
    tensor = all_weights[0]
    size_in_bytes = tensor.element_size() * tensor.numel()
    size_in_gb = size_in_bytes / (1024 ** 3)
    print(f"Size of the all_weights in GB: {size_in_gb * len(all_weights):.6f} GB")

    class w2wDataset(Dataset):
        def __init__(self, all_weights, normalize_by_10):
            self.all_weights = all_weights
            self.normalize_by_10 = normalize_by_10
            
        def __len__(self):
            return len(self.all_weights)
        
        def __getitem__(self, idx):
            if self.normalize_by_10:
                return self.all_weights[idx] * 10
            return self.all_weights[idx]

    # Dataset splits
    train_ratio = 0.8
    test_ratio = 0.1
    val_ratio = 0.1

    # Calculate split sizes
    train_size = int(train_ratio * len(all_weights))
    test_size = int(test_ratio * len(all_weights))
    val_size = len(all_weights) - train_size - test_size

    # Create and split dataset
    full_dataset = w2wDataset(all_weights, normalize_by_10)
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    in_dim = len(all_weights[0])
    vae = VanillaVAE(in_dim, latent_dim, hidden_dims, 
                     batch_norm=batch_norm, activation_func=activation).to(device)
    vae.train()

    def evaluate(dataloader, dataname, vae, beta):
        size = len(dataloader.dataset)
        vae.eval()
        avg_loss = 0
        avg_recon_loss = 0
        avg_kl_loss = 0
        with torch.no_grad():
            for weights in dataloader:
                weights = weights.to(device)
                recon, mu, log_var = vae(weights)
                loss, mse, kl = vae.loss(weights, recon, mu, log_var, 
                                       recon_weight=1.0, kl_weight=beta)
                avg_loss += loss.item() * weights.shape[0]
                avg_recon_loss += mse.item() * weights.shape[0]
                avg_kl_loss += kl.item() * weights.shape[0]
        avg_loss /= size
        avg_recon_loss /= size
        avg_kl_loss /= size
        print(f"{dataname} avg loss = {avg_loss:>8f}; avg recon loss = {avg_recon_loss:>8f}; avg kl loss = {avg_kl_loss:>8f}")
        return avg_loss

    # Initialize optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    # Setup save directory
    vae_save_path = "./outputs/"+run_name+"/"
    os.makedirs(vae_save_path, exist_ok=True)

    # Initialize wandb
    if wandb_flag:
        wandb.init(
            project="w2w-proj",
            name=run_name,
        )

    # Training loop
    for epoch in range(epochs):
        vae.train()
        cur_num_examples = 0
        for i, weights in enumerate(train_loader):
            weights = weights.to(device)
            optimizer.zero_grad()
            recon, mu, log_var = vae(weights)

            # Calculate beta value based on schedule
            if epoch < beta_schedule['start_epoch']:
                beta = beta_schedule['pre_start_beta']
            elif epoch >= beta_schedule['start_epoch'] and epoch < beta_schedule['end_epoch']:
                beta = (beta_schedule['end_beta'] - beta_schedule['start_beta']) / \
                      (beta_schedule['end_epoch'] - beta_schedule['start_epoch']) * \
                      (epoch - beta_schedule['start_epoch']) + beta_schedule['start_beta']
            else:
                beta = beta_schedule['end_beta']
            
            loss, mse, kl = vae.loss(weights, recon, mu, log_var, 
                                   recon_weight=1.0, kl_weight=beta)
            loss.backward()
            optimizer.step()

            if i % 500 == 0:
                print(f"Epoch[{epoch}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.8f}")

            cur_num_examples += weights.shape[0]
            if wandb_flag:
                wandb.log({"Loss": loss,
                          "Recon loss": mse,
                          "KL loss": kl,
                          'beta': beta})

        # Evaluate
        print(f"Epoch[{epoch}/{epochs}]")
        train_avg_loss = evaluate(train_loader, "Train", vae, beta)
        test_avg_loss = evaluate(test_loader, "Test", vae, beta)
        val_avg_loss = evaluate(val_loader, "Validation", vae, beta)

        if wandb_flag:
            wandb.log({
                "Epoch": epoch,
                "Train Avg Loss": train_avg_loss, 
                "Test Avg Loss": test_avg_loss,
                "Validation Avg Loss": val_avg_loss,
            })
        
        # Save checkpoints
        if epoch != (epochs-1) and (epoch+1) % 3 == 0:
            checkpoint_path = os.path.join(vae_save_path, f"vae_checkpoint_epoch{epoch}.pt")
            checkpoint = {'model_state_dict': vae.state_dict()}
            torch.save(checkpoint, checkpoint_path)

    # Save final model
    del all_weights
    torch.save(vae.state_dict(), os.path.join(vae_save_path, "vae.pt"))
    return vae

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train a Vanilla VAE model')
    parser.add_argument('--latent-dim', type=int, required=True)
    parser.add_argument('--hidden-dims', type=int, nargs='+', required=True)
    parser.add_argument('--normalize-by-10', type=bool, default=True)
    parser.add_argument('--batch-norm', type=bool, default=False)
    parser.add_argument('--activation', type=str, default='LeakyReLU',
                       choices=['ReLU', 'LeakyReLU'])
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--wandb-flag', type=bool, default=True)
    parser.add_argument('--beta-schedule', type=str, default=None,
                       help='JSON string containing beta schedule parameters')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Convert activation string to function
    activation_map = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU
    }
    activation = activation_map[args.activation]
    
    # Parse beta schedule if provided
    if args.beta_schedule:
        import json
        beta_schedule = json.loads(args.beta_schedule)
    else:
        beta_schedule = None
    
    # Train model
    vae = train_vae(
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        normalize_by_10=args.normalize_by_10,
        batch_norm=args.batch_norm,
        activation=activation,
        run_name=args.run_name,
        wandb_flag=args.wandb_flag,
        beta_schedule=beta_schedule
    )