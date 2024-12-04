import sys
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
from vanilla_vae import VanillaVAE
from yuchen_vae import YuchenVAE
import wandb
from tqdm import tqdm
device = "cuda:0"


# some flag for training and debugging
normalize_by_10 = True
batch_norm = True
run_name = "vanilla-vae-normalizeby10-batchnorm-biggerhidden_dims-epoch15#1"
wandb_flag = True

# latent_dim = 32
# hidden_dims = [1024, 512, 256, 128, 64]
latent_dim = 128
hidden_dims = [4096, 2048, 1024, 512, 256]
epochs = 15
learning_rate = 1e-4

if wandb_flag:
    wandb.init(
            project="w2w-proj",
            name=run_name,
        )

weights_path = '../all_weights.pt'
all_weights = torch.load(weights_path, map_location=torch.device("cpu"))
print(f"all_weights shape: {all_weights.shape}")

tensor = all_weights[0]
size_in_bytes = tensor.element_size() * tensor.numel()

# Convert to gigabytes
size_in_gb = size_in_bytes / (1024 ** 3)
print(f"Size of the all_weights in GB: {size_in_gb * len(all_weights):.6f} GB")

class w2wDataset(Dataset):
    def __init__(self, all_weights):
        self.all_weights = all_weights
        
    def __len__(self):
        return len(self.all_weights)
    
    def __getitem__(self, idx):
        if normalize_by_10:
            return self.all_weights[idx] * 10
        return self.all_weights[idx]
    
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1
batch_size = 32

# Calculate the sizes of the splits
train_size = int(train_ratio * len(all_weights))
test_size = int(test_ratio * len(all_weights))
val_size = len(all_weights) - train_size - test_size

# Split the dataset
full_dataset = w2wDataset(all_weights)
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size, val_size])

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

in_dim = len(all_weights[0])
# latent_dim = 32
# hidden_dims = [1024, 512, 256, 128, 64]

vae = YuchenVAE(
        input_dim=in_dim,
        latent_dim=latent_dim,
    ).to(device)
vae.train()

def evaluate(dataloader, dataname, vae):
    size = len(dataloader.dataset)
    vae.eval()
    avg_loss = 0
    with torch.no_grad():
        for weights in dataloader:
            weights = weights.to(device)
            recon, mu, log_var = vae(weights)
            avg_loss += vae.loss(weights, recon, mu, log_var)['loss'].item()
    avg_loss /= size
    print(f"{dataname} avg loss = {avg_loss:>8f}")
    return avg_loss


# epochs = 5
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

vae_save_path = "./outputs/"+run_name+"/"
os.makedirs(vae_save_path, exist_ok=True)
# Training loop
for epoch in range(epochs):
    vae.train()
    cur_num_examples = 0
    bar = tqdm(train_loader, desc=f"Epoch[{epoch}/{epochs}]")
    for i, weights in enumerate(bar):
        weights = weights.to(device)
        optimizer.zero_grad()
        recon, mu, log_var = vae(weights)
        
        loss = vae.loss(weights, recon, mu, log_var)
        loss['loss'].backward()
        optimizer.step()
        
        bar.set_postfix(loss=loss['loss'].item())
        
        cur_num_examples += weights.shape[0]
        if wandb_flag:
            wandb.log(loss)

    train_avg_loss = evaluate(train_loader, "Train", vae)
    test_avg_loss = evaluate(test_loader, "Test", vae)
    val_avg_loss = evaluate(val_loader, "Validation", vae)

    if wandb_flag:
        wandb.log({"Epoch": epoch,
                    "Train Avg Loss": train_avg_loss, 
                    "Test Avg Loss": test_avg_loss,
                    "Validation Avg Loss": val_avg_loss,})
    
    if epoch != 0 and (epoch+1) % 5 == 0:
        checkpoint_path = "./outputs/"+run_name+"/"+f"vae_checkpoint_epoch{epoch}.pt"
        checkpoint = {
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, checkpoint_path)
    

# save model

torch.save(vae.state_dict(), vae_save_path+"vae.pt")