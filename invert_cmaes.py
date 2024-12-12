import torch
from torch import nn
import torchvision.transforms as transforms
import tqdm
from utils import load_models, apply_flattened_weights
import wandb
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# CMA-ES:
from evotorch.algorithms import CMAES, CEM
from evotorch.logging import WandbLogger
from evotorch import Problem

from lora_VAEw2w import LoRAw2wVAE



'''
Goal: given a target image, find the latent vector z in the latent space of a VAE whose decoded version best produces the target iamge (minimized denoising objective).

To be more clear, the decoder produces parameters for the LoRA layers of the stable diffusion pipeline. As such, we want to optimize over the latent z to find the best decoded model capable of denoising the target image while being on the dense latent space, preventing it from being out of distribution and overfit to the single target image.

This optimization will define a black-box loss function that takes a latent, decodes it with the VAE, integrates these parameters into the stable diffusion pipeline (the unet) and returns the denoising mse loss of the unet using the decoded parameters.

To optimize for z, we'll use the CMA-ES algorithm, implemented by EvoTorch.

'''


def invert_evo(vae, img_path, mask_path, device, n_epochs=50, n_samples=10, popsize=10, wandb_name=None):

    # diffusion pipeline models
    unet, diffusion_vae, text_encoder, tokenizer, noise_scheduler = load_models(device)

    latent_dim = vae.latent_dim

    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = transform(image).unsqueeze(0).to(device)
    
    if mask_path is not None and mask_path != "":
        mask = Image.open(mask_path)
        mask = transforms.Resize((64,64))(mask)
        mask = transforms.functional.pil_to_tensor(mask).unsqueeze(0).to(device)
    else:
        mask = torch.ones((1,1,64,64)).to(device)

    weigth_dimensions = torch.load("../files/weight_dimensions.pt")
    prompt = "sks person"

    network = LoRAw2wVAE(vae, unet, rank=4, multiplier=1.0, alpha=1.0).to(device)

    # def inversion_loss(z: torch.Tensor) -> float:
    #     # Update network's latent vector
    #     with torch.no_grad():
    #         network.z.copy_(z)
        
    #     # Aggregate loss across samples
    #     loss = 0
    #     for _ in range(n_samples):
    #         with torch.no_grad():
    #             batch = image.to(dtype=torch.bfloat16)
    #             latents = diffusion_vae.encode(batch).latent_dist.sample() * 0.18215
    #             noise = torch.randn_like(latents)
    #             timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
                
    #             text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, 
    #                                 truncation=True, return_tensors="pt")
    #             text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
            
    #         # Use network context manager to apply LoRA updates
    #         with network:
    #             model_pred = unet(latents, timesteps, text_embeddings).sample
    #             loss += torch.nn.functional.mse_loss(mask * model_pred.float(), mask * noise.float())

    #     return loss.item() / n_samples
    _n_samples = max(1, n_samples)

    def inversion_loss(z: torch.Tensor) -> float:
        with torch.no_grad():  # Ensure no gradients are stored
            network.z.copy_(z)
            
            loss = 0
            # Use smaller number of samples initially, increase later
            for _ in range(_n_samples):
                batch = image.to(dtype=torch.bfloat16) 
                latents = diffusion_vae.encode(batch).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
                
                # Clear cache after text encoding
                text_embeddings = text_encoder(tokenizer(prompt, padding="max_length", 
                    max_length=tokenizer.model_max_length, truncation=True, 
                    return_tensors="pt").input_ids.to(device))[0]
                torch.cuda.empty_cache()
                
                with network:
                    model_pred = unet(latents, timesteps, text_embeddings).sample
                    curr_loss = torch.nn.functional.mse_loss(
                        mask * model_pred.float(), 
                        mask * noise.float()
                    )
                    loss += curr_loss.item()  # Convert to item immediately
                    
                # Clear cache after each sample
                torch.cuda.empty_cache()

            return loss / _n_samples
        
    print('latent dim:', latent_dim)
    problem = Problem(
        "min",  # minimize the objective
        inversion_loss,
        bounds=(-2.1,2.1), #
        solution_length=latent_dim,
        device=device,
        dtype=torch.float32
    )

    # searcher = CEM(
    #     problem=problem,
    #     popsize=popsize,
    #     parenthood_ratio=0.2, # parenthood_ratio 
    #     stdev_init=0.05
    # )
        # initial_mean=torch.zeros(latent_dim, device=device),
    searcher = CMAES(
        problem=problem,
        stdev_init = 0.8,
        popsize=popsize,
        stdev_min = 1e-4,
        stdev_max = 1.5,
        limit_C_decomposition=False
    )

    if wandb_name is not None:
        wandb.init(
            project=wandb_name['proj'],
            name=wandb_name['name'],
            config={
                "n_epochs": n_epochs,
                "n_samples": n_samples,
                "latent_dim": latent_dim,
            }
        )
        for epoch in tqdm.tqdm(range(n_epochs)):
            searcher.step()
            # print(searcher.status)
            wandb.log({
                "best_fitness_score": searcher.status['pop_best_eval'],
                "mean_fitness_score": searcher.status['mean_eval'],
                "median_fitness_score": searcher.status['median_eval'],
                # "generation": epoch
            })
        wandb.finish()
    else:
        for _ in tqdm.tqdm(range(n_epochs)):
            searcher.step()

    best_potential = searcher.status['pop_best']
    return best_potential