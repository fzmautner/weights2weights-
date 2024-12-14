import torchvision
import torch
from torch import nn
import torchvision.transforms as transforms
import tqdm
from utils import load_models
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
This optimization will define a black-box loss function that takes a latent, decodes it with the VAE, integrates these parameters into the stable diffusion pipeline (the unet) and returns the denoising mse loss of the unet using the decoded parameters. The loss function is taken from the standard inversion_vae.py file, and slightly adapted to fit 
the requirements for using EvoTorch.

To optimize for z, we'll use the CMA-ES algorithm, implemented by EvoTorch.

'''

# generation_idx = 0
def invert_evo(network, unet, diffusion_vae, text_encoder, tokenizer, prompt, noise_scheduler, image_path, mask_path, n_time_steps, device, n_epochs=50, n_samples=10, popsize=10, wandb_name=None):

    # diffusion pipeline models
    # unet, diffusion_vae, text_encoder, tokenizer, noise_scheduler = load_models(device)

    latent_dim = network.vae.latent_dim

    if mask_path: 
        mask = Image.open(mask_path)
        mask = transforms.Resize((64,64), interpolation=transforms.InterpolationMode.BILINEAR)(mask)
        mask = torchvision.transforms.functional.pil_to_tensor(mask).unsqueeze(0).to(device).bfloat16()
        mask = mask.repeat(n_time_steps, 1, 1, 1)  # Repeat along batch dim
    else: 
        mask = torch.ones((n_time_steps,1,64,64)).to(device).bfloat16()

    image_transforms = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=image_path, transform=image_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True) 
    weigth_dimensions = torch.load("../files/weight_dimensions.pt")
    prompt = "sks person"

    _n_samples = max(1, n_samples)

    def inversion_loss(_z: torch.Tensor) -> float:
        with torch.no_grad():  

            network.z.data = _z
            
            epoch_loss = 0.0
            for batch, _ in train_dataloader:

                batch = batch.to(device).bfloat16()
                latents = diffusion_vae.encode(batch).latent_dist.sample()
                latents = latents * 0.18215
                latents = latents.repeat(_n_samples, 1, 1, 1)  # [n_time_steps, 4, 64, 64]
                noise = torch.randn_like(latents)  # Will automatically have the repeated shape
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (_n_samples,), device=latents.device)
                
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                text_embeddings = text_encoder(text_input.input_ids.to(device))[0]  # [1, 77, 768]
                text_embeddings = text_embeddings.repeat(_n_samples, 1, 1)  # [n_time_steps, 77, 768]
                
                # forward pass with network
                with network:
                    model_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                    if mask_path:
                        loss = torch.nn.functional.mse_loss(
                            mask * model_pred.float(), 
                            mask * noise.float(), 
                            reduction="mean"
                        )
                    else:
                        loss = torch.nn.functional.mse_loss(
                            model_pred.float(), 
                            noise.float(), 
                            reduction="mean"
                        )
                
                epoch_loss += loss.item()

            return epoch_loss / _n_samples
        
    print('latent dim:', latent_dim)
    problem = Problem(
        "min",  # minimize the objective
        inversion_loss,
        bounds=(-1.3,1.3), 
        solution_length=latent_dim,
        device=device,
        dtype=torch.float32
    )

    # Uncomment to switch CEM and CMA-ES

    # searcher = CEM(
    #     problem=problem,
    #     popsize=popsize,
    #     parenthood_ratio=0.15, # parenthood_ratio 
    #     stdev_init=0.5
    # )

    searcher = CMAES(
        problem=problem,
        stdev_init = 0.75,
        popsize=popsize,
        stdev_min = 1e-5,
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
        for _ in tqdm.tqdm(range(n_epochs)):
            searcher.step()
            # print(searcher.status)
            wandb.log({
                "best_fitness_score": searcher.status['pop_best_eval'],
                "mean_fitness_score": searcher.status['mean_eval'],
                "median_fitness_score": searcher.status['median_eval'],
            })
        wandb.finish()
    else:
        for _ in tqdm.tqdm(range(n_epochs)):
            searcher.step()

    best_potential = searcher.status['pop_best']

    return best_potential