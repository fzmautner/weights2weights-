import torch
import torchvision
import tqdm
import torchvision.transforms as transforms
from PIL import Image
import warnings
import wandb
warnings.filterwarnings("ignore")

def invert(network, unet, vae, text_encoder, tokenizer, prompt, noise_scheduler, epochs, image_path, mask_path, n_time_steps, device,latent_mask, wandb_name=None, weight_decay=1e-10, lr=1e-4, grad_clip=0.5):

    if latent_mask == None:
        latent_mask = torch.ones(network.vae.latent_dim).to(device)

    latent_mask = latent_mask.to(device)

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

    network.vae.eval()

    train_dataset = torchvision.datasets.ImageFolder(root=image_path, transform=image_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True) 
    print("len dataloader:", len(train_dataloader))

    optim = torch.optim.AdamW([network.z], lr=lr, weight_decay=weight_decay)  
    num_params = sum(param.numel() for param in network.parameters())
    print(f"Total parameters being optimized: {num_params:,}")

    wandb_toggle = False
    if wandb_name is not None:
        wandb.init(project=wandb_name['proj'], name=wandb_name['run'])
        wandb.config.update({
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "n_time_steps": n_time_steps
        })
        wandb_toggle = True
    
    unet.eval()
    for epoch in tqdm.tqdm(range(epochs)):
        epoch_loss = 0.0
        for batch, _ in train_dataloader:
            # Prepare inputs
            batch = batch.to(device).bfloat16()
            latents = vae.encode(batch).latent_dist.sample()
            latents = latents * 0.18215
            # Repeat latents n_time_steps times
            latents = latents.repeat(n_time_steps, 1, 1, 1)  # [n_time_steps, 4, 64, 64]
            noise = torch.randn_like(latents)  # Will automatically have the repeated shape
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (n_time_steps,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]  # [1, 77, 768]
            text_embeddings = text_embeddings.repeat(n_time_steps, 1, 1)  # [n_time_steps, 77, 768]
            
            # Forward pass with network
            with network:
                model_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                if mask_path:
                    loss = torch.nn.functional.mse_loss(mask * model_pred.float(), mask * noise.float(), reduction="mean")
                else:
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                optim.zero_grad()
                loss.backward()
                # weight or cancel gradients based on variance activity:
                with torch.no_grad():
                    network.z.grad.mul_(latent_mask)  # mask or weighted mask
                optim.step()
            
            epoch_loss += loss.item()
            
            if wandb_toggle:
                wandb.log({
                    'batch_loss': loss.item()
                })

            # Clear memory
            del model_pred, noisy_latents, latents, noise, text_embeddings
            torch.cuda.empty_cache()
        
        if wandb_toggle:
            wandb.log({
                'epoch_loss': epoch_loss / len(train_dataloader),
            })

    return network