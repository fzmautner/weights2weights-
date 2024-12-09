import torch
import torchvision
import tqdm
import torchvision.transforms as transforms
from PIL import Image
import warnings
import wandb

warnings.filterwarnings("ignore")

def invert_single_image(network, unet, vae, text_encoder, tokenizer, prompt, noise_scheduler, epochs, image_path, mask_path, device, weight_decay=1e-10, lr=1e-1):
    # Initialize wandb
    wandb.init(
        project="w2w-proj",
        name=f"inversion_run1_{epochs}",
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "prompt": prompt
        }
    )

    # Prepare image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = transform(image).unsqueeze(0).to(device)
    
    # Prepare mask
    if mask_path:
        mask = Image.open(mask_path)
        mask = transforms.Resize((64,64))(mask)
        mask = torchvision.transforms.functional.pil_to_tensor(mask).unsqueeze(0).to(device)
    else:
        mask = torch.ones((1,1,64,64)).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(network.prepare_optimizer_params(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    for epoch in tqdm.tqdm(range(epochs)):
        with torch.no_grad():
            batch = image.to(dtype=torch.bfloat16)
            latents = vae.encode(batch).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, 
                                 truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        
        # Forward pass
        model_pred = network(noisy_latents, timesteps, text_embeddings).sample
        
        # Verify gradients are being tracked
        assert model_pred.requires_grad, "Model prediction doesn't require gradients!"
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(
            mask * model_pred.float(), 
            mask * noise.float()
        )
        
        # Verify loss requires gradients
        assert loss.requires_grad, "Loss doesn't require gradients!"
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics
        wandb.log({"loss": loss.item()})
    
    wandb.finish()
    return network