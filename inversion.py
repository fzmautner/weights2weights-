import torch
import torchvision
import tqdm
import torchvision.transforms as transforms
from PIL import Image
import warnings
import wandb
warnings.filterwarnings("ignore")



### run inversion  (optimize PC coefficients) given single image
def invert(network, unet, vae, text_encoder, tokenizer, prompt, noise_scheduler, epochs, image_path, mask_path, device, wandb_name=None, weight_decay = 1e-10, lr=1e-1, num_samples=1, z_regularizer=False):

    if wandb_name is not None:
        proj = wandb_name['proj']
        run_name = wandb_name['run']
        wandb.init(project=proj, name=run_name)
        wandb.config.update({
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "epochs": epochs
        })
    ### load mask
    if mask_path: 
        mask = Image.open(mask_path)
        mask = transforms.Resize((64,64), interpolation=transforms.InterpolationMode.BILINEAR)(mask)
        mask = torchvision.transforms.functional.pil_to_tensor(mask).unsqueeze(0).to(device).bfloat16()
    else: 
        mask = torch.ones((1,1,64,64)).to(device).bfloat16()

    ### single image dataset
    image_transforms = transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                                                transforms.RandomCrop(512),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5], [0.5])])


    train_dataset = torchvision.datasets.ImageFolder(root=image_path, transform = image_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True) 

    ### optimizer 
    optim = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)    
    ### print optimizer size in GB
    optimizer_size_bytes = sum(param.numel() * param.element_size() for param in network.parameters()) * 2  # Adam stores 2 values per parameter
    optimizer_size_mb = optimizer_size_bytes / (1024**2)
    print(f"Optimizer size: {optimizer_size_mb:.5f} MB")

    ### training loop
    unet.train()
    for epoch in tqdm.tqdm(range(epochs)):
        epoch_loss = 0
        for batch,_ in train_dataloader:
            ### prepare inputs
            batch = batch.to(device).bfloat16()
            latents = vae.encode(batch).latent_dist.sample()
            latents = latents*0.18215
            del batch  # Free up memory

            losses = []
            for _ in range(num_samples):
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
                del text_input  # Free up memory

                ### loss + sgd step
                with network:
                    model_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                    loss = torch.nn.functional.mse_loss(mask*model_pred.float(), mask*noise.float(), reduction="mean")
                    losses.append(loss)
                
                # Clean up intermediate tensors
                del noise, timesteps, noisy_latents, text_embeddings, model_pred
            
            del latents  # Clean up latents after all samples
            
            diff_loss = torch.stack(losses).mean()
            z_loss = 0
            if z_regularizer:
                z_loss = torch.norm(network.z)

            epoch_loss += diff_loss.item() + z_loss.item()
            loss = torch.stack(losses).mean() + z_loss

            wandb.log({'loss':epoch_loss, 'denoising loss': diff_loss, 'z_loss': z_loss})
            optim.zero_grad()
            loss.backward()
            optim.step()

    ### return optimized network
    return network


