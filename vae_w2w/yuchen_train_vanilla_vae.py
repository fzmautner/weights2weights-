import sys
import os 
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
import time
import random
import argparse

from vanilla_vae import VanillaVAE
from yuchen_vae import YuchenVAE, YuchenDiscriminator

sys.path.append('../')
from lora_w2w import LoRAw2w
from diffusers import DiffusionPipeline
from peft import PeftModel
from peft.utils.save_and_load import load_peft_weights
from PIL import Image

from utils import unflatten

def init_training(cfg, args=None):
    
    # get export folder
    export_path = cfg.train.export_path if cfg.train.export_path else './outputs'
    if cfg.train.train_tag is None:
        cfg.train.train_tag = time.strftime("%Y%m%d_%H_%M_%S")
    export_path = os.path.join(export_path, cfg.train.train_tag)
    if os.path.exists(export_path):
        if args is not None and not args.overwrite:
            overwrite = input(f'Warning: export path {export_path} already exists. Exit?(y/n)')
            if overwrite.lower() == 'y':
                exit()
    else:
        os.makedirs(export_path)
        os.makedirs(os.path.join(export_path, 'images'))
        os.makedirs(os.path.join(export_path, 'checkpoints'))
    
    # set seed
    seed = cfg.train.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # init torch
    device = f'cuda:{cfg.train.gpu}'
    torch_device = torch.device(device)
    torch.cuda.set_device(cfg.train.gpu)
    torch.backends.cudnn.benchmark = False
    print(f'\nusing device: {device}\n')
    
    # init wandb
    wandb.init(project="w2w-proj", name=cfg.train.train_tag)
    
    # export config
    print(f'exporting to: {export_path}\n')
    with open(os.path.join(export_path, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    
    return torch_device, export_path

MIN = -0.094693
RANGE = 0.186591
def normalize_weights(weights):
    weights = (weights - MIN) / RANGE
    weights = weights * 2 - 1
    return weights
def denormalize_weights(weights):
    weights = (weights + 1) / 2
    weights = weights * RANGE + MIN
    return weights

def save_model_architecture(model, directory: str, model_name='model') -> None:
    """Save the model architecture to a `.txt` file."""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    message = f'Number of trainable / all parameters: {num_trainable_params} / {num_params}\n\n' + str(model)

    with open(os.path.join(directory, f'{model_name}.txt'), 'w') as f:
        f.write(message)

class w2wDataset(Dataset):
    def __init__(self, all_weights):
        self.all_weights = all_weights
        
    def __len__(self):
        return len(self.all_weights)
    
    def __getitem__(self, idx):
        return self.all_weights[idx]
    
def inference(
    weight, 
    weight_dimensions,
    export_path, 
    inference_num: 10,
    denormalize=False
):  
    if denormalize:
        weight = denormalize_weights(weight)
    if len(weight.shape) == 1:
        weight = weight.unsqueeze(0)
    # create unet weight
    unet_path = os.path.join(export_path, 'unet')
    if os.path.exists(unet_path):
        shutil.rmtree(unet_path)
    unflatten(weight, weight_dimensions, export_path)
    
    # load pipe
    device = weight.device
    pipe = DiffusionPipeline.from_pretrained(
        "stablediffusionapi/realistic-vision-v51", 
        torch_dtype=torch.float16,
        safety_checker = None,
        requires_safety_checker = False
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_path, adapter_name="identity1")
    adapters_weights1 = load_peft_weights(unet_path)
    pipe.unet.load_state_dict(adapters_weights1, strict = False)
    pipe.to(device)
    
    # generate
    prompts = ["sks person"] * inference_num
    negative_prompts = ["low quality, blurry, unfinished"] * inference_num
    generators = [torch.Generator(device=device).manual_seed(i) for i in range(inference_num)]
    guidance_scale = 3.0
    ddim_steps = 50
    images = pipe(
        prompts, 
        num_inference_steps=ddim_steps, 
        guidance_scale=guidance_scale, 
        negative_prompt=negative_prompts,
        generator=generators
    ).images

    ### display images
    w, h = 512,512
    grid = Image.new('RGB', size=(5*512, 2*512))
    grid_w, grid_h = grid.size
    for i, img in enumerate(images):
        grid.paste(img, box=(i%5*w, i//5*h))
    
    return images, grid

   

def main(cfg, args=None):
    
    train_cfg = cfg.train
    data_cfg = cfg.data
    model_cfg = cfg.model
    
    # init training
    device, export_path = init_training(cfg, args)

    # load data
    weights_path = data_cfg.weights_path
    all_weights = torch.load(weights_path, map_location=torch.device('cpu'))
    print(f"all_weights shape: {all_weights.shape}")
    # load weight dimensions
    weight_dimensions = torch.load(data_cfg.weight_dimensions_path)

    if data_cfg.normalize:
        print("Normalizing the weights")
        all_weights = normalize_weights(all_weights)

    all_weights = all_weights.to(device)

    tensor = all_weights[0]
    size_in_bytes = tensor.element_size() * tensor.numel()
    size_in_gb = size_in_bytes / (1024 ** 3)
    print(f"Size of the all_weights in GB: {size_in_gb * len(all_weights):.6f} GB")

    # Split the dataset
    
    train_ratio = train_cfg.train_ratio
    batch_size = train_cfg.batch_size
    train_size = int(train_ratio * len(all_weights))
    test_size = len(all_weights) - train_size
    
    full_dataset = w2wDataset(all_weights)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    in_dim = len(all_weights[0])

    # Initialize the model
    vae = YuchenVAE(
            input_dim=in_dim,
            latent_dim=model_cfg.vae.latent_dim,
            hidden_dim=model_cfg.vae.hidden_dim,
            blocks=model_cfg.vae.blocks
        ).to(device)
    save_model_architecture(vae, export_path, model_name='vae')
    use_discriminator = model_cfg.use_discriminator
    if use_discriminator:
        print("[WARNING] Using Discriminator")
        discriminator = YuchenDiscriminator(
                input_dim=in_dim,
                hidden_dim=model_cfg.discriminator.hidden_dim,
                blocks=model_cfg.discriminator.blocks
            ).to(device)
        save_model_architecture(discriminator, export_path, model_name='discriminator')
    else:
        print("[WARNING] Not using Discriminator")
    
    # Initialize the optimizer
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=train_cfg.lr * 10)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=train_cfg.lr)
    
    # Train!
    epochs = train_cfg.epochs
    for epoch in range(epochs):
        
        vae.train()
        bar = tqdm(train_loader, desc=f"Epoch[{epoch}/{epochs}]")
        for i, weights in enumerate(bar):
            
            # vae forward
            recon, mu, log_var = vae(weights)
            vae_losses = vae.loss(weights, recon, mu, log_var, beta=model_cfg.vae.kl_beta)
            vae_loss, vae_recon_loss, vae_kl_loss = vae_losses['loss'], vae_losses['recon_loss'], vae_losses['kl_loss']
            
            # get generator loss
            gen_loss = vae_loss
            if use_discriminator:
                g_loss, _ = discriminator(recon, label=True)
                gen_loss = gen_loss + model_cfg.discriminator.dis_beta * g_loss
            gen_loss = gen_loss / len(weights)
            vae_optimizer.zero_grad()
            gen_loss.backward()
            vae_optimizer.step()
            
            if use_discriminator:
                # get discriminator loss
                discriminator_losses = discriminator.get_discriminator_losses(weights, recon)
                dis_loss, real_loss, fake_loss = discriminator_losses['loss'], discriminator_losses['real_loss'], discriminator_losses['fake_loss']
                dis_loss = dis_loss / len(weights) * model_cfg.discriminator.dis_beta
                discriminator_optimizer.zero_grad()
                dis_loss.backward()
                discriminator_optimizer.step()
                
            # log
            loss_log = {
                'gen_loss': gen_loss.item(), 
                'recon_loss': vae_recon_loss.item(),
                'kl_loss': vae_kl_loss.item(), 
            }
            if use_discriminator:
                loss_log.update({
                    'g_loss': g_loss.item(),
                    'dis_loss': dis_loss.item(),
                    'real_loss': real_loss.item(),
                    'fake_loss': fake_loss.item()
                })
            bar.set_postfix(loss_log)
            wandb.log(loss_log)
            with open(os.path.join(export_path, 'log.txt'), 'a') as f:
                if use_discriminator:
                    f.write(f"Epoch {epoch:03d} Iter {i:03d} Gen Loss {gen_loss.item():.6f} G Loss {g_loss.item():.6f} Recon Loss {vae_recon_loss.item():.6f} KL Loss {vae_kl_loss.item():.6f} Dis Loss {dis_loss.item():.6f} Real Loss {real_loss.item():.6f} Fake Loss {fake_loss.item():.6f}\n")
                else:
                    f.write(f"Epoch {epoch:03d} Iter {i:03d} Gen Loss {gen_loss.item():.6f} Recon Loss {vae_recon_loss.item():.6f} KL Loss {vae_kl_loss.item():.6f}\n")

        # Evaluate 
        vae.eval()
        with torch.no_grad():
            test_loss = 0
            for i, weights in enumerate(test_loader):
                recon, mu, log_var = vae(weights)
                vae_losses = vae.loss(weights, recon, mu, log_var, beta=model_cfg.vae.kl_beta)
                vae_loss, vae_recon_loss, vae_kl_loss = vae_losses['loss'], vae_losses['recon_loss'], vae_losses['kl_loss']
                loss = vae_loss / len(weights)
                test_loss += loss.item()
            test_loss /= len(test_loader)
            print(f"Epoch[{epoch}/{epochs}] Test Loss: {test_loss:.6f}")
            wandb.log({"Test Loss": test_loss})
            with open(os.path.join(export_path, 'log.txt'), 'a') as f:
                f.write(f"Epoch {epoch:03d} Test Loss {test_loss:.6f}\n")
            
            if epoch != 0 and (epoch + 1) % train_cfg.inference_interval == 0:
                # inference
                print(f"Epoch[{epoch}/{epochs}] Inference")
                gt_inference_weight = all_weights[random.choice(range(len(all_weights)))].to(device)
                recon_inference_weight = vae(gt_inference_weight)[0]
                print(f"recon_loss: {torch.nn.functional.mse_loss(gt_inference_weight, recon_inference_weight).item()}")
                # inference ground truth weights
                gt_images, gt_grid = inference(
                    gt_inference_weight, 
                    weight_dimensions, 
                    export_path, 
                    inference_num=train_cfg.inference_num,
                    denormalize=data_cfg.normalize
                )
                # inference reconstructed weights
                recon_images, recon_grid = inference(
                    recon_inference_weight, 
                    weight_dimensions, 
                    export_path, 
                    inference_num=train_cfg.inference_num,
                    denormalize=data_cfg.normalize
                )
                # save grid
                gt_grid.save(os.path.join(export_path, 'images', f'epoch_{epoch:03d}_gt.png'))
                recon_grid.save(os.path.join(export_path, 'images', f'epoch_{epoch:03d}_recon.png'))
                # log images
                wandb.log({
                    "gt_images": [wandb.Image(img) for img in gt_images],
                    "recon_images": [wandb.Image(img) for img in recon_images]
                })
            
        if epoch != 0 and (epoch + 1) % train_cfg.save_interval == 0:
            print(f"Saving checkpoint at epoch {epoch}")
            checkpoint_path = os.path.join(export_path, 'checkpoints', f'epoch_{epoch:03d}.pt')
            checkpoint = {
                'vae_state_dict': vae.state_dict(),
                'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                'discriminator_state_dict': discriminator.state_dict() if use_discriminator else None,
                'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict() if use_discriminator else None,
                'epoch': epoch
            }
            torch.save(checkpoint, checkpoint_path)
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--tag", type=str)
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    if args.gpu is not None:
        cfg.train.gpu = args.gpu
    if args.tag is not None:
        cfg.train.train_tag = args.tag
    
    main(cfg, args)