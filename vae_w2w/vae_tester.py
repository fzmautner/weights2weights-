import torch
from PIL import Image
from diffusers import DiffusionPipeline
from peft import PeftModel
from peft.utils.save_and_load import load_peft_weights
import shutil
import os
from typing import List, Optional
import tempfile
from utils import unflatten

class VAESampleTester:
    def __init__(self, vae, weight_dimensions, base_model="stablediffusionapi/realistic-vision-v51", device="cuda:0"):
        self.vae = vae
        self.weight_dimensions = weight_dimensions
        self.device = device
        self.base_model = base_model
        self.pipe = None
        self.temp_dir = None
        
        # Default generation parameters
        self.default_params = {
            'prompt': "sks person",
            'negative_prompt': "low quality, blurry, unfinished",
            'guidance_scale': 3.0,
            'num_inference_steps': 50,
        }
        
        # Initialize pipeline
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Initialize the diffusion pipeline"""
        self.pipe = DiffusionPipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)

    def _create_temp_dir(self):
        """Create a temporary directory for weights"""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = tempfile.mkdtemp()
        return self.temp_dir

    def _reset_pipeline(self):
        """Reset the pipeline to its initial state"""
        del self.pipe
        torch.cuda.empty_cache()
        self._setup_pipeline()

    def generate_from_sample(self, 
                           sample_idx: int, 
                           vae_samples: torch.Tensor,
                           n_images: int = 10,
                           seed_start: int = 0,
                           **generation_params) -> List[Image.Image]:
        """
        Generate images from a specific VAE sample
        """
        # Reset pipeline for each new sample
        self._reset_pipeline()
        
        # Create temporary directory for weights
        temp_path = self._create_temp_dir()
        
        # Unflatten the selected sample
        unflatten(vae_samples[[sample_idx]], self.weight_dimensions, temp_path)
        
        # Load the weights into the pipeline
        self.pipe.unet = PeftModel.from_pretrained(
            self.pipe.unet, 
            f"{temp_path}/unet", 
            adapter_name="identity1"
        )
        adapters_weights = load_peft_weights(f"{temp_path}/unet", device=self.device)
        self.pipe.unet.load_state_dict(adapters_weights, strict=False)
        
        # Merge generation parameters
        params = self.default_params.copy()
        params.update(generation_params)
        
        # Generate images
        images = []
        for i in range(n_images):
            generator = torch.Generator(device=self.device)
            generator = generator.manual_seed(seed_start + i)
            
            latents = torch.randn(
                (1, self.pipe.unet.in_channels, 512 // 8, 512 // 8),
                generator=generator,
                device=self.device
            ).half()
            
            image = self.pipe(
                latents=latents,
                **params
            ).images[0]
            
            images.append(image)
        
        return images

    def create_image_grid(self, images: List[Image.Image], cols: int = 5) -> Image.Image:
        """Create a grid from the generated images"""
        w, h = 512, 512
        rows = (len(images) + cols - 1) // cols
        grid = Image.new('RGB', size=(cols * w, rows * h))
        
        for i, img in enumerate(images):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        
        return grid

    def cleanup(self):
        """Clean up temporary directory and pipeline"""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None
        if self.pipe:
            del self.pipe
            torch.cuda.empty_cache()

    def __del__(self):
        self.cleanup()