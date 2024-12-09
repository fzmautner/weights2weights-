import torch
import torch.nn as nn
from typing import Optional, List
from safetensors.torch import save_file
import os

class LoRAVAEw2w(nn.Module):
    def __init__(
        self,
        vae: nn.Module,
        unet: nn.Module,
        device: str = "cuda",
        multiplier: float = 1.0,
        weight_dim_path=None,
    ) -> None:
        super().__init__()
        
        self.vae = vae
        self.vae.eval()
        self.unet = unet
        self.unet.eval()
        self.device = device
        self.multiplier = multiplier
        
        self.weight_dimensions = torch.load(weight_dim_path or "../files/weight_dimensions.pt")
        self.z = nn.Parameter(torch.randn(1, vae.latent_dim, device=device, dtype=torch.float32))
        
        # set the lora layers
        self.lora_layers = []
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ == "Attention":
                for child_name, child_module in module.named_modules():
                    if child_name.endswith(('to_q', 'to_k', 'to_v')):
                        key = f"lora_unet_{name}_{child_name}"
                        if key in self.weight_dimensions:
                            self.lora_layers.append((key, child_module))

    def get_lora_weights(self):
        """Decode z into LoRA weights while maintaining gradient graph"""
        decoded = self.vae.decode(self.z)
        weights_dict = {}
        counter = 0
        
        for key, _ in self.lora_layers:
            dims = self.weight_dimensions[key]
            weight_size = dims[0][0]
            shape = dims[1]
            weights_dict[key] = decoded[0, counter:counter+weight_size].reshape(shape)
            counter += weight_size
            
        return weights_dict

    def apply_lora(self, x, orig_module, lora_weight):
        """Apply LoRA transformation maintaining gradient graph"""
        requires_grad = x.requires_grad
        
        original_output = orig_module(x)
        if lora_weight.dtype != x.dtype:
            lora_weight = lora_weight.to(dtype=x.dtype)
        
        x_reshaped = x if x.requires_grad else x.detach().requires_grad_(True)
        lora_output = torch.matmul(lora_weight, x_reshaped.T).T * self.multiplier
        
        output = original_output + lora_output
        
        if requires_grad and not output.requires_grad:
            output.requires_grad_(True)
            
        return output

    def forward(self, x, timesteps, text_embeddings):

        weights_dict = self.get_lora_weights()
        
        original_forwards = {}
        for key, module in self.lora_layers:
            original_forwards[key] = module.forward
            lora_weight = weights_dict[key]
            
            # honestly Im not sure what this is doing
            def make_forward(orig_forward, weight):
                def new_forward(inp):
                    # Ensure input can receive gradients
                    inp = inp if inp.requires_grad else inp.detach().requires_grad_(True)
                    return self.apply_lora(inp, orig_forward, weight)
                return new_forward
            
            module.forward = make_forward(original_forwards[key], lora_weight)
        
        try:
            with torch.set_grad_enabled(True):
                output = self.unet(x, timesteps, text_embeddings)
                
                if not output.sample.requires_grad:
                    output.sample.requires_grad_(True)
                
                return output
            
        finally:
            for key, module in self.lora_layers:
                module.forward = original_forwards[key]

    def prepare_optimizer_params(self):
        """Verify z requires gradients and return it"""
        assert self.z.requires_grad, "Latent vector z doesn't require gradients!"
        return [self.z]