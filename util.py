import os
import glob
import copy
import numpy as np

import torch
from torch import optim
from torch.cuda.amp import GradScaler
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, EulerAncestralDiscreteScheduler
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

from convert_diff_to_sd import ConvertToCheckpoint

# From OneTrainer
def create_optimizer( optimizer_name, parameters, learning_rate, optimizer_states_path, stochastic_rounding ) -> torch.optim.Optimizer:
    optimizer = None
    
    match optimizer_name:
        case 'ADAMW':
            optimizer = torch.optim.AdamW(
                params = parameters,
                lr = learning_rate,
                betas = (0.9, 0.999),
                weight_decay = 0.01,
                eps = 1e-8,
                amsgrad = False,
                foreach = False,
                maximize = False,
                capturable = False,
                differentiable = False,
                fused = True, # may be worse in fp16 / bf16?
            )
            
            if stochastic_rounding:
                from bf16_stochastic_rounding import step_adamw
                optimizer.step = step_adamw.__get__(optimizer, torch.optim.AdamW)

        case 'ADAMW_8BIT':
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params = parameters,
                lr = learning_rate,
                #betas = (0.9, 0.999),
                weight_decay = 0.01,
                #eps = 1e-8,
                #min_8bit_size = 4096,
                #percentile_clipping = 100,
                #block_wise = True,
                #is_paged = False,
            )
        
        case 'ADAFACTOR':
            from transformers.optimization import Adafactor
            
            relative_step = False #True
            
            if relative_step:
                for parameter in parameters:
                    if isinstance(parameter, dict) and 'lr' in parameter:
                        parameter.pop('lr')

            optimizer = Adafactor(
                params=parameters,
                lr = None if relative_step == True else learning_rate,
                eps = (1e-30, 1e-3),
                clip_threshold = 1.0,
                decay_rate = -0.8,
                beta1 = None,
                weight_decay = 0.0,
                scale_parameter = False, #True,
                relative_step = relative_step,
                warmup_init = False,
            )
            
            if stochastic_rounding:
                from bf16_stochastic_rounding import step_adafactor
                optimizer.step = step_adafactor.__get__(optimizer, Adafactor)
    
        case "LION":
            import lion_pytorch as lp
            optimizer = lp.Lion(
                params = parameters,
                lr = learning_rate,
                betas=(0.9, 0.99),
                weight_decay = 0,
                use_triton = False,
            )
            
        case "LION_8BIT":
            import bitsandbytes as bnb
            optimizer = bnb.optim.Lion8bit(
                params = parameters,
                lr = learning_rate,
                weight_decay = 0,
                betas = (0.9, 0.999),
                min_8bit_size = 4096,
                percentile_clipping = 100,
                block_wise = True,
                is_paged = False,
            )
    
    # this may break if changed optimizer between training sessions
    if os.path.exists(optimizer_states_path):
        optimizer_states = torch.load(optimizer_states_path)
        optimizer.load_state_dict(optimizer_states)
    
    return optimizer


def load_checkpoint(models_dir, model_name):
    pipeline = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict = os.path.join(models_dir, model_name),
        #original_config_file = './v1-inference.yaml', # let it infer?
        num_in_channels = 4,
        scheduler_type = "ddim",
        load_safety_checker = False,
        from_safetensors = True,
    )
    
    return pipeline.unet, pipeline.text_encoder

def save_checkpoint(model_dir, model_name, steps, unet, text_encoder, vae, save_vae, half):
    filename = os.path.join(model_dir, f'{model_name}_{steps}.safetensors')
    ConvertToCheckpoint(filename, unet.state_dict(), text_encoder.state_dict(), vae.state_dict(), save_vae, half)

def create_tokenizer():
    return CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only = False)

def create_scheduler(name = "DDIM"):
    scheduler = None

    match name:
        case 'DDIM':
            scheduler = DDIMScheduler(
                beta_start = 0.00085, # 0.02
                beta_end = 0.0120, # 0.085
                beta_schedule = "scaled_linear",
                trained_betas=None,
                num_train_timesteps = 1000,
                steps_offset = 1,
                clip_sample = False,
                set_alpha_to_one = False,
                prediction_type = "epsilon",
            )
            
            # maybe not needed
            scheduler.register_to_config(clip_sample = False) # make sure scheduler works correctly with DDIM
    
    if not scheduler:
        print("Invalid scheduler name")
    
    return scheduler

def get_embedding_vectors(embeddings_dir, embedding_name):
    pattern = os.path.join(embeddings_dir, '**', f'{embedding_name}.pt')
    file_list = glob.glob(pattern, recursive=True)
    if not file_list:
        print(f'Embedding not found: {embedding_name}')
    
    filepath = file_list[0]
    data = torch.load(filepath, map_location="cpu")
    
    param_dict = data['string_to_param']
    param_dict = getattr(param_dict, '_parameters', param_dict)  # fix for torch 1.12.1 loading saved file from torch 1.11
    assert len(param_dict) == 1, 'embedding file has multiple terms in it'
    emb = next(iter(param_dict.items()))[1]
    
    vec = emb.detach().to("cuda", dtype=torch.float32)
    
    return vec

def save_embedding_vec(embeddings_dir, embedding_name, vec):
    filepath = os.path.join(embeddings_dir, f'{embedding_name}.pt')
    embedding_data = {
        "string_to_token": {"*": 265},
        "string_to_param": {"*": vec},
        "name": embedding_name,
        "step": 0,
        "sd_checkpoint": None,
        "sd_checkpoint_name": None,
    }
    
    torch.save(embedding_data, filepath)

# adapted from OneTrainer. Betas are the amount of noise added at a given timestep, alphas are the inverse, the amount of original image left at a given timestep.
def get_min_snr_values(noise_scheduler, device):
    betas = noise_scheduler.betas.to(device=device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0) # cumulative product is the total at a given timestep, i.e. [ 0, 0*1, 0*1*2, ...] (less and less of the image visible with each timestep)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    return { "sqrt_alphas_cumprod": sqrt_alphas_cumprod, "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod }
    
# adapted from OneTrainer
def min_snr_weights(timesteps, gamma, min_snr_values, device):
    all_snr = (min_snr_values['sqrt_alphas_cumprod'] / min_snr_values['sqrt_one_minus_alphas_cumprod']) ** 2
    all_snr = all_snr.to(device)
    snr = all_snr[timesteps]

    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    snr_weights = torch.div(min_snr_gamma, snr).float().to(device)

    return snr_weights
    
    
class Historical_Loss_Bucket:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.losses = np.array([0.0] * capacity)
        self.index = 0
        self.writes = 0
        self.percentiles = None
        
    def add_loss(self, loss):
        self.losses[self.index] = loss
        self.index = (self.index + 1) % self.capacity
        self.writes += 1
        self.percentiles = None

# tracks losses at various timestep ranges, returns 0/1 scalars for losses depending on whether they are performing much better than average for that timestep, indicting potential overfitting
class Historical_Loss_Cutoff:
    def __init__(self, num_buckets=10, bucket_capacity=100, num_timesteps=1000, cutoff_percentile = 10, cutoff_during_warmup = False):
        self.bucket_capacity = bucket_capacity
        self.cutoff_percentile = cutoff_percentile
        self.cutoff_during_warmup = cutoff_during_warmup
        
        self.bucket_range = num_timesteps / num_buckets
        self.buckets = [Historical_Loss_Bucket(capacity=bucket_capacity) for _ in range(num_buckets)]
   
    def add_losses(self, timesteps, losses):
        for i, timestep in enumerate(timesteps):
            loss = losses[i]
            bucket_index = min(int(timestep // self.bucket_range), len(self.buckets)-1) # if ever adjust timesteps to be able to do the last one, could cause an invalid index?
            self.buckets[bucket_index].add_loss(loss)
            
            # lazy solution to the fair distribution problem, just add to 2 buckets if near an edge
            start_of_bucket = bucket_index * self.bucket_range
            percent_in_bucket = ((timestep - start_of_bucket) / self.bucket_range) * 100
            
            if percent_in_bucket < 25 and bucket_index > 0:
                self.buckets[bucket_index - 1].add_loss(loss)
            elif percent_in_bucket > 75 and bucket_index < len(self.buckets) - 1:
                self.buckets[bucket_index + 1].add_loss(loss)
    
    def loss_scales(self, timesteps, losses):
        scales = []
        
        for i, timestep in enumerate(timesteps):
            bucket_index = min(int(timestep // self.bucket_range), len(self.buckets)-1) # if ever adjust timesteps to be able to do the last one, could cause an invalid index?
            bucket = self.buckets[ bucket_index ]
            loss = losses[i]
            
            if bucket.writes < self.bucket_capacity:
                scales.append(0 if self.cutoff_during_warmup else 1)
                continue
            
            if bucket.percentiles is None:
                sorted_losses = np.sort(bucket.losses)
                bucket.percentiles = np.percentile(sorted_losses, self.cutoff_percentile)
            
            scales.append(0 if loss < bucket.percentiles else 1)
        
        return np.array(scales)