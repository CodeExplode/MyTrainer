import os
import glob
import copy

import torch
from torch import optim
#from torch.cuda.amp import GradScaler
from transformers import CLIPTokenizer
from diffusers import DDIMScheduler, AutoencoderKL
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
                betas = (0.9, 0.999),
                weight_decay = 0.01,
                eps = 1e-8,
                min_8bit_size = 4096,
                percentile_clipping = 100,
                block_wise = True,
                is_paged = False,
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
    
    # this may break if changed optimizer between training sessions
    if os.path.exists(optimizer_states_path):
        optimizer_states = torch.load(optimizer_states_path)
        optimizer.load_state_dict(optimizer_states)
    
    return optimizer

def load_checkpoint(models_dir, model_name):
    pipeline = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict = os.path.join(models_dir, model_name),
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

def create_scheduler():
    scheduler = DDIMScheduler(
        beta_start = 0.00085,
        beta_end = 0.0120,
        beta_schedule = "scaled_linear",
        trained_betas=None,
        num_train_timesteps = 1000,
        steps_offset = 1,
        clip_sample = False,
        set_alpha_to_one = False,
        prediction_type = "epsilon",
    )
    
    # from another repo, maybe not needed
    scheduler.register_to_config(clip_sample = False) # make sure scheduler works correctly with DDIM
    
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