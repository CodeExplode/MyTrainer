import os
import glob
import copy

import torch
from torch import optim
from torch.cuda.amp import GradScaler
from transformers import CLIPTokenizer
from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

from convert_diff_to_sd import ConvertToCheckpoint

# From OneTrainer
def create_optimizer( optimizer_name, parameters, learning_rate, optimizer_states_path ) -> torch.optim.Optimizer:
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
    
    # this may break if changed optimizer between training sessions
    if os.path.exists(optimizer_states_path):
        optimizer_states = torch.load(optimizer_states_path)
        optimizer.load_state_dict(optimizer_states)
    
    return optimizer

'''def create_grad_scaler(save_path):
    scaler = GradScaler(
            enabled=True,
            init_scale=2**17.5,
            growth_factor=2,
            backoff_factor=1.0/2,
            growth_interval=25,
        )

    if os.path.exists(save_path):
        scaler_state = torch.load(save_path)
        scaler.load_state_dict(scaler_state)
    
    return scaler'''

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
    
    # maybe not needed, was from another repo
    scheduler.register_to_config(clip_sample = False)
    
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