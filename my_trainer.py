import os
import sys
import glob
import gc
from enum import Enum
from datetime import datetime
from importlib.machinery import SourceFileLoader

import torch
from torch import nn, optim
from torch.nn import Parameter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import XFormersAttnProcessor #, AttnProcessor2_0, AttnProcessor, 
from diffusers.utils import is_xformers_available
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from tqdm import tqdm

import util

######################
# config
######################

class Config:
    def __init__(self):
        self.model_name = 'my_model'
        self.init_model = 'v1-5-pruned-emaonly.safetensors'
        self.models_dir = 'U:/Work/Stable Diffusion/stable-diffusion-webui/models/Stable-diffusion/'
        self.vae = 'U:/Work/Stable Diffusion/stable-diffusion-webui/models/VAE/vae-ft-mse-840000-ema-pruned.vae.pt'
        self.embeddings_dir = 'U:/Work/Stable Diffusion/stable-diffusion-webui/embeddings/'
        self.train_unet = True
        self.train_text = True
        self.lr_unet = 1.5e-6
        self.lr_text = 1.5e-6
        self.lr_ti = 3e-4
        self.doing_textual_inversion = False
        self.textual_inversion_token_ids = []
        self.clip_skip = 0
        self.precision_training = torch.bfloat16 # torch.float32, torch.bfloat16
        self.precision_inference = torch.bfloat16 # torch.float16, torch.bfloat16, torch.float32
        self.stochastic_rounding = True # only used in bf16 training precision
        self.optimizer = 'ADAMW' # 'ADAMW', 'ADAMW_8BIT'- if changed between runs on same model will cause crash on loading optimizer state?
        self.xformers = True
        self.frozen_embeddings = [ 49406, 49407 ] # 1.5 bos & eos
        self.epochs = 1000
        self.save_frequency_models = 1
        self.save_frequency_embeddings = 3
        self.save_vae = False # whether to package VAE in checkpoint
        self.save_half = True # save final checkpoint in half precision
config = Config()

######################
# Logging
######################

model_cache_path = f'./cache/{config.model_name}/'
unet_path = os.path.join(model_cache_path, 'unet')
text_path = os.path.join(model_cache_path, 'text_encoder')
unet_optimizer_path = os.path.join(unet_path, 'optimizer_states.pth')
text_optimizer_path = os.path.join(text_path, 'optimizer_states.pth')
steps_path = os.path.join(model_cache_path, 'steps.txt')

resuming = os.path.exists(model_cache_path)

if not resuming:
    os.makedirs(model_cache_path)

log_file = open(os.path.join(model_cache_path, 'log.txt'), 'a')

def log(message, doPrint=True, doLog=True):
    if doPrint:
        print(message)
    if doLog:
        log_file.write(message)

log(f'Starting training at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')    

######################
# Load the models
######################

device = 'cuda'
steps = 0

if resuming:
    log(f'Resuming from previous model\n')
    unet = UNet2DConditionModel.from_pretrained( unet_path, use_safetensors=True )
    text_encoder = CLIPTextModel.from_pretrained( text_path, use_safetensors=True )
    
    with open(steps_path, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1].strip()
        steps = int(last_line)
else:
    log(f'Creating new model from {config.init_model}\n')
    unet, text_encoder = util.load_checkpoint(config.models_dir, config.init_model)

vae = AutoencoderKL.from_single_file(config.vae)

tokenizer = util.create_tokenizer()
scheduler = util.create_scheduler()

######################
# Dataset
######################

log(f'Loading dataset\n')

SuperDatasetCustom = SourceFileLoader("SuperDatasetCustom", "U:/Work/Stable Diffusion/CustomDataloader/SuperDatasetCustom.py").load_module().SuperDatasetCustom

dataset = SuperDatasetCustom(
    tokenizer1 = tokenizer,
    pad_tokens = True,
    max_token_length = tokenizer.model_max_length - 2, # for BOS and EOS
    timesteps = 1000,
    preload_all_images = True,
)

log(f':Datset loaded with {len(dataset)} items\n')

######################
# Embeddings
######################

with torch.no_grad():
    embeddings_layer = text_encoder.get_input_embeddings()

    if dataset.doing_textual_inversion:
        config.train_unet = False
        config.train_text = False
        config.doing_textual_inversion = True
        
        for idx, inversion in enumerate(dataset.textual_inversions):
            token_ids = tokenizer.encode(dataset.textual_inversions_mapped[idx], add_special_tokens=False)
            for token_id in token_ids:
                if token_id not in config.textual_inversion_token_ids:
                    config.textual_inversion_token_ids.append(token_id)
        
        log(f'Doing textual inversion for {dataset.textual_inversions} with tokens {config.textual_inversion_token_ids}\n')
        
        frozen_embeds_mask = torch.ones(len(tokenizer), dtype=torch.bool).to(device)
        frozen_embeds_mask[config.textual_inversion_token_ids] = False
        
        min_embedding = embeddings_layer.weight.min(0).values
        max_embedding = embeddings_layer.weight.max(0).values
    elif config.frozen_embeddings:
        frozen_embeds_mask = torch.zeros(len(tokenizer), dtype=torch.bool).to(device)
        frozen_embeds_mask[config.frozen_embeddings] = True
    else:
        frozen_embeds_mask = None

    log(f'Total embeddings: {len(frozen_embeds_mask)}, total frozen embeddings: {torch.sum(frozen_embeds_mask)}\n' )

    if len(dataset.import_embeddings) > 0:
        log(f'Inserting embeddings {dataset.import_embeddings}\n', doPrint=False, doLog=True)
        
        for import_embedding in dataset.import_embeddings:
            emb_text = import_embedding[0]
            emb_name = import_embedding[1]
            
            log(f'Looking for embedding {emb_name}')
            emb_vec = util.get_embedding_vectors(config.embeddings_dir, emb_name)
            emb_token_ids = tokenizer.encode(emb_text, add_special_tokens=False)
            
            num_vecs = len(emb_token_ids)
            
            log(f'Importing embedding {emb_name} over {num_vecs} tokens for phrase {emb_text}\n')
            
            for idx, emb_token_id in enumerate(emb_token_ids):
                if idx < num_vecs:
                    embeddings_layer.weight[emb_token_id] = emb_vec[idx]

if len(dataset) == 0:
    log("Empty Dataset")
    sys.exit(1)

#########################
# Unfreezing
#########################

log('Unfreezing Models\n')

unet.requires_grad_(config.train_unet)
text_encoder.requires_grad_(config.train_text)
vae.requires_grad_(False)

if config.doing_textual_inversion:
    embeddings_layer.requires_grad_(True)

unet.train(config.train_unet)
text_encoder.train(config.train_text or config.doing_textual_inversion)
vae.train(False)

precision_unet = config.precision_training if config.train_unet else config.precision_inference
precision_text = config.precision_training if config.train_text or config.doing_textual_inversion else config.precision_inference

unet = unet.to(device, dtype=precision_unet)
text_encoder = text_encoder.to(device, dtype=precision_text)
vae = vae.to(device, dtype=config.precision_inference)

#########################
# Xformers
#########################

if config.xformers and is_xformers_available():
    log('Applying xformers\n')
    try:
        unet.set_attn_processor(XFormersAttnProcessor())
        vae.enable_xformers_memory_efficient_attention()
    except Exception as e:
        log(f'Could not enable xformers.\n{e}\n')

#########################
# Optimizer
#########################

log(f'Creating optimizers\n')

optimizer_unet = None
optimizer_text = None

if config.train_unet:
    optimizer_unet = util.create_optimizer( config.optimizer, unet.parameters(), config.lr_unet, unet_optimizer_path )

if config.train_text:
    optimizer_text = util.create_optimizer( config.optimizer, text_encoder.parameters(), config.lr_text, text_optimizer_path )

if config.doing_textual_inversion:
    optimizer_text = util.create_optimizer( config.optimizer, embeddings_layer.parameters(), config.lr_ti, None )

if config.precision_training == torch.bfloat16 and config.stochastic_rounding:
    from adamw_extensions import step_adamw
    
    if optimizer_unet:
        optimizer_unet.step = step_adamw.__get__(optimizer_unet, torch.optim.AdamW)
    if optimizer_text:
        optimizer_text.step = step_adamw.__get__(optimizer_text, torch.optim.AdamW)


##########################
# Training
#########################

def train_loop():
    global steps  # Declare steps as global to modify the global variable
    
    save_frequency = config.save_frequency_models if not config.doing_textual_inversion else config.save_frequency_embeddings
    running_loss = 0.0
    running_loss_smooth = 0.01

    for epoch in range(1, config.epochs+1):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        
        with tqdm(total=len(dataset), desc=f'Epoch {epoch}') as pbar:
            for i, batch in enumerate(dataset):
                try:
                    if optimizer_unet:
                        optimizer_unet.zero_grad()
                    
                    if optimizer_text:
                        optimizer_text.zero_grad()
                    
                    loss = calculate_loss(batch["images"], batch["input_ids"])
                    
                    loss.backward()
                    
                    if frozen_embeds_mask is not None:
                        with torch.no_grad():
                            embeddings_layer.weight.grad[ frozen_embeds_mask ] = 0
                    
                    if config.train_unet:
                        nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                    if config.train_text:
                        nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
                    
                    if optimizer_unet:
                        optimizer_unet.step()
                    
                    if optimizer_text:
                        optimizer_text.step()
                    
                    steps += 1
                    
                    session_steps = (epoch-1) * len(dataset) + i
                    running_loss = running_loss_smooth * loss.item() + (1 - running_loss_smooth) * running_loss if session_steps > 0 else loss.item()
                    
                    pbar.set_description(f'epoch {epoch}, steps {session_steps}, total steps {steps}, loss: {loss:.4f}, running loss:{running_loss:.4f}')
                    pbar.update(1)
                    
                except KeyboardInterrupt:
                    if not config.doing_textual_inversion:
                        print("\nTraining interrupted. Export checkpoint? (y/n): ", end='')
                        if input().lower() == 'y':
                            export_checkpoint()
                        
                        print("\nSave mid-epoch training?: ", end='')
                        if input().lower() == 'y':
                            save()
                    
                    sys.exit(0)
        
        log(f'\Finished epoch {epoch} at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n', doPrint=False, doLog=True)
        
        if epoch % save_frequency == 0:
            save()

# adapted from EveryDream2
def calculate_loss(images, tokens):
    with torch.no_grad():
        latents = vae.encode(images.to(device=device, dtype=vae.dtype)).latent_dist.sample()
        
        latents = latents * 0.18215
        latents = latents.to(unet.dtype)
        
        noise = torch.randn_like(latents, dtype=unet.dtype, device=device)
        
        batch_size = latents.shape[0]
        
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)
        timesteps = timesteps.long()
        
        tokens = tokens.to(device=device) # don't cast, needs to be int or long
    
    text_encoder_output = text_encoder(tokens, return_dict=True, output_hidden_states=True)
    embeddings = text_encoder.text_model.final_layer_norm( text_encoder_output.hidden_states[-(1 + config.clip_skip)] )
    embeddings.to(device=device, dtype=unet.dtype)
    
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    
    del latents
    
    model_pred = unet(noisy_latents, timesteps, embeddings).sample
    loss = torch.nn.functional.mse_loss(model_pred.to(dtype=torch.float32), noise.to(dtype=torch.float32), reduction="mean")
    
    if config.doing_textual_inversion:
        distribution_loss = torch.tensor(0.0, device=device)
        
        ids_used = tokens[0]
        distribution_loss_tokens = 0
        
        for idx, inversion in enumerate(dataset.textual_inversions):
            # TODO: should pre-cache these and move to device
            inversion_token_ids = tokenizer.encode(dataset.textual_inversions_mapped[idx], add_special_tokens=False)
            
            if not any(token in ids_used for token in inversion_token_ids):
                continue
            
            distribution_loss_tokens += len(inversion_token_ids)
            
            ti_weights = embeddings_layer.weight[inversion_token_ids]
            out_of_bounds_loss = (ti_weights < min_embedding).float() * (min_embedding - ti_weights) + (ti_weights > max_embedding).float() * (ti_weights - max_embedding)
            distribution_loss += out_of_bounds_loss.sum() #.item()?
        
        if distribution_loss_tokens > 0:
            distribution_loss /= distribution_loss_tokens
            
            if torch.abs(distribution_loss) > 1e-7:
                old_loss = loss.item()
                reg_strength = 0.01
                loss += distribution_loss * reg_strength
                
                log(f'distribution loss: {distribution_loss.item()}, old loss: {old_loss}, new loss: {loss.item()}')
    
    return loss


def save():
    log(f'saving backup at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n', doPrint = not config.doing_textual_inversion, doLog=True)
    
    if config.train_unet or (not resuming and not config.doing_textual_inversion):
        unet.save_pretrained( unet_path, safe_serialization=True )
        if optimizer_unet:
            torch.save(optimizer_unet.state_dict(), unet_optimizer_path)
    
    if config.train_text or (not resuming and not config.doing_textual_inversion):
        text_encoder.save_pretrained( text_path, safe_serialization=True )
        if optimizer_text:
            torch.save(optimizer_text.state_dict(), text_optimizer_path)
    
    if config.doing_textual_inversion:
         for idx, inversion in enumerate(dataset.textual_inversions):
            inversion_token_ids = tokenizer.encode(dataset.textual_inversions_mapped[idx], add_special_tokens=False) # could be cached
            ti_weights = embeddings_layer.weight[inversion_token_ids]
            
            timestring = datetime.now().strftime("%d%m%Y_%H%M%S")
            embedding_name = inversion.replace(' ', '_') + "_" + timestring
            util.save_embedding_vec(util.embeddings_dir, embedding_name, ti_weights)
    else:
        with open(steps_path, 'a') as file:
            file.write(f"{steps}\n")

def export_checkpoint():
    log("exporting checkpoint\n")
    util.save_checkpoint(config.models_dir, config.model_name, steps, unet, text_encoder, vae, config.save_vae, config.save_half)
    log("finished exporting checkpoint\n")
    
train_loop()