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
from torch.cuda.amp import autocast
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
        self.model_name = 'MyModel_v1'
        self.init_model = 'v1-5-pruned-emaonly.safetensors'
        self.models_dir = 'U:/Work/Stable Diffusion/stable-diffusion-webui/models/Stable-diffusion/'
        self.vae = 'U:/Work/Stable Diffusion/stable-diffusion-webui/models/VAE/vae-ft-mse-840000-ema-pruned.vae.pt'
        self.embeddings_dir = 'U:/Work/Stable Diffusion/stable-diffusion-webui/embeddings/'
        self.train_unet = True
        self.train_text = True
        self.lr_unet = 1.5e-6
        self.lr_text = 1e-6
        self.lr_ti = 3e-4
        self.doing_textual_inversion = False
        self.textual_inversion_token_ids = []
        self.clip_skip = 0
        self.precision_unet = torch.float32 # torch.float16, torch.bfloat16, torch.float32
        self.precision_text = torch.float32
        self.precision_vae = torch.float16 # 16 bit okay if using AMP autocast and not training the VAE?  
        self.optimizer = 'ADAMW' # 'ADAMW', 'ADAMW_8BIT'
        self.xformers = True
        #self.stochastic_rounding = True # not yet implemented
        self.frozen_embeddings = [ 49406, 49407 ] # 1.5 bos & eos
        self.epochs = 1000
        self.save_frequency_models = 1 # backup frequency
        self.save_frequency_embeddings = 3 # exports a new embedding file with a timestamp name
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
    log(f'Importing base model from {config.init_model}\n')
    unet, text_encoder = util.import_checkpoint(config.models_dir, config.init_model)

vae = AutoencoderKL.from_single_file(config.vae)

tokenizer = util.create_tokenizer()
scheduler = util.create_scheduler()


######################
# Dataset
######################

log(f':Loading dataset\n')

# ! custom DataLoader, will need to be replaced to run this code, and tokenization might need to happen in this code after the DataLoader returns captions (without taking tokenizer)
SuperDatasetCustom = SourceFileLoader("SuperDatasetCustom", "U:/Work/Stable Diffusion/CustomDataloader/SuperDatasetCustom.py").load_module().SuperDatasetCustom

dataset = SuperDatasetCustom(
    tokenizer1 = tokenizer,
    pad_tokens = True,
    max_token_length = tokenizer.model_max_length - 2 # for 1.5 BOS and EOS
)

log(f':Datset loaded with {len(dataset)} items\n')


######################
# Embeddings
######################

with torch.no_grad():
    embeddings_layer = text_encoder.get_input_embeddings()

    # old code from another project, hasn't been tested in this implementation yet
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
            
            # don't think this is used any more, allowed capping how many vectors import?
            if len(import_embedding) > 2:
                num_vecs = import_embedding[2]
            
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

if not config.train_unet:
    config.precision_unet = torch.float16

# if doing textual inversion, could maybe set all layers of TextEncoder except embeddings layer to fp16, but unsure if can iterate a transformers wrapper on the model
if not config.train_text and not config.doing_textual_inversion:
    config.precision_text = torch.float16

unet = unet.to(device, dtype=config.precision_unet)
text_encoder = text_encoder.to(device, dtype=config.precision_text)
vae = vae.to(device, dtype=config.precision_vae) 

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
# unsure if this else needed
#else:
#    unet.set_attn_processor(AttnProcessor())

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

#########################
# Training
#########################

def train_loop():
    global steps  # need this to modify a global variable
    
    save_frequency = config.save_frequency_models if not config.doing_textual_inversion else config.save_frequency_embeddings

    for epoch in range(1, config.epochs+1):        
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
                    
                    if optimizer_unet:
                        optimizer_unet.step()
                    
                    if optimizer_text:
                        optimizer_text.step()
                    
                    steps += 1
                    
                    pbar.set_description(f'epoch {epoch}, steps {epoch * i}, total steps {steps}')
                    pbar.update(1)
                
                except KeyboardInterrupt:
                    if not config.doing_textual_inversion:
                        print("\nTraining interrupted. Export checkpoint? (y/n): ", end='')
                        if input().lower() == 'y':
                            export_checkpoint()
                        
                        print("\nSave mid-epoch training?: ", end='')
                        if input().lower() == 'y':
                            backup()
                    
                    sys.exit(0)
        
        log(f'\Finished epoch {epoch} at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n', doPrint=False, doLog=True)
        
        if epoch % save_frequency == 0:
            backup()

# adapted from EveryDream2
def calculate_loss(images, tokens):

    with torch.no_grad():
        latents = vae.encode(images.to(device=device, dtype=vae.dtype)).latent_dist.sample()
        latents = latents * 0.18215
        
        latents.to(unet.dtype)
        
        noise = torch.randn_like(latents, dtype=unet.dtype, device=device)
        
        batch_size = latents.shape[0]
        
        timestep_end = 0
        timestep_start = scheduler.config.num_train_timesteps
        
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)
        timesteps = timesteps.long()
        
        cuda_caption = tokens.to(device)
    
    encoder_hidden_states = text_encoder(cuda_caption, output_hidden_states=True) # is outputting all hidden states inefficient?

    if config.clip_skip > 0:
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states.hidden_states[-config.clip_skip])
    else:
        encoder_hidden_states = encoder_hidden_states.last_hidden_state
    
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    
    del latents, cuda_caption
    
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    
    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
    
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


def backup()():
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
    util.export_checkpoint(config.models_dir, config.model_name, steps, unet, text_encoder, vae, config.save_vae, config.save_half)
    log("finished exporting checkpoint\n")
    
train_loop()