import os
import sys
import glob
import gc
import numpy as np
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
        self.models_dir = 'U:/Work/Stable Diffusion/_models/models'
        self.vae = 'U:/Work/Stable Diffusion/_models/vae/vae-ft-mse-840000-ema-pruned.vae.pt'
        self.embeddings_dir = 'U:/Work/Stable Diffusion/_models/embeddings/'
        self.train_unet = True
        self.train_text = True
        self.lr_unet = 3.5e-6 # 1.5e-6 
        self.lr_text = 3.5e-6 # 1.5e-6
        self.lr_ti = 3e-4
        self.doing_textual_inversion = False
        self.textual_inversion_token_ids = []
        self.max_batch_size = 4 # dataloader can return less if resolution buckets don't evenly divide by batch size
        self.clip_skip = 0
        self.precision_model = torch.float32 # float32 or bfloat16 - seemingly should only ever cast the model down if can't even load it in fp32, better to just use fp32 and an 8bit optimizer otherwise
        self.precision_training = torch.bfloat16 # float32 or bfloat16 - some parts of the process are autocast with this precision, not sure how helpful it actually is for speed or vram
        self.precision_inference = torch.bfloat16 # torch.float16, torch.bfloat16, torch.float32
        self.optimizer = 'ADAMW_8BIT' # 'ADAMW', 'ADAMW_8BIT', 'ADAFACTOR', "LION", "LION_8BIT" - 8bit versions seem just as good with huge savings, if changed between runs on same model will cause crash on loading optimizer state?
        self.set_grads_to_none = True # slightly save vram, may be incompatible with stochastic_rounding implementation because a None gradient and a Zero gradient may be handled differently
        self.stochastic_rounding = False # simulate higher precision param updates if model loaded in bf16
        self.min_snr_gamma = 0 # 0 disabled, 1 for SD?, 20 max?
        self.discard_best_losses = True # try to prevent overfitting by scaling the losses of the items which are performing the best to 0
        self.xformers = False # seems worse for speed & vram on pytorch 2 / rtx 3090
        self.unet_only_train_attention = True
        self.frozen_embeddings = [ 49406, 49407 ] # 1.5 bos & eos - try to prevent the padding tokens from being trained so much on shorter prompts
        self.epochs = 1000
        self.save_frequency_models = 1 # backup frequency
        self.save_frequency_embeddings = 3 # export frequency
        self.samples_per_epoch = 2
        self.package_vae = True # whether to include VAE in checkpoint
        self.package_half = True # save final checkpoint in half precision, no real downsides during inference
config = Config()

######################
# Logging
######################

model_cache_path = f'./cache/{config.model_name}/'
unet_path = os.path.join(model_cache_path, 'unet')
text_path = os.path.join(model_cache_path, 'text_encoder')
unet_optimizer_path = os.path.join(unet_path, 'optimizer_states.pth')
text_optimizer_path = os.path.join(text_path, 'optimizer_states.pth')
steps_path = os.path.join(model_cache_path, 'tracker.txt')
samples_path = os.path.join(model_cache_path, 'samples')

resuming = os.path.exists(model_cache_path)

if not resuming:
    os.makedirs(model_cache_path)
    os.makedirs(samples_path)

log_file = open(os.path.join(model_cache_path, 'log.txt'), 'a')

def log(message, doPrint=True, doLog=True):
    if doPrint:
        print(f'{message}\n')
    if doLog:
        log_file.write(f'{message}\n')

log(f'Started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

######################
# Load the models
######################

device = 'cuda'
tracker = { "session": 0, "total": 0, "epochs": 0, "loss_min": 0.0, "loss_median": 0.0, "loss_max": 0.0 }

if resuming:
    log(f'Resuming from previous model')
    unet = UNet2DConditionModel.from_pretrained( unet_path, use_safetensors=True )
    text_encoder = CLIPTextModel.from_pretrained( text_path, use_safetensors=True )
    
    with open(steps_path, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1].strip()
        tracker["total"] = int(last_line)
else:
    log(f'Creating new model from {config.init_model}')
    unet, text_encoder = util.load_checkpoint(config.models_dir, config.init_model)

vae = AutoencoderKL.from_single_file(config.vae)

tokenizer = util.create_tokenizer()
scheduler = util.create_scheduler()
scheduler.set_timesteps(20) # for previews

######################
# Dataset
######################

### will need to replace this with something which implements DataSet to use this code

log(f'Loading dataset')

SuperDatasetCustom = SourceFileLoader("SuperDatasetCustom", "U:/Work/Stable Diffusion/CustomDataloader/SuperDatasetCustom.py").load_module().SuperDatasetCustom

dataset = SuperDatasetCustom(
    tokenizer1 = tokenizer,
    pad_tokens = True,
    max_token_length = tokenizer.model_max_length - 2, # for BOS and EOS
    batch_size = config.max_batch_size,
    timesteps = 1000, # not sure if this needs to be per SD version
    keep_images_in_memory = True,
)

log(f'Datset loaded with {len(dataset.images)} items')

######################
# Embeddings
######################

with torch.no_grad():
    embeddings_layer = text_encoder.text_model.embeddings.token_embedding

    if dataset.doing_textual_inversion:
        config.train_unet = False
        config.train_text = False
        config.doing_textual_inversion = True
        
        for idx, inversion in enumerate(dataset.textual_inversions):
            token_ids = tokenizer.encode(dataset.textual_inversions_mapped[idx], add_special_tokens=False)
            for token_id in token_ids:
                if token_id not in config.textual_inversion_token_ids:
                    config.textual_inversion_token_ids.append(token_id)
        
        log(f'Doing textual inversion for {dataset.textual_inversions} with tokens {config.textual_inversion_token_ids}')
        
        frozen_embeds_mask = torch.ones(len(tokenizer), dtype=torch.bool).to(device)
        frozen_embeds_mask[config.textual_inversion_token_ids] = False
        
        min_embedding = embeddings_layer.weight.min(0).values
        max_embedding = embeddings_layer.weight.max(0).values
    elif config.frozen_embeddings:
        frozen_embeds_mask = torch.zeros(len(tokenizer), dtype=torch.bool).to(device)
        frozen_embeds_mask[config.frozen_embeddings] = True
    else:
        frozen_embeds_mask = None
    
    if frozen_embeds_mask is not None:
        log(f'Total embeddings: {len(frozen_embeds_mask)}, total frozen embeddings: {torch.sum(frozen_embeds_mask)}' )

    if len(dataset.import_embeddings) > 0:
        log(f'Inserting embeddings {dataset.import_embeddings}', doPrint=False, doLog=True)
        
        for import_embedding in dataset.import_embeddings:
            emb_text = import_embedding[0]
            emb_name = import_embedding[1]
            
            log(f'Looking for embedding {emb_name}')
            emb_vec = util.get_embedding_vectors(config.embeddings_dir, emb_name)
            emb_token_ids = tokenizer.encode(emb_text, add_special_tokens=False)
            
            num_vecs = len(emb_token_ids)
            
            log(f'Importing embedding {emb_name} over {num_vecs} tokens for phrase {emb_text}')
            
            for idx, emb_token_id in enumerate(emb_token_ids):
                if idx < num_vecs:
                    embeddings_layer.weight[emb_token_id] = emb_vec[idx]

if len(dataset.images) == 0:
    log("Empty Dataset")
    log_file.close()
    sys.exit(1)


######################
# Record Config
######################

log('Config:', doPrint=False, doLog=True)
for key, value in config.__dict__.items():
    log(f'{key}: {value}', doPrint=False, doLog=True)

log_file.flush()

#########################
# Unfreezing
#########################

log('Unfreezing Models')

unet.requires_grad_(config.train_unet)
text_encoder.requires_grad_(config.train_text)
vae.requires_grad_(False)

text_encoder.text_model.embeddings.position_embedding.requires_grad_(False) # these are determined by some wizard science and probably shouldn't be changed

if config.doing_textual_inversion:
    embeddings_layer.requires_grad_(True)

unet.train(config.train_unet)
text_encoder.train(config.train_text or config.doing_textual_inversion)
vae.train(False)

precision_unet = config.precision_model if config.train_unet else config.precision_inference
precision_text = config.precision_model if config.train_text or config.doing_textual_inversion else config.precision_inference

unet = unet.to(device, dtype=precision_unet)
text_encoder = text_encoder.to(device, dtype=precision_text)
vae = vae.to(device, dtype=config.precision_inference)

#########################
# Xformers
#########################

if config.xformers and is_xformers_available():
    log('Applying xformers')
    try:
        unet.set_attn_processor(XFormersAttnProcessor())
        vae.enable_xformers_memory_efficient_attention()
    except Exception as e:
        log(f'Could not enable xformers.\n{e}')
#else:
#    unet.set_attn_processor(AttnProcessor())


#########################
# Experiments
#########################

if config.train_unet and config.unet_only_train_attention and not config.doing_textual_inversion:
    unet_params_trained = 0
    unet_total_params = 0
    
    for i, (name, param) in enumerate(unet.named_parameters()):
        if 'attentions' in name:
            param.requires_grad = True
            unet_params_trained += 1
        else:
            param.requires_grad = False
        unet_total_params += 1
    
    log(f'Training {unet_params_trained} / {unet_total_params} unet components')

if config.min_snr_gamma:
    min_snr_values = util.get_min_snr_values(scheduler, device)
    
    # print out the timestep where SNR loss weights become 1.0
    test_snr_timesteps = torch.arange(1, scheduler.config.num_train_timesteps, device=device)
    test_snr_weights = util.min_snr_weights(test_snr_timesteps, config.min_snr_gamma, min_snr_values, device)
    test_snr_indices = torch.nonzero(test_snr_weights == 1.0, as_tuple=False)
    test_snr_inflection = test_snr_timesteps[test_snr_indices[0].item()]
    log(f'SNR with gamma {config.min_snr_gamma} has full loss weights after timestep {test_snr_inflection}')

if config.discard_best_losses:
    loss_adjuster = util.Historical_Loss_Cutoff(num_buckets=10, bucket_capacity=min(200, len(dataset.images)), num_timesteps=scheduler.config.num_train_timesteps, cutoff_percentile = 10, cutoff_during_warmup = resuming)


#########################
# Optimizer
#########################

log(f'Creating optimizers')

optimizer_unet = None
optimizer_text = None

if not config.precision_model == torch.bfloat16:
    config.stochastic_rounding = False

if config.train_unet:
    params = [param for param in unet.parameters() if param.requires_grad]
    optimizer_unet = util.create_optimizer( config.optimizer, params, config.lr_unet, unet_optimizer_path, config.stochastic_rounding )

if config.train_text:
    params = [param for param in text_encoder.parameters() if param.requires_grad]
    optimizer_text = util.create_optimizer( config.optimizer, params, config.lr_text, text_optimizer_path, config.stochastic_rounding )

if config.doing_textual_inversion:
    optimizer_text = util.create_optimizer( config.optimizer, embeddings_layer.parameters(), config.lr_ti, None, config.stochastic_rounding )


##########################
# Training
#########################

def clear_gc():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

def train_loop():
    log(f'Starting training at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log(f'Num Images: {len(dataset.images)}')

    save_frequency = config.save_frequency_models if not config.doing_textual_inversion else config.save_frequency_embeddings
    running_loss = 0.0
    running_loss_smooth = 0.001

    generate_samples(config.samples_per_epoch) # make sure the model is actually working before starting

    for epoch in range(1, config.epochs+1):
        clear_gc()
        
        dataset.start_new_epoch()
        
        with tqdm(total=len(dataset), desc=f'Epoch {epoch}') as progress_bar:
            for i, batch in enumerate(dataset):
                # batch .images: list of image tensors
                #       .tokens: all prompts in the batch encoded at once in a list, i.e. tokenizer( captions, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids (maybe move that here?)
                #       .noise_limits: optional list of noise limits per image, experimental
                
                try:
                    if optimizer_unet:
                        optimizer_unet.zero_grad(set_to_none=config.set_grads_to_none)
                    
                    if optimizer_text:
                        optimizer_text.zero_grad(set_to_none=config.set_grads_to_none)
                    
                    preview_vae = (i < 2)
                    
                    ### pretend collate_fn:
                    batch["images"] = torch.stack(batch["images"]).to(memory_format=torch.contiguous_format).float()
                    #batch["loss_weights"] = torch.tensor(batch["loss_weights"], dtype=torch.float32, device=device)
                    batch_size = batch["images"].shape[0]
                    ###
                    
                    loss, loss_scalars = calculate_loss(batch["images"], batch["tokens"], batch["noise_limits"], preview_vae)
                    
                    loss.backward()
                    
                    if frozen_embeds_mask is not None:
                        with torch.no_grad():
                            embeddings_layer.weight.grad[ frozen_embeds_mask ] = 0
                    
                    if config.train_unet:
                        unet_norm_pre_clip = nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                    if config.train_text:
                        text_norm_pre_clip = nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
                    
                    lr_scaling = batch_size / config.max_batch_size
                    
                    if optimizer_unet:
                        for param_group in optimizer_unet.param_groups:
                            param_group['lr'] = lr_scaling * config.lr_unet
                    
                        optimizer_unet.step()
                    
                    if optimizer_text:
                        for param_group in optimizer_text.param_groups:
                            param_group['lr'] = lr_scaling * (config.lr_text if not config.doing_textual_inversion else config.lr_ti)
                        
                        optimizer_text.step()
                    
                    tracker["session"] += batch_size
                    tracker["total"] += batch_size
                    
                    running_loss = running_loss_smooth * loss.item() + (1 - running_loss_smooth) * running_loss if tracker["session"] > 0 else loss.item()
                    
                    avg_loss_scalar = np.mean(loss_scalars) if config.discard_best_losses else 0
                    
                    progress_bar.set_description(f'epoch {epoch}, steps {tracker["session"]}, total steps {tracker["total"]}, loss: {loss:.4f}, running loss:{running_loss:.4f}, loss scalars: {avg_loss_scalar:.3f}, grad norms: {unet_norm_pre_clip:.4f}, {text_norm_pre_clip:.4f}')
                    progress_bar.update(1)
                
                except KeyboardInterrupt:
                    if not config.doing_textual_inversion:
                        print("\nTraining interrupted. Export checkpoint? (y/n): ", end='')
                        if input().lower() == 'y':
                            export_checkpoint()
                        
                        print("\nSave mid-epoch training?: ", end='')
                        if input().lower() == 'y':
                            save()
                    
                    return
        
        clear_gc()
        
        log(f'Finished epoch {epoch} at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', doPrint=False, doLog=True)
        log_file.flush()
        
        if epoch % save_frequency == 0:
            save()
        
        generate_samples(config.samples_per_epoch)

def calculate_loss(images, tokens, noise_limits=None, preview_vae = False):
    with torch.no_grad():
        with autocast(dtype = config.precision_inference): # dtype = config.precision_inference
            latents = vae.encode(images.to(device=device, dtype=vae.dtype)).latent_dist.sample()
        
        if preview_vae:
            decoded_images = vae.decode(latents.to(vae.dtype)).sample
            decoded_images = (decoded_images + 1) * 0.5
            decoded_images = decoded_images.clamp(0, 1)
            transformPil = transforms.ToPILImage()
            
            for i, image_tensor in enumerate(decoded_images):
                image = transformPil(image_tensor.cpu())  # Move tensor to CPU for PIL conversion
                filename = os.path.join(samples_path, f'VAE_preview_step_{tracker["total"]}_{i}_{datetime.now().strftime("%Y-%m-%d %H_%M_%S")}.png')
                image.save(filename)
        
        latents = latents * 0.18215
        latents = latents.to(unet.dtype)
        
        noise = torch.randn_like(latents, dtype=unet.dtype, device=device)
        
        batch_size = latents.shape[0]
        
        # in theory should be adding +1 to the upper limit ("Common Diffusion Noise Schedules and Sample Steps are Flawed")
        if noise_limits is None:
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)
        else:
            timesteps_list = []
            for i in range(batch_size):
                noise_min, noise_max = noise_limits[i]
                timestep = torch.randint(noise_min, noise_max, (1,), device=device)
                timesteps_list.append(timestep)
            
            timesteps = torch.cat(timesteps_list)
        
        timesteps = timesteps.long()
        tokens = tokens.to(device=device) # don't cast, needs to be long
    
    text_encoder_output = text_encoder(tokens, return_dict=True, output_hidden_states=True)
    final_layer_norm = text_encoder.text_model.final_layer_norm
    embeddings = final_layer_norm( text_encoder_output.hidden_states[ - max(1, config.clip_skip) ] )
    embeddings.to(device=device, dtype=unet.dtype)
    
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    
    del latents
    
    with autocast(dtype = config.precision_training): #dtype = precision_unet
        model_pred = unet(noisy_latents, timesteps, embeddings).sample
    
    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="none")
    loss = loss.mean([1, 2, 3]) # loss per item, from [batch, channels, height, width] to [batch]
    
    if config.min_snr_gamma:
        snr_weights = util.min_snr_weights(timesteps, config.min_snr_gamma, min_snr_values, device)
        loss *= snr_weights
    
    if config.discard_best_losses:
        loss_per_item = loss.detach().cpu().numpy()
        loss_scalars = loss_adjuster.loss_scales(timesteps, loss_per_item)
        loss_adjuster.add_losses(timesteps, loss_per_item)
        loss = loss * torch.from_numpy(loss_scalars).to(device=device)
    
    loss = loss.mean()
    
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
                reg_strength = 0.01 #0.000001
                loss += distribution_loss * reg_strength
                
                log(f'distribution loss: {distribution_loss.item()}, old loss: {old_loss}, new loss: {loss.item()}')
    
    return loss, loss_scalars


def save():
    log(f'saving backup at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', doPrint = not config.doing_textual_inversion, doLog=True)
    
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
            steps_total = tracker["total"]
            file.write(f"{steps_total}\n")

def export_checkpoint():
    log("exporting checkpoint")
    util.save_checkpoint(config.models_dir, config.model_name, tracker["total"], unet, text_encoder, vae, config.package_vae, config.package_half)
    log("finished exporting checkpoint")

def generate_samples(n):
    if not config.samples_per_epoch:
        return

    log(f'generating {n} samples')
    
    clear_gc()

    with torch.no_grad():
        unet.eval()
        text_encoder.eval()
        
        guidance_scale = 7.5
        
        generator = torch.manual_seed(0)
        
        for i in range(n):
            prompt, resolution = dataset.get_random_prompt_and_resolution()
            w, h = resolution
            
            tokens = tokenizer( [ prompt ], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            uncond_tokens = tokenizer( [""], padding="max_length", max_length=tokens.input_ids.shape[-1], return_tensors="pt")
            prompt_embeddings = text_encoder(tokens.input_ids.to(device))[0]
            uncond_embeddings = text_encoder(uncond_tokens.input_ids.to(device))[0]   
            embeddings = torch.cat([uncond_embeddings, prompt_embeddings])
            
            latents = torch.randn( (1, unet.config.in_channels, h // 8, w // 8), generator=generator, )
            latents = latents.to(device, dtype=unet.dtype)
            
            for t in scheduler.timesteps:
                latent_model_input = torch.cat([latents] * 2) # join the latents when doing classifier-free guidance to avoid doing two forward passes
                
                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
                
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeddings).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample
                
            latents = 1 / 0.18215 * latents
            
            decoded_images = vae.decode(latents.to(vae.dtype)).sample
            decoded_images = ((decoded_images + 1) * 0.5).clamp(0, 1)
            transformPil = transforms.ToPILImage()
            image = transformPil(decoded_images[0].cpu())
            prompt = prompt[:50] if len(prompt) > 50 else prompt
            filename = os.path.join(samples_path, f'sample_step_{tracker["total"]}_{i}_{prompt}.png')
            image.save(filename)
            
    unet.train(config.train_unet)
    text_encoder.train(config.train_text or config.doing_textual_inversion)

train_loop()
log_file.close()