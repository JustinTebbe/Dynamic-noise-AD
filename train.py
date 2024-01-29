import torch
import os
import torch.nn as nn
from forward_process import *
from dataset import *
from diffusers import AutoencoderKL
from torch.optim import Adam
from dataset import *
from noise import *
from visualize import show_tensor_image

from test import *
from loss import *
from optimizer import *
from sample import *






def trainer(model, constants_dict, ema_helper, config):
    optimizer = build_optimizer(model, config)
    if config.data.name == 'MVTec' or config.data.name == 'BTAD' or config.data.name == 'MTD' or config.data.name =='VisA':
        train_dataset = MVTecDataset(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=True,
        )
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.model.num_workers,
            drop_last=True,
        )
    if config.data.name == 'cifar10':
        trainloader, testloader = load_data(dataset_name='cifar10')

    if config.model.latent:
        
        
        if config.model.latent_backbone == "VAE":
            vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
            vae.to(config.model.device)
            vae.eval()
        else:
            print(f"error: backbone needs to be VAE")
        
    
        

    for epoch in range(config.model.epochs):
        for step, batch in enumerate(trainloader):
            
            t = torch.randint(0, config.model.trajectory_steps, (batch[0].shape[0],), device=config.model.device).long()
            optimizer.zero_grad()
            if config.model.latent:
                
                if config.model.latent_backbone == "VAE":     
                    features = vae.encode(batch[0].to(config.model.device)).latent_dist.sample() * 0.18215
                else:
                    print(f"error: backbone needs to be VAE")
                
             
                loss = get_loss(model, constants_dict, features, t, config)
            else:
                loss = get_loss(model, constants_dict, batch[0], t, config) 
  
            loss.backward()
            optimizer.step()
            
    
            if epoch % 3 == 0 and step == 0:
                print(f"Epoch {epoch} | Loss: {loss.item()}")
            if epoch % 25 == 0 and step ==0:
                if config.model.save_model:
                    if config.data.category:
                        model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category)
                    else:
                        model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir)
                    if not os.path.exists(model_save_dir):
                        os.mkdir(model_save_dir)
                    torch.save(model.state_dict(), os.path.join(model_save_dir, f"{config.model.latent_size}_{config.model.unet_channel}_{config.model.n_head}_{config.model.head_channel}_diffusers_unet_{str(epoch)}")) #config.model.checkpoint_name

                
    if config.model.save_model:
        if config.data.category:
            model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category)
        else:
            model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir)
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        torch.save(model.state_dict(), os.path.join(model_save_dir, f"{config.model.latent_size}_{config.model.unet_channel}_{config.model.n_head}_{config.model.head_channel}_diffusers_unet_{str(epoch)}")) #config.model.checkpoint_name
