import torch
import torch.nn as nn
from tqdm import tqdm
from forward_process import *
from dataset import *
from dataset import *
import timm
import random
from torch import Tensor, nn
from typing import Callable, List, Tuple, Union
from unet import *
from omegaconf import OmegaConf
from sample import *
from visualize import *
from resnet import *
import torchvision.transforms as T
from diffusers import AutoencoderKL


#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

torch.manual_seed(42)

def build_model(config):
    unet = UNetModel(256, 64, dropout=0, n_heads=4 ,in_channels=config.data.fe_input_channel)
    return unet



def loss_fucntion(a, b, config):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))
    return loss


def roundup(x, n=10):
    res = math.ceil(x/n)*n
    if (x%n < n/2)and (x%n>0):
        res-=n
    return res
              

def Domain_adaptation(unet, feature_extractor, vae, config, fine_tune, constants_dict, dataloader,consistency_decoder):

    if fine_tune:    
        unet.eval()
        feature_extractor.train()
        
        for param in feature_extractor.parameters():
            param.requires_grad = True

        transform = transforms.Compose([
                    transforms.Lambda(lambda t: (t + 1) / (2)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        optimizer = torch.optim.AdamW(feature_extractor.parameters(),lr=config.model.DA_learning_rate)      
        for epoch in range(config.model.DA_epochs):
            for step, batch in enumerate(dataloader):
                
                with torch.no_grad():
                    if config.model.DA_rnd_step:
                        step_percentage = np.random.randint(1,11)
                        step_size = config.model.test_trajectoy_steps_DA / 10 * step_percentage
                        test_trajectoy_steps = torch.Tensor([step_size]).type(torch.int64).to(config.model.device)
                        step_size = roundup(step_size)
                        skip = int(max(step_size / 10, 1))
                        seq = range(0 , step_size, skip)
                    else:
                        
                        test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps_DA]).type(torch.int64).to(config.model.device)
                        seq = range(0 , config.model.test_trajectoy_steps_DA, config.model.skip_DA)
                        
                    at = compute_alpha(constants_dict["betas"], test_trajectoy_steps.long(),config)
                    
                    
                    target = batch[0].to(config.model.device)  
                    target_vae = vae.encode(target.to(config.model.device)).latent_dist.sample() * 0.18215   
                    if config.model.noise_sampling:
                        noise = torch.randn_like(target_vae).to(config.model.device)
                        
                        noisy_image = at.sqrt() * target_vae + (1- at).sqrt() * noise
                    else:
                        noisy_image = target_vae
                        if config.model.downscale_first:
                            noisy_image = noisy_image * at.sqrt()
                        
                    
                        
                    reconstructed, _ = DA_generalized_steps(target_vae, noisy_image, seq, unet, constants_dict["betas"], config, eta2=config.model.eta2 , eta3=0 , constants_dict=constants_dict ,eraly_stop = False)
                    data_reconstructed = reconstructed[-1].to(config.model.device)
                    del target_vae
                    del noisy_image
                    torch.cuda.empty_cache()
                    data_reconstructed = 1 / 0.18215 * data_reconstructed
                    if config.model.consistency_decoder:
                        data_reconstructed = consistency_decoder(data_reconstructed)
                    else:
                        data_reconstructed = vae.decode(data_reconstructed.to(config.model.device)).sample
                
                data_reconstructed = transform(data_reconstructed)
                reconst_fe = feature_extractor(data_reconstructed)
                
                del data_reconstructed
                torch.cuda.empty_cache()
                target = transform(target)
                target_fe = feature_extractor(target)

                loss = loss_fucntion(reconst_fe, target_fe, config)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch} | Loss: {loss.item()}")
            
            torch.save(feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,f'feature_recon_sim{epoch+1}'))

    else:
        checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,f'feature_recon_sim{config.model.DA_epochs}')) 
        feature_extractor.load_state_dict(checkpoint)  
        print("loaded fe recon sim")
    return feature_extractor


