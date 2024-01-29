import torch
import numpy as np
import random



def get_noise(x, config, seed=42):
    if config.model.noise == 'Gaussian':
        noise = torch.randn_like(x).to(config.model.device)
        return noise


    else:
        print('noise is not selected correctly. Default is Gaussian noise')
        noise = torch.randn_like(x)
        return noise
