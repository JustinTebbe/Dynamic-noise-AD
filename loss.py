import torch
import os
import torch.nn as nn
from forward_process import *
from noise import *



def get_loss(model, constant_dict, x_0, t, config):

    x_0 = x_0.to(config.model.device)
    b = constant_dict['betas'].to(config.model.device)
    e = torch.randn_like(x_0, device = x_0.device)
    at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = at.sqrt() * x_0 + (1- at).sqrt() * e
    output = model(x, t.float())

    return F.mse_loss(e, output)


