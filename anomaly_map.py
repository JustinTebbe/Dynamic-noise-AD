import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
import torchvision
from torchvision.transforms import transforms
import math 
from utilities import *
from dataset import *
from visualize import *
from feature_extractor import *
import numpy as np


def recon_heat_map(output, target, config):
    sigma = 4
    kernel_size = 2 * int(4 * sigma + 0.5) +1
    ano_map = 0
    
    output = output.to(config.model.device)
    target = target.to(config.model.device)
    
    i_d = color_distance(output, target, config, config.data.image_size)
    ano_map += i_d
    ano_map = gaussian_blur2d(
        ano_map , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
        )
    ano_map = torch.sum(ano_map, dim=1).unsqueeze(1)
    
    return ano_map

def feature_heat_map(output,target,fe,config):
    sigma = 4
    kernel_size = 2 * int(4 * sigma + 0.5) +1
    anomaly_map = 0
    output = output.to(config.model.device)
    target = target.to(config.model.device)

    
    f_d = feature_distance_new((output),  (target), fe,config)
    f_d = torch.Tensor(f_d).to(config.model.device)

    anomaly_map += f_d
    
    anomaly_map = gaussian_blur2d(
        anomaly_map , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
        )
        
    anomaly_map = torch.sum(anomaly_map, dim=1).unsqueeze(1)
    

    return anomaly_map


def heatmap_latent(l1_latent,cos_list, config):
    sigma = 4
    kernel_size = 2 * int(4 * sigma + 0.5) +1
 
    heatmap_latent_list = []
    for i in range(len(l1_latent)):

        anomaly_map = config.model.anomap_weighting * l1_latent[i] +(1-config.model.anomap_weighting)*cos_list[i]
    
        anomaly_map = gaussian_blur2d(
            anomaly_map , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
            )
            
        anomaly_map = torch.sum(anomaly_map, dim=1).unsqueeze(1)
        heatmap_latent_list.append(anomaly_map)
        
    return heatmap_latent_list


def color_distance(image1, image2, config,out_size=256):
 

    transform = transforms.Compose([ 
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if config.model.latent:
        image1 = image1.to(config.model.device)
        image2 = image2.to(config.model.device)
        
        distance_map = torch.mean(torch.abs(image1 - image2), dim=1).unsqueeze(1)
        
        distance_map = F.interpolate(distance_map, size=out_size, mode='bilinear', align_corners=True)
    
        
    else:
        image1 = transform(image1)
        image2 = transform(image2)
        distance_map = torch.mean(torch.abs(image1 - image2), dim=1).unsqueeze(1)

    return distance_map


def feature_distance_new(output, target, FE, config):
    '''
    Feature distance between output and target
    '''
    FE.eval()
    transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / (2)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    output = output.to(config.model.device)
    target = target.to(config.model.device)
    target = transform(target)
    output = transform(output)
    inputs_features = FE(target)
    output_features = FE(output)
    out_size = config.data.image_size
    anomaly_map = torch.zeros([inputs_features[0].shape[0] ,1 ,out_size, out_size]).to(config.model.device)
    for i in range(len(inputs_features)):
    
        if i in config.model.anomap_excluded_layers:
            continue

        a_map = 1 - F.cosine_similarity(inputs_features[i], output_features[i])
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        anomaly_map += a_map
    return anomaly_map 
    

def calculate_min_max_of_tensors(tensors):
    # Use a list comprehension to get all min and max values across tensors
    min_values = [torch.min(tensor) for tensor in tensors]
    max_values = [torch.max(tensor) for tensor in tensors]
    
    # Determine the global min and max values
    min_value = torch.min(torch.tensor(min_values))
    max_value = torch.max(torch.tensor(max_values))
    
    return min_value, max_value

def scale_values_between_zero_and_one(tensors):
    min_value, max_value = calculate_min_max_of_tensors(tensors)
    
    # Use list comprehension and broadcasting to scale all tensors
    scaled_tensors = [(tensor - min_value) / (max_value - min_value) for tensor in tensors]
    
    return scaled_tensors
