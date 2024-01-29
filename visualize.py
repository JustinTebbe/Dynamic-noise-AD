import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np
import torch
import os
from forward_process import *
from dataset import *
from sample import *

from noise import *


def visualize_reconstructed(input, data,s):

    fig, axs = plt.subplots(int(len(data)/5),6)
    row = 0
    col = 1
    axs[0,0].imshow(show_tensor_image(input))
    axs[0, 0].get_xaxis().set_visible(False)
    axs[0, 0].get_yaxis().set_visible(False)
    axs[0,0].set_title('input')
    for i, img in enumerate(data):
        axs[row, col].imshow(show_tensor_image(img))
        axs[row, col].get_xaxis().set_visible(False)
        axs[row, col].get_yaxis().set_visible(False)
        axs[row, col].set_title(str(i))
        col += 1
        if col == 6:
            row += 1
            col = 0
    col = 6
    row = int(len(data)/5)
    remain = col * row - len(data) -1
    for j in range(remain):
        col -= 1
        axs[row-1, col].remove()
        axs[row-1, col].get_xaxis().set_visible(False)
        axs[row-1, col].get_yaxis().set_visible(False)
        
    
        
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    k = 0

    while os.path.exists(f'results/reconstructed{k}{s}.png'):
        k += 1
    plt.savefig(f'results/reconstructed{k}{s}.png')
    plt.close()



def visualize(image, noisy_image, GT, pred_mask, anomaly_map, category, config, orig_img, step_list, filename_list, anomaly_map_recon_list, anomaly_map_latent_list, anomaly_map_feature_list) :
    for idx, img in enumerate(image):
        
        if config.model.visual_all:
            if config.model.dynamic_steps:
                plt.imsave('results/{}/{}sample{}_{}_save_all_clear_T_hat{}_T_max{}.png'.format(category,category,idx,config.model.skip,step_list[idx],config.model.test_trajectoy_steps), show_tensor_image(orig_img[idx]))
                plt.imsave('results/{}/{}sample{}_{}_save_all_recon_T_hat{}_T_max{}.png'.format(category,category,idx,config.model.skip,step_list[idx],config.model.test_trajectoy_steps), show_tensor_image(noisy_image[idx]))
                plt.imsave('results/{}/{}sample{}_{}_save_all_GT_mask_T_hat{}_T_max{}.png'.format(category,category,idx,config.model.skip,step_list[idx],config.model.test_trajectoy_steps), show_tensor_mask(GT[idx],config))
                plt.imsave('results/{}/{}sample{}_{}_save_all_pred_mask_T_hat{}_T_max{}.png'.format(category,category,idx,config.model.skip,step_list[idx],config.model.test_trajectoy_steps), show_tensor_mask(pred_mask[idx],config))
            else:
                if config.model.noise_sampling:
                    plt.imsave('results/{}/{}sample{}_{}_save_all_clear_T_max{}_noise.png'.format(category,category,idx,config.model.skip,config.model.test_trajectoy_steps), show_tensor_image(orig_img[idx]))
                    plt.imsave('results/{}/{}sample{}_{}_save_all_recon_T_max{}_noise.png'.format(category,category,idx,config.model.skip,config.model.test_trajectoy_steps), show_tensor_image(noisy_image[idx]))
                    plt.imsave('results/{}/{}sample{}_{}_save_all_GT_mask_T_max{}_noise.png'.format(category,category,idx,config.model.skip,config.model.test_trajectoy_steps), show_tensor_mask(GT[idx],config))
                    plt.imsave('results/{}/{}sample{}_{}_save_all_pred_mask_T_max{}_noise.png'.format(category,category,idx,config.model.skip,config.model.test_trajectoy_steps), show_tensor_mask(pred_mask[idx],config))
                    
                else:
                    plt.imsave('results/{}/{}sample{}_{}_save_all_clear_T_max{}.png'.format(category,category,idx,config.model.skip,config.model.test_trajectoy_steps), show_tensor_image(orig_img[idx]))
                    plt.imsave('results/{}/{}sample{}_{}_save_all_recon_T_max{}.png'.format(category,category,idx,config.model.skip,config.model.test_trajectoy_steps), show_tensor_image(noisy_image[idx]))
                    plt.imsave('results/{}/{}sample{}_{}_save_all_GT_mask_T_max{}.png'.format(category,category,idx,config.model.skip,config.model.test_trajectoy_steps), show_tensor_mask(GT[idx],config))
                    plt.imsave('results/{}/{}sample{}_{}_save_all_pred_mask_T_max{}.png'.format(category,category,idx,config.model.skip,config.model.test_trajectoy_steps), show_tensor_mask(pred_mask[idx],config))

            

        
        else:
            plt.figure(figsize=(11,11))
            
            
            plt.subplot(1, 2, 1).axis('off')
            plt.subplot(1, 2, 2).axis('off')
            plt.subplot(1, 2, 1)
            plt.imshow(show_tensor_image(image[idx]))
            plt.title(f'clear image {filename_list[idx]}')

            plt.subplot(1, 2, 2)

            plt.imshow(show_tensor_image(noisy_image[idx]))
            plt.title('reconstructed image')
        if config.model.dynamic_steps:
            if int(step_list[idx]) >= 7:
                
                plt.savefig('results/{}/{}sample{}_dynamic_big_step{}.png'.format(category,category,idx,config.model.skip))
            else:
                
                plt.savefig('results/{}/{}sample{}_dynamic_step{}.png'.format(category,category,idx,config.model.skip))
        else:
            plt.savefig('results/{}/{}sample{}_step{}.png'.format(category,category,idx,config.model.skip))
        plt.close()

       
        plt.figure(figsize=(15,11))
        plt.subplot(1, 5, 1).axis('off')
        plt.subplot(1, 5, 2).axis('off')
        plt.subplot(1, 5, 3).axis('off')
        plt.subplot(1, 5, 4).axis('off')
        plt.subplot(1, 5, 5).axis('off')

        plt.subplot(1, 5, 1)
        plt.imshow(show_tensor_mask(GT[idx],config))
        plt.title('ground truth')


        plt.subplot(1, 5, 2)
        plt.imshow(show_tensor_mask(pred_mask[idx], config))
        plt.title('normal' if torch.max(pred_mask[idx]) == 0 else 'abnormal', color="g" if torch.max(pred_mask[idx]) == 0 else "r")

        plt.subplot(1, 5, 3)
        plt.imshow(show_tensor_image(anomaly_map[idx]))
        plt.title('heat map combined')
        
        
        plt.subplot(1, 5, 4)
        plt.imshow(show_tensor_image(anomaly_map_latent_list[idx]))
        plt.title('heat map latent')
        
        plt.subplot(1, 5, 5)
        plt.imshow(show_tensor_image(anomaly_map_feature_list[idx]))
        plt.title('heat map feature')
            
        
        if config.model.dynamic_steps:
                if int(step_list[idx]) >= 7:
                    
                    plt.savefig('results/{}/{}sample{}_dynamic_big_heatmap_step{}.png'.format(category,category,idx,config.model.skip))
                else:
                   
                    plt.savefig('results/{}/{}sample{}_dynamic_heatmap_step{}.png'.format(category,category,idx,config.model.skip))
        else:
            if config.model.noise_sampling:
                plt.savefig('results/{}/{}sample{}_heatmap_step{}_noise.png'.format(category,category,idx,config.model.skip))
            else:
                plt.savefig('results/{}/{}sample{}_heatmap_step{}.png'.format(category,category,idx,config.model.skip))
       
        plt.close()



def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    return reverse_transforms(image)

def show_tensor_mask(image, config):
    if config.model.visual_all:
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t.squeeze(2)),
            transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        ])
    else:
        reverse_transforms = transforms.Compose([
        
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
       
            transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    
        ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    return reverse_transforms(image)
        

