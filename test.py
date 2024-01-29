from asyncio import constants
import torch
import pickle 
from unet import *
from utilities import *
from forward_process import *
from dataset import *
from visualize import *
from anomaly_map import *
from metrics import metric
from feature_extractor import *
import time
from datetime import timedelta
from diffusers import AutoencoderKL
from sklearn.neighbors import NearestNeighbors
from consistencydecoder import ConsistencyDecoder





class KNN:
    def __init__(self,config, k=5, num_bins=10):
        self.k = k
        self.config = config
        self.model = NearestNeighbors(n_neighbors=k, metric=config.model.KNN_metric)
        self.num_bins = num_bins
    def fit(self, X): # X entire trainingset
        X = X.detach().cpu().numpy()
        self.model.fit(X) # determining nearest neighbours
        
        distances, _ = self.model.kneighbors(X) # get distances of the neighbours
        # compute the average distance for each point
        avg_distances = distances.mean(axis=1)
    
        # define bins based on these average distances
    
        self.histogram, self.bin_edges = np.histogram(avg_distances, self.num_bins)

        
        print(f"bin edges: {self.bin_edges}")
        print(f"histogram: {self.histogram}")

    def transform(self, X):
        distances, indices = self.model.kneighbors(X)
        return distances, indices



def get_bins_and_mappings(knn, distances, indices):
    mappings = []
    keys = []
    for i in range(distances.shape[0]):
        avg_distance = np.mean(distances[i])  # average the distances
        
        
        
        
        bin_id = np.digitize(avg_distance, knn.bin_edges, right=True) -1 
        bin_id = min(bin_id, len(knn.bin_edges) - 2) + 1
        keys.append(bin_id)  # append the key to the list
        mapping = {bin_id: [ind.item() for ind in indices[i]]}
        mappings.append(mapping)
    return mappings, keys



def validate(unet, constants_dict, config):

    if config.data.name == 'BTAD' or config.data.name =='VisA' or config.data.name =='MVTec':
        
        test_dataset = MVTecDataset(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=False,
        )
        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size= config.data.batch_size,
            shuffle=False,
            num_workers= config.model.num_workers,
            drop_last=False,
        )

            
        train_dataset = MVTecDataset(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=True,
        )
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.data.DA_batch_size,
            #batch_size=30,
            shuffle=True,
            num_workers=config.model.num_workers,
            drop_last=False,
            ) 
    
    
    
    labels_list = []
    predictions= []
    anomaly_map_list = []
    GT_list = []
    reconstructed_list = []
    forward_list = []
    forward_list_orig = []
    l1_latent_list = []
    cos_dist_list = []
    step_list = []
    filename_list = []
    anomaly_map_recon_list = []
    anomaly_map_feature_list = []
    anomaly_map_latent_list = []


    if config.model.latent:
        
                    
        if config.model.latent_backbone == "VAE":
            
            
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
            vae.to(config.model.device)
            if config.model.consistency_decoder:
                consistency_decoder = ConsistencyDecoder(device=config.model.device)
            else:
                consistency_decoder = False
            vae.eval()
        else:
            print("Error backbone needs to be VAE")

        if config.model.dynamic_steps or (config.model.distance_metric_eval == "combined"):
        
            #FE backbone
            if config.model.fe_backbone == "wide_resnet50":
                feature_extractor = wide_resnet50_2(pretrained=True)[0]
            elif config.model.fe_backbone == "resnet34":
                feature_extractor = resnet34(pretrained=True)[0]
            elif config.model.fe_backbone == "resnet101":
                feature_extractor = resnet101(pretrained=True)[0]
            elif config.model.fe_backbone == "wide_resnet101":
                feature_extractor = wide_resnet101_2(pretrained=True)[0]
            else:
                print("error: no valid fe backbone selected")
            feature_extractor.to(config.model.device)
            feature_extractor = Domain_adaptation(unet, feature_extractor,vae, config, fine_tune=config.model.DA_fine_tune, constants_dict=constants_dict,dataloader=trainloader, consistency_decoder=consistency_decoder)   
            feature_extractor.eval()
            

            knn_transform = transforms.Compose([
                        transforms.Lambda(lambda t: (t + 1) / (2)),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

            knn = KNN(config=config,k=config.model.knn_k,num_bins=10)

            # We're going to stack the extracted features of the training data here
            train_stack = []
            
            del vae
            torch.cuda.empty_cache()
            
            for i, train_batch in enumerate(trainloader):
                
                train_batch = knn_transform(train_batch[0])
                
                train_batch = feature_extractor(train_batch.to(config.model.device))
                selected_features = [train_batch[i] for i in config.model.selected_features]
                common_size = (16,16)
                # Use adaptive pooling to resize feature maps to the common size
                adaptive_pool = nn.AdaptiveAvgPool2d(common_size)
                pooled_features = [adaptive_pool(feature_map) for feature_map in selected_features]

                # Flatten each feature map in the batch and concatenate along the feature dimension
                flattened_features = [pf.view(pf.size(0), -1) for pf in pooled_features]  # Flatten each feature map
                train_batch = torch.cat(flattened_features, dim=1)  # Concatenate along the feature dimension
                
                train_stack.append(train_batch.detach().cpu())
                torch.cuda.empty_cache()
                
            # fit KNN model on training data
            knn.fit(torch.cat(train_stack, dim=0))

            knnPickle = open(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,f"knn_{config.model.knn_k}_{config.model.DA_epochs}"), 'wb') 
            del train_stack
            del trainloader
            torch.cuda.empty_cache()
            # source, destination 
            pickle.dump(knn, knnPickle)
            # close the file
            knnPickle.close()
    

      

    
    def roundup(x, n=10):
        res = np.ceil(x/n)*n
        mask = np.logical_and(x % n < n/2, x % n > 0)
        res[mask] -= n
        return res
    


    if config.model.latent_backbone == "VAE":
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae.to(config.model.device)

        vae.eval()     

    #eval    
    if config.data.name == 'BTAD' or config.data.name == "VisA" or config.data.name == "MVTec":
        
        with torch.no_grad():
            start = time.time()
            for step, (data, targets, labels, filename) in enumerate(testloader):
                
                
                data_placeholder = data
                
                if config.model.dynamic_steps:

                    #extract features and peform KNN on training set to determine noise level

                    test_batch = data
                    test_batch = knn_transform(test_batch)
                    test_batch = feature_extractor(test_batch.to(config.model.device))
                    selected_features = [test_batch[i] for i in config.model.selected_features]
                    adaptive_pool = nn.AdaptiveAvgPool2d(common_size)
                    pooled_features = [adaptive_pool(feature_map) for feature_map in selected_features]

                    flattened_features = [pf.view(pf.size(0), -1) for pf in pooled_features] 
                    test_batch = torch.cat(flattened_features, dim=1)

                    test_batch = test_batch.detach().cpu().numpy()
                    torch.cuda.empty_cache()

                    distances, indices = knn.transform(test_batch)
                
                    mappings, keys = get_bins_and_mappings(knn, distances, indices)

                    mapping_int = int(list(set(mappings[0].keys()))[0])

                    bin_ids_array = np.array(keys)

                    # Compute step_sizes directly using element-wise operations
                    step_sizes_array = np.maximum(bin_ids_array, 2) / 10 * config.model.test_trajectoy_steps
                    step_size = roundup(step_sizes_array)

                    # Compute skips directly using element-wise operations
                    skip = np.maximum(step_size / 10, 1).astype(int)
            
                    step_list.extend(step_size)
                    
                    
                else:
                    step_size = config.model.test_trajectoy_steps
                    skip = config.model.skip
                    
                filename_list.append(filename)
                forward_list_orig.append(data)
                forward_list.append(data)
                if config.model.latent:
                    data = data.to(config.model.device)
                    data = vae.encode(data).latent_dist.sample() * 0.18215    
                
            
                test_trajectoy_steps = torch.Tensor([step_size]).type(torch.int64).to(config.model.device)[0]
                
                
                at = compute_alpha2(constants_dict['betas'], test_trajectoy_steps.long(),config)

                if config.model.noise_sampling:
                    noise = torch.randn_like(data).to(config.model.device)
                    noisy_image = at.sqrt() * data + (1- at).sqrt() * noise
                else:
                    noisy_image = data
                    if config.model.downscale_first:
                        noisy_image = noisy_image * at.sqrt()
                if config.model.dynamic_steps:
                    seq = [torch.arange(0, end, step).to(test_trajectoy_steps.device) for end, step in zip(test_trajectoy_steps, skip)]
                else:
                    seq = range(0 , test_trajectoy_steps, skip)
                

                
                if config.model.dynamic_steps:            
        
                    reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, unet, constants_dict['betas'], config, eta2=config.model.eta2 , eta3=0 , constants_dict=constants_dict ,eraly_stop = False)

                else:
                    reconstructed, rec_x0 = DA_generalized_steps(data, noisy_image, seq, unet, constants_dict['betas'], config, eta2=config.model.eta2 , eta3=0 , constants_dict=constants_dict ,eraly_stop = False)

                data_reconstructed = reconstructed[-1].to(config.model.device)
                
                
                if config.model.latent_backbone == "VAE":
                    #reconstruct image from latent space
                    reconstructed = 1 / 0.18215 * data_reconstructed
                    if config.model.consistency_decoder:
                        reconstructed = consistency_decoder(reconstructed)
                    else:
                        reconstructed = vae.decode(reconstructed.to(config.model.device)).sample
                else:
                    print(f"error: backbone needs to be VAE")
                l1_latent = color_distance(data_reconstructed, data, config, out_size=config.data.image_size)
                cos_dist = feature_distance_new(reconstructed, data_placeholder, feature_extractor,config)
                
                anomaly_map_latent = recon_heat_map(data_reconstructed,data,config)
                anomaly_map_feature = feature_heat_map(reconstructed,data_placeholder,feature_extractor,config)
                    
                
                l1_latent_list.append(l1_latent)
                cos_dist_list.append(cos_dist)
                
                anomaly_map_latent_list.append(anomaly_map_latent)
                anomaly_map_feature_list.append(anomaly_map_feature)
                    
                GT_list.append(targets)
                reconstructed_list.append(reconstructed)

                for label in labels:
                    labels_list.append(0 if label == 'good' else 1)
                
                
            

    l1_latent_normalized_list = scale_values_between_zero_and_one(l1_latent_list)
    cos_dist_normalized_list = scale_values_between_zero_and_one(cos_dist_list)

    heatmap_latent_list = heatmap_latent(l1_latent_normalized_list,cos_dist_normalized_list, config)

    concat_heatmap = torch.cat(heatmap_latent_list, dim=0) 
    predictions_normalized = []
    for heatmap in concat_heatmap:
        predictions_normalized.append(torch.max(heatmap).item() )
        

    threshold = metric(labels_list, predictions_normalized, heatmap_latent_list, GT_list, config)
        
    
    end = time.time()
    print('Inference time is ', str(timedelta(seconds=end - start)))
    print('threshold: ', threshold)

    
 
    reconstructed_list = torch.cat(reconstructed_list, dim=0)
    forward_list = torch.cat(forward_list, dim=0)
    if config.model.latent:
        forward_list_orig = torch.cat(forward_list_orig, dim=0)
      
        filename_list = [item for tup in filename_list for item in tup]
        anomaly_map_latent_list = torch.cat(anomaly_map_latent_list, dim=0)
        anomaly_map_feature_list = torch.cat(anomaly_map_feature_list, dim=0)
        
        
    GT_list = torch.cat(GT_list, dim=0)
    
    pred_mask = (concat_heatmap> threshold).float()
    visualize(forward_list, reconstructed_list, GT_list, pred_mask, concat_heatmap, config.data.category, config, forward_list_orig, step_list,filename_list, anomaly_map_recon_list, anomaly_map_latent_list, anomaly_map_feature_list)
