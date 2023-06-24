from pepflow.model.model import BackboneModel
from pepflow.model.dynamics import DynamicsRotamer, DynamicsHydrogen
from pepflow.model.model_configs import config_rotamer, config_hydrogen,\
    config_backbone_bert, config_backbone
from pepflow.model.ema import ExponentialMovingAverage
from pepflow.model import sampling
from pepflow.model.sampling import TimeOutException
from pepflow.data.dataset import FragmentDatasetRotamer, FragmentDatasetHydrogen,\
    FragmentDatasetBackbone, PeptideTestDataset, PEDDataset, MDDataset
from pepflow.utils.dataset_utils import collate, collate_multiple_coords
from pepflow.utils import pdb_utils
from pepflow.utils.constants import residue_mapping
import torch
import argparse
import os
import numpy as np
import sys

def to_cuda(features):
    
    for feature in features:
        
        if "atom_names" not in feature and "cyclic_bond_indices" not in feature:
            
            features[feature] = features[feature].cuda()
            
    
    return features


def generate_samples(model, sde, model_type, config, output_directory, dataset="fragment",
                     output_ground_truth=True, length=None, num_samples_param=1, chunk_size_param=None,
                     likelihood=False):
    
    model.eval()

    config.device = "cuda"

    if not os.path.isdir(output_directory):
        
        os.mkdir(output_directory)
    
    
    if dataset == "fragment":
        if model_type == "rotamer":
            loader = torch.utils.data.DataLoader(FragmentDatasetRotamer(mode="val_subset", pad=False,
                                                                        length=length),
                                                 batch_size=1, num_workers=2, collate_fn=collate,
                                                 shuffle=True)
        
        elif model_type == "protonation":
            loader = torch.utils.data.DataLoader(FragmentDatasetHydrogen(mode="val_subset", pad=False,
                                                                        length=length),
                                                 batch_size=1, num_workers=2, collate_fn=collate,
                                                 shuffle=True)
        
        elif model_type == "backbone":
            loader = torch.utils.data.DataLoader(FragmentDatasetBackbone(mode="val_subset", pad=False,
                                                                        length=length),
                                                 batch_size=1, num_workers=2, collate_fn=collate,
                                                 shuffle=True)
            
    elif dataset == "peptide":
        
            if model_type == "backbone":
                collate_fn = collate
            else:
                collate_fn = collate_multiple_coords

            loader = torch.utils.data.DataLoader(PeptideTestDataset(pad=False, model=model_type),
                                                 batch_size=1, num_workers=2, collate_fn=collate_fn,
                                                 shuffle=True)
    
    elif dataset == "ped":
        
            loader = torch.utils.data.DataLoader(PEDDataset(pad=False, model=model_type),
                                                 batch_size=1, num_workers=2, 
                                                 collate_fn=collate_multiple_coords,
                                                 shuffle=True)
        
    elif dataset == "md":
    
        loader = torch.utils.data.DataLoader(MDDataset(pad=False, mode="test", 
                                                       model=model_type, no_mod=True),
                                             batch_size=1, num_workers=2, 
                                             collate_fn=collate_multiple_coords,
                                             shuffle=True)
            
    
    len_dataset = len(loader)
    
    
    if likelihood:
        all_likelihoods = {}
    
    for value, features in enumerate(loader):
    
        if features == None:
            print(value)
            continue

        if value % 100 == 0:
            
            print("Currently on number %i of %i" %(value, len_dataset))
   
            sys.stdout.flush()
        
        sequence = ''.join([residue_mapping[i] for i in features["fragment_seq"][0].numpy() if i < 20])
        
        features = to_cuda(features)
        
        if model_type == "rotamer":
            
            if features["coordinates_rotamer"].shape[1] > 1:
                
                
                num_samples = features["coordinates_rotamer"].shape[1]

                
                coordinates_rotamer_full = features["coordinates_rotamer"]
                
                edge_mask_rotamer_full = features["edge_mask_rotamer"]
                

            else:
                coordinates_rotamer_full = features["coordinates_rotamer"]
                
                num_samples = num_samples_param


            atom_mask = features["atom_mask_rotamer"]
            
            atom_indices = torch.sum(atom_mask, dim=-1) > 0
          
            atom_mask = atom_mask[:, atom_indices[0], :]
            
            indices = features["amino_acid_pos_rotamer"][:, atom_indices[0]] - 1
            
            means_full = torch.stack(
                [features["centroid_coordinates"]\
                 [i][:,features['centroid_indices'][i]\
                     [features['amino_acid_pos_rotamer'][i]-1], :]\
                     for i in range(len(features["centroid_coordinates"]))])
            
            means_full = means_full[:, :, atom_indices[0], :]
        

        elif model_type == "protonation":
            
            atom_mask = features["atom_mask_h"]
            
            if features["coordinates_heavy_h"].shape[1] > 1:
                
                num_samples = features["coordinates_heavy_h"].shape[1]
                
                coordinates_heavy_h_full = features["coordinates_heavy_h"]
                
                coordinates_h_full = features["coordinates_h"]
                
                edge_mask_heavy_neighbors_full = features["edge_mask_heavy_neighbors"]
                
                edge_mask_heavy_h_heavy_full = features["edge_mask_h_heavy"]
               
            else:
                
                num_samples = num_samples_param
                
                coordinates_heavy_h_full = features["coordinates_heavy_h"]
                
                coordinates_h_full = features["coordinates_h"]

            atom_indices = torch.sum(atom_mask, dim=-1) > 0
            
            indices = None
            
            means = None
            
        elif model_type == "backbone":
            
            atom_mask = features["atom_mask_backbone"]
            
            atom_indices = torch.sum(atom_mask, dim=-1) > 0
            
            indices = None
            
            means = None
            
            num_samples = num_samples_param
            
        if chunk_size_param == None:
            chunk_size = num_samples
        else:
            chunk_size = chunk_size_param
            
        
        
        for i in range(num_samples//chunk_size):
            with torch.no_grad():
                if model_type == "backbone":
                    score_fn = model.get_score_fn(features, sde)
                elif model_type == "rotamer":
                    
                    if coordinates_rotamer_full.shape[1] > 1:
                        
                        features["coordinates_rotamer"] = coordinates_rotamer_full[:, i*chunk_size:(i+1)*chunk_size,
                                                                                   ...]
                        
                        features["edge_mask_rotamer"] = edge_mask_rotamer_full[:, i*chunk_size:(i+1)*chunk_size,
                                                                                   ...]
                        
                        means = means_full[:, i*chunk_size:(i+1)*chunk_size, ...]
                        
                    else:
                        
                        means = means_full
                        
                    score_fn = model.get_score_fn(features, sde, side_chain_only=True)
                    
                    
                elif model_type == "protonation":
                    
                    if coordinates_heavy_h_full.shape[1] > 1:
                        
                        features["coordinates_heavy_h"] = coordinates_heavy_h_full[:, i*chunk_size:(i+1)*chunk_size,
                                                                                   ...]
                        
                        features["coordinates_h"] = coordinates_h_full[:, i*chunk_size:(i+1)*chunk_size,
                                                                       ...]
                        
                        features["edge_mask_heavy_neighbors"] = edge_mask_heavy_neighbors_full[:, i*chunk_size:(i+1)*chunk_size,
                                                                                              ...]
                        
                        features["edge_mask_h_heavy"] = edge_mask_heavy_h_heavy_full[:, i*chunk_size:(i+1)*chunk_size,
                                                                                              ...]
                        
                    score_fn = model.get_score_fn(features, sde)
                
            sampling_fn = sampling.get_sampling_fn(config, sde, (atom_mask.shape[0],
                                                         chunk_size,
                                                         torch.sum(atom_indices),
                                                         3),
                                           inverse_scaler = lambda x: x,
                                           eps=1e-5,
                                           atom_mask=atom_mask,
                                           indices=indices, 
                                           means=means,
                                           likelihood=likelihood)
    
            complete = False
            while not complete:
       
                try:
                    if not likelihood:
                        z_0, n = sampling_fn(score_fn)
                    else:
                        logp, z_0, n = sampling_fn(score_fn)
                except TimeOutException:
                    continue
                
                else:
                    complete = True
                    
                    
                    
            
      
            
            if model_type == "protonation":
                
                length = len(features["atom_names_heavy"][0])
                
                length_h = len(features["atom_names_h"][0])
            
                atom_names = features["atom_names_heavy"][0] + features["atom_names_h"][0]
                
                amino_acid_pos = np.concatenate((features["amino_acid_pos_heavy"][:, :length].cpu().numpy(),
                                                 features["amino_acid_pos_h"][:, :length_h].cpu().numpy()), axis=1)[0]
                
                z_0 = z_0 + features["coordinates_heavy_h"][:, :, features["bound_atom"][0], :]
                
                
                if features["coordinates_heavy_h"].shape[1] == z_0.shape[1]:
                    full_coords  = np.concatenate((features["coordinates_heavy_h"][:, :, :length].cpu().numpy(),
                                               z_0.cpu().detach().numpy()[:, : , :length_h]), axis=2)
                else:
                    full_coords  = np.concatenate((features["coordinates_heavy_h"][:, :, :length].repeat(1, z_0.shape[1], 1, 1).cpu().numpy(),
                                               z_0.cpu().detach().numpy()[:, : , :length_h]), axis=2)
                    
                
            
            elif model_type == "backbone":
                
                z_0 =  torch.sum(z_0.unsqueeze(2).repeat(1, 1, z_0.shape[2], 1, 1)*\
                   features["restore_indices"][:, None, :, :, None], axis=3)
                    
                atom_names = features["atom_names_backbone"][0]
                
                amino_acid_pos = features["amino_acid_pos_backbone"][0].cpu().numpy()
                
                full_coords = z_0 - z_0[:, :, :1, :]
                
            elif model_type == "rotamer":
                
                full_coords = features["coordinates_rotamer"].\
                    masked_scatter(features["atom_mask_rotamer"].unsqueeze(
                   1).bool(), z_0)
                        
                atom_names = features["atom_names_rotamer"][0]
                
                amino_acid_pos = features["amino_acid_pos_rotamer"][0].cpu().numpy()
                        
            if chunk_size == num_samples:
                pdb_utils.output_structure(full_coords[0], 
                        atom_names, 
                        sequence, 
                        amino_acid_pos,
                        os.path.join(output_directory, sequence + ".pdb"))
                
                
                if likelihood:
                    all_likelihoods[sequence] = logp.cpu().detach().numpy()
                    
            else:
                pdb_utils.output_structure(full_coords[0], 
                        atom_names, 
                        sequence, 
                        amino_acid_pos,
                        os.path.join(output_directory, sequence + "_" + str(i) + ".pdb"))
                
                
                if likelihood:
                    all_likelihoods[sequence + "_" + str(i)] = logp
            

        if output_ground_truth:
            
            if model_type == "protonation":
                
                h_coords = coordinates_h_full + coordinates_heavy_h_full\
                    [:, :, features["bound_atom"][0], :]
                 
                full_coords = np.concatenate((coordinates_heavy_h_full[:, :, :length].cpu().numpy(),
                                              h_coords.cpu().detach().numpy()[:, :, :length_h]), axis=2)
            
            elif model_type == "backbone":
                
                full_coords =  torch.sum(features["coordinates_backbone"].unsqueeze(2).repeat(1, 1, features["coordinates_backbone"].shape[2], 1, 1)*\
                                       features["restore_indices"][:, None, :, :, None], axis=3)
                
                
            elif model_type == "rotamer":
                
                full_coords = features["coordinates_rotamer"]\
                        [:, :, :len(features["atom_names_rotamer"][0]), :]
                      
                
         
            pdb_utils.output_structure(full_coords[0], 
                    atom_names, 
                    sequence, 
                    amino_acid_pos,
                    os.path.join(output_directory, sequence + "_ground_truth.pdb"))
                
               
                    
    if likelihood:
        
        np.save(os.path.join(output_directory, "likelihoods.npy"),
                             all_likelihoods)
        
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument("-sm", dest="saved_model", help="File containing saved params",
                        required=False, type=str, default="Saved.pth")
    parser.add_argument("-o", dest="output_directory",
                        help="Directory to output generated pdbs", required=True, type=str)
    parser.add_argument("--ogt", dest="output_gt",
                        help="Whether to output ground truth", required=False,
                        action="store_true")
    parser.add_argument("-m", dest="model", type=str, 
                        help="Model to train, backbone, rotamer or protonation",
                        required=False, default="rotamer")
    parser.add_argument("-d", dest="dataset", type=str, 
                        help="fragment, peptide or ped",
                        required=False, default="fragment")
    parser.add_argument("-l", dest="length", type=int, 
                        help="Length of peptides to generate",
                        required=False, default=None)
    parser.add_argument("-n", dest="num_samples", type=int, 
                        help="Number of samples to generate",
                        required=False, default=1)
    parser.add_argument("-c", dest="chunk_size", type=int, 
                        help="Chunk size to use when generating samples",
                        required=False, default=None)
    parser.add_argument("-ht", dest="hypernetwork_type", type=str, 
                        help="Type of hypernetwork to use, either bert or resnet",
                        required=False, default=None)
    parser.add_argument("-t", dest="temp", type=float, required=False, default=1.0,
                        help="sampling temperature")
    parser.add_argument("--l", dest="likelihood", help="Whether to compute likelihoods",
                        required=False, default=False,
                        action="store_true")
    args = parser.parse_args()

    if args.model != "backbone" and args.model != "protonation" and args.model != "rotamer":
        raise ValueError("Model argument is invalid, must be one of backbone, rotamer or protonation")
        
        
    if args.model == "rotamer":
        config = config_rotamer
        model = DynamicsRotamer(config.model_config)
    elif args.model == "protonation":
        
        config = config_hydrogen
        
  
        model = DynamicsHydrogen(config.model_config)
 
    elif args.model == "backbone":
        
        if args.hypernetwork_type == "bert":
            config = config_backbone_bert
        elif args.hypernetwork_type == "resnet":
            config = config_backbone
        else:
           raise ValueError("Hypernetwork type argument is invalid, must be one of bert or resnet")

        model = BackboneModel(config)

   
    sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
                                beta_max=config.sde_config.beta_max,
                                temperature=args.temp)
    
    
        
    torch.cuda.empty_cache()
    model.cuda()

    
    if config.training.ema != None and "ema_state_dict" in torch.load(args.saved_model):
        ema = ExponentialMovingAverage(model.parameters(), decay=config.training.ema)
        ema.load_state_dict(torch.load(args.saved_model)["ema_state_dict"])
        
        ema.copy_to(model.parameters())
    
    else:

        keys = torch.load(args.saved_model)["model_state_dict"].keys()

        rotamer_dynamics = "rotamer_dynamics" in [i.split(".")[0] for i in keys]

        hydrogen_dynamics = "hydrogen_dynamics" in [i.split(".")[0] for i in keys]

        if rotamer_dynamics and args.model == "rotamer":

            params_dict = {}

            params = torch.load(args.saved_model)["model_state_dict"]

            for param in params:

                if "rotamer_dynamics" in param:

                    params_dict[param[len("rotamer_dynamics."):]] = params[param]


            model.load_state_dict(params_dict)
            
        elif hydrogen_dynamics and args.model == "protonation":
            
            params_dict = {}

            params = torch.load(args.saved_model)["model_state_dict"]
            
            for param in params:

                if "hydrogen_dynamics" in param:

                    params_dict[param[len("hydrogen_dynamics."):]] = params[param]

            model.load_state_dict(params_dict)
        else:
            model.load_state_dict(torch.load(args.saved_model)["model_state_dict"])
    

    generate_samples(model, sde, args.model, config, args.output_directory,
                     output_ground_truth=args.output_gt, dataset=args.dataset, 
                     length=args.length, num_samples_param=args.num_samples,
                     chunk_size_param=args.chunk_size, likelihood=args.likelihood)
    
    
