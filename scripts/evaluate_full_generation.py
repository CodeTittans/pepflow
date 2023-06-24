from pepflow.model.model import BackboneModel
from pepflow.model.dynamics import DynamicsRotamer, DynamicsHydrogen
from pepflow.model.model_configs import config_rotamer, config_hydrogen,\
    config_backbone_bert
from pepflow.model.ema import ExponentialMovingAverage
from pepflow.utils.dataset_utils import get_features_from_seq
from pepflow.utils import pdb_utils
import numpy as np
import torch
import argparse
import os
import time

def to_cuda(features):
    
    for feature in features:
        
        if "atom_names" not in feature and "cyclic_bond" not in feature:
            
            features[feature] = features[feature].cuda()
            
    
    return features

def generate_samples(sequence, model, backbone_sde, rotamer_sde, protonation_sde, num_samples,
                     chunksize, likelihood, output_directory, protonate, sample_rotamers):
    
    
    features = get_features_from_seq(sequence)
    
    all_coords = []
    
    all_energies = []
    
    all_likelihoods = []
    
    all_times = []
    
    for num in range(num_samples//chunksize):
        
        start_time = time.time()
        
        full_coords, likelihoods, energies, _ = model.sample_all_atoms(to_cuda(features), backbone_sde, 
                                                               rotamer_sde, protonation_sde, chunksize,
                                                               likelihood, protonate=protonate, 
                                                               sample_rotamers=sample_rotamers)
        
        all_times.append(time.time() - start_time)
        
        all_coords.append(full_coords)
        
        all_energies.append(energies)
        
        all_likelihoods.append(likelihoods)
        
    full_coords = torch.cat(all_coords, dim=1)
    
    if sample_rotamers and protonate:
        energies = torch.cat(all_energies, dim=1)
    else:
        energies = None
        
    if likelihood:
        likelihoods = torch.cat(all_likelihoods, dim=1)
    else:
        likelihoods = None
    
    
    if not sample_rotamers:
        atom_names = features["atom_names_backbone"][0] 
        
        length = len(features["atom_names_backbone"][0])
        
        amino_acid_pos = features["amino_acid_pos_backbone"][0, :length].cpu().numpy()
        
    elif not protonate:
        
        atom_names = features["atom_names_rotamer"][0]
        
        length = len(features["atom_names_rotamer"][0])   
        
        features["amino_acid_pos_rotamer"][0, :length].cpu().numpy()
        
        amino_acid_pos = features["amino_acid_pos_rotamer"][0, :length].cpu().numpy()
    else:
        atom_names = features["atom_names_heavy"][0] + features["atom_names_h"][0]
        
        length = len(features["atom_names_heavy"][0])
        
        length_h = len(features["atom_names_h"][0])
        
        amino_acid_pos = np.concatenate((features["amino_acid_pos_heavy"][:, :length].cpu().numpy(),
                                                   features["amino_acid_pos_h"][:, :length_h].cpu().numpy()), axis=1)[0]
    
    pdb_utils.output_structure(full_coords[0], 
        atom_names, 
        sequence, 
        amino_acid_pos,
        os.path.join(output_directory, sequence + ".pdb"))
    
    
    return likelihoods, energies, all_times
                    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument("-bm", dest="backbone_model", help="File containing saved backbone params",
                        required=False, type=str, default="Saved.pth")
    parser.add_argument("-rm", dest="rotamer_model", help="File containing saved rotamer params",
                        required=False, type=str, default="Saved.pth")
    parser.add_argument("-hm", dest="hydrogen_model", help="File containing saved hydrogen params",
                        required=False, type=str, default="Saved.pth")
    parser.add_argument("-fm", dest="full_model", help="File containing saved params for the full model",
                        required=False, type=str, default=None)
    parser.add_argument("-o", dest="output_directory",
                        help="Directory to output generated pdbs", required=True, type=str)
       
    parser.add_argument("-n", dest="num_samples", type=int, 
                        help="Number of samples to generate",
                        required=False, default=1)
    parser.add_argument("-c", dest="chunk_size", type=int, 
                        help="Chunk size to use when generating samples",
                        required=False, default=1)
    parser.add_argument("-s", dest="sequence_list", type=str, 
                        help=".npy file containing peptide sequences on which to test",
                        required=True, default=None)

    
    parser.add_argument("--r", dest="sample_rotamers", help="Whether to sample rotamers",
                        required=False, default=False,
                        action="store_true")
    parser.add_argument("--p", dest="protonate", help="Whether to protonate sampled molecules",
                    required=False, default=False,
                    action="store_true")
        
    parser.add_argument("--l", dest="likelihood", help="Whether to compute likelihoods",
                        required=False, default=False,
                        action="store_true")
    args = parser.parse_args()

    if args.full_model == None:
        config = config_backbone_bert
        
        backbone_sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
                                    beta_max=config.sde_config.beta_max,
                                    temperature=1.0, mask_first_atom=True)
        
        
        model = BackboneModel(config)
    
        if config.training.ema != None and "ema_state_dict" in torch.load(args.backbone_model):
            ema = ExponentialMovingAverage(model.parameters(), decay=config.training.ema)
            ema.load_state_dict(torch.load(args.backbone_model)["ema_state_dict"])
            
            ema.copy_to(model.parameters())
        
        else:
            
            model.load_state_dict(torch.load(args.backbone_model)["model_state_dict"])
        
        
        
        config = config_rotamer
        rotamer_dynamics = DynamicsRotamer(config.model_config)
        
        rotamer_sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
                                beta_max=config.sde_config.beta_max,
                                temperature=0.4)
        
        if config.training.ema != None and "ema_state_dict" in torch.load(args.rotamer_model):
            ema = ExponentialMovingAverage(rotamer_dynamics.parameters(), decay=config.training.ema)
            ema.load_state_dict(torch.load(args.rotamer_model)["ema_state_dict"])
            
            ema.copy_to(rotamer_dynamics.parameters())
        
        else:
            
            rotamer_dynamics.load_state_dict(torch.load(args.rotamer_model)["model_state_dict"])
        
        
    
        config = config_hydrogen
        hydrogen_dynamics = DynamicsHydrogen(config.model_config)
        
        hydrogen_sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
                                beta_max=config.sde_config.beta_max,
                                temperature=1.0)
        
        if config.training.ema != None and "ema_state_dict" in torch.load(args.hydrogen_model):
            ema = ExponentialMovingAverage(hydrogen_dynamics.parameters(), decay=config.training.ema)
            ema.load_state_dict(torch.load(args.hydrogen_model)["ema_state_dict"])
            
            ema.copy_to(hydrogen_dynamics.parameters())
        
        else:
            
            hydrogen_dynamics.load_state_dict(torch.load(args.hydrogen_model)["model_state_dict"])
        
        
        model.hydrogen_dynamics = hydrogen_dynamics
        
        model.rotamer_dynamics = rotamer_dynamics
        
        torch.cuda.empty_cache()
        model.cuda()
        
    else:

        config = config_backbone_bert
        
        backbone_sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
                                    beta_max=config.sde_config.beta_max,
                                    temperature=0.7, mask_first_atom=True)
        
        
        model = BackboneModel(config)
    
      
        
        
        config = config_rotamer
        rotamer_dynamics = DynamicsRotamer(config.model_config)
        
        rotamer_sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
                                beta_max=config.sde_config.beta_max,
                                temperature=0.4)
        
        
    
        config = config_hydrogen
        hydrogen_dynamics = DynamicsHydrogen(config.model_config)
        
        hydrogen_sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
                                beta_max=config.sde_config.beta_max,
                                temperature=1.0)
        
        
        model.hydrogen_dynamics = hydrogen_dynamics
        
        model.rotamer_dynamics = rotamer_dynamics
        
        if "ema_state_dict" in torch.load(args.full_model):
            ema = ExponentialMovingAverage(model.parameters(), decay=config.training.ema)
            ema.load_state_dict(torch.load(args.full_model)["ema_state_dict"])

            ema.copy_to(model.parameters())
        else:
            model.load_state_dict(torch.load(args.full_model)["model_state_dict"])
        
        torch.cuda.empty_cache()
        model.cuda()        

    
    model.eval()
    
    if not os.path.isdir(args.output_directory):
        os.mkdir(args.output_directory)

    sequences = np.load(args.sequence_list)
     
    for sequence in sequences:
        likelihoods, energies, all_times = generate_samples(sequence, model, backbone_sde, rotamer_sde, hydrogen_sde, 
                     args.num_samples, args.chunk_size, args.likelihood, args.output_directory,
                     args.protonate, args.sample_rotamers)
        
    
        
        if args.protonate and args.sample_rotamers:
            np.save(os.path.join(args.output_directory, "energies_"+sequence+".npy"), energies.cpu().detach().numpy())
    
        if args.likelihood:
            np.save(os.path.join(args.output_directory, "likelihoods_"+sequence+".npy"), likelihoods.cpu().detach().numpy())

        np.save(os.path.join(args.output_directory, "times_"+sequence+".npy"), all_times)