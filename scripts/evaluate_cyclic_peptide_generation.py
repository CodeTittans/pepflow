from pepflow.model.model import BackboneModel
from pepflow.model.dynamics import DynamicsRotamer, DynamicsHydrogen
from pepflow.model.model_configs import config_rotamer, config_hydrogen,\
    config_backbone_bert
from pepflow.model.ema import ExponentialMovingAverage
from pepflow.utils import pdb_utils
from pepflow.utils.constants import residue_mapping
from pepflow.utils.dataset_utils import get_features_from_seq
import torch
import argparse
import os
import sys
import numpy as np
import copy

def to_cuda(features):
    
    for feature in features:
        
        if "atom_names" not in feature and "cyclic_bond_indices" not in feature:
            
            features[feature] = features[feature].cuda()
            
    
    return features

    
def generate_samples(model, sequence_list, sde, sde_rotamer, sde_hydrogen, config, 
                     output_directory, num_samples=1, chunk_size=None,
                     likelihood=False, backbone_cycle=True):
    
    model.eval()

    config.device = "cuda"

    if not os.path.isdir(output_directory):
        
        os.mkdir(output_directory)
    
    
    
    len_dataset = len(sequence_list)
    
    
    for value, (sequence, bonds) in enumerate(sequence_list):
         
        
        features = get_features_from_seq(sequence, cyclic_bonds=bonds)
        
        not_backbone = len(features["backbone_cyclic_bond_indices"][0]) == 0
        
        if not_backbone and backbone_cycle:
            continue
        elif not not_backbone and not backbone_cycle:
            continue

        if value % 100 == 0:
            
            print("Currently on number %i of %i" %(value, len_dataset))
   
            sys.stdout.flush()
      
        sequence = ''.join([residue_mapping[i] for i in features["fragment_seq"][0].numpy() if i < 20])
        
        if os.path.exists(os.path.join(output_directory, sequence + "_bond_deviations.npy")):
            continue
        
        features = to_cuda(features)
        


            
        if chunk_size == None:
            chunk_size = num_samples
            
            
        all_bond_deviations = []
        all_tolerances = []
        all_logps = []
        

            
        for i in range(num_samples//chunk_size):

            if backbone_cycle:
                full_coords, bond_deviations, tolerances, logps = model.sample_with_bond_constraint(
                    copy.deepcopy(features), sde, sde_rotamer, sde_hydrogen, chunk_size)
          
                
                atom_names = features["atom_names_heavy"][0] + features["atom_names_h"][0]
                
                length = len(features["atom_names_heavy"][0])
                
                length_h = len(features["atom_names_h"][0])
                
                amino_acid_pos = np.concatenate((features["amino_acid_pos_heavy"][:, :length].cpu().numpy(),
                                                           features["amino_acid_pos_h"][:, :length_h].cpu().numpy()), axis=1)[0]
                        
                    
                            
                    
                full_coords = full_coords - full_coords[:, :, :1, :]


                full_coords = full_coords.cpu().detach().numpy()

                if chunk_size == num_samples:
                    pdb_utils.output_structure(full_coords[0], 
                            atom_names, 
                            sequence, 
                            amino_acid_pos,
                            os.path.join(output_directory, sequence + ".pdb"))
                    
                        
                else:
                    pdb_utils.output_structure(full_coords[0], 
                            atom_names, 
                            sequence, 
                            amino_acid_pos,
                            os.path.join(output_directory, sequence + "_" + str(i) + ".pdb"))
                    
                

                    
                
                all_bond_deviations.append(bond_deviations)
                all_tolerances.append(tolerances)
                all_logps.append(logps)
                
            else:
                
                full_coords, bond_deviations, tolerances, logps = model.sample_with_bond_constraint_sidechain(
                    copy.deepcopy(features), sde, sde_rotamer, sde_hydrogen, chunk_size)
          
                
                
          
                atom_names = features["atom_names_heavy"][0] + features["atom_names_h"][0]
                
                length = len(features["atom_names_heavy"][0])
                
                length_h = len(features["atom_names_h"][0])
                
                amino_acid_pos = np.concatenate((features["amino_acid_pos_heavy"][:, :length].cpu().numpy(),
                                                           features["amino_acid_pos_h"][:, :length_h].cpu().numpy()), axis=1)[0]
                        
                    
                            
                    
                full_coords = full_coords - full_coords[:, :, :1, :]


                full_coords = full_coords.cpu().detach().numpy()

                if chunk_size == num_samples:
                    pdb_utils.output_structure(full_coords[0], 
                            atom_names, 
                            sequence, 
                            amino_acid_pos,
                            os.path.join(output_directory, sequence + ".pdb"))
                    
                        
                else:
                    pdb_utils.output_structure(full_coords[0], 
                            atom_names, 
                            sequence, 
                            amino_acid_pos,
                            os.path.join(output_directory, sequence + "_" + str(i) + ".pdb"))
                    
                    
                
                all_bond_deviations.append(bond_deviations)
                all_tolerances.append(tolerances)
                all_logps.append(logps)
                
                
        bond_deviations = all_bond_deviations
        tolerances = all_tolerances
        logps = np.stack(all_logps)
            
                
        np.save(os.path.join(output_directory, sequence + "_bond_deviations.npy"), bond_deviations)
        np.save(os.path.join(output_directory, sequence + "_tolerances.npy"), tolerances)
            
        np.save(os.path.join(output_directory, sequence + "_logps.npy"), 
                    logps.flatten())                
                   
    

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
                        required=False, default=None)
    parser.add_argument("-s", dest="sequence_list", type=str, 
                        help=".npy file containing peptide sequences on which to test",
                        required=True, default=None)
    parser.add_argument("-t", dest="temp", type=float, required=False, default=1.0,
                        help="sampling temperature")
    parser.add_argument("--b", dest="backbone_cycle", help="Whether to generate backbone cycilized backbone",
                        required=False, default=True,
                        action="store_false")
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
        
        model.load_state_dict(torch.load(args.full_model)["model_state_dict"])
        
        torch.cuda.empty_cache()
        model.cuda()        

    
    model.eval()
    
    
    sequences = np.load(args.sequence_list, allow_pickle=True)
     

    generate_samples(model, sequences, backbone_sde, rotamer_sde, hydrogen_sde, config, 
                     args.output_directory,num_samples=args.num_samples,
                     chunk_size=args.chunk_size, likelihood=args.likelihood,
                     backbone_cycle=args.backbone_cycle)
    
    
