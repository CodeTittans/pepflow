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
import numpy as np
import copy

def to_cuda(features):
    
    for feature in features:
        
        if "atom_names" not in feature and "cyclic_bond_indices" not in feature:
            
            features[feature] = features[feature].cuda()
            
    
    return features

    
def generate_samples(model, sequence, bonds, sde, sde_rotamer, sde_hydrogen, 
                     output_directory, num_samples=1, chunk_size=None,
                     single_structure_prediction=False, num_steps=250, fix_chirality=True,
                     d_peptide=False):



    if not os.path.isdir(output_directory):
        
        os.mkdir(output_directory)
    
    
    
  
        
    features = get_features_from_seq(sequence, cyclic_bonds=bonds)
    
    not_backbone = len(features["backbone_cyclic_bond_indices"][0]) == 0
    

  
    sequence = ''.join([residue_mapping[i] for i in features["fragment_seq"][0].numpy() if i < 20])
    

    
    features = to_cuda(features)
    


        
    if chunk_size == None:
        chunk_size = num_samples
        
        
    all_bond_deviations = []
    
    all_logps = []
    
    all_coords = []

        
    for i in range(num_samples//chunk_size):

        if not not_backbone:
            full_coords, bond_deviations, tolerances, logps = model.sample_with_bond_constraint(
                copy.deepcopy(features), sde, sde_rotamer, sde_hydrogen, chunk_size, num_steps=num_steps,
                fix_chirality=fix_chirality, d_peptide=d_peptide)
      
            
            atom_names = features["atom_names_heavy"][0] + features["atom_names_h"][0]
            
            length = len(features["atom_names_heavy"][0])
            
            length_h = len(features["atom_names_h"][0])
            
            amino_acid_pos = np.concatenate((features["amino_acid_pos_heavy"][:, :length].cpu().numpy(),
                                                       features["amino_acid_pos_h"][:, :length_h].cpu().numpy()), axis=1)[0]
                    
                
                        
                
            full_coords = full_coords - full_coords[:, :, :1, :]


            all_coords.append(full_coords)
                
            all_bond_deviations.append(bond_deviations)
            
            all_logps.append(logps)
            
        else:
            
            full_coords, bond_deviations, tolerances, logps = model.sample_with_bond_constraint_sidechain(
                copy.deepcopy(features), sde, sde_rotamer, sde_hydrogen, chunk_size, num_steps=num_steps,
                fix_chirality=fix_chirality, d_peptide=d_peptide)
      
            
            
      
            atom_names = features["atom_names_heavy"][0] + features["atom_names_h"][0]
            
            length = len(features["atom_names_heavy"][0])
            
            length_h = len(features["atom_names_h"][0])
            
            amino_acid_pos = np.concatenate((features["amino_acid_pos_heavy"][:, :length].cpu().numpy(),
                                                       features["amino_acid_pos_h"][:, :length_h].cpu().numpy()), axis=1)[0]
                    
                
                        
                
            full_coords = full_coords - full_coords[:, :, :1, :]

            all_coords.append(full_coords)
                
            all_bond_deviations.append(bond_deviations)
            
            all_logps.append(logps)


    full_coords = torch.cat(all_coords, dim=1)
    
    pdb_utils.output_structure(full_coords[0], 
        atom_names, 
        sequence, 
        amino_acid_pos,
        os.path.join(output_directory, sequence + ".pdb"))

    bond_deviations = np.concatenate(all_bond_deviations, axis=2)
    logps = np.concatenate(all_logps, axis=1)
    
    if single_structure_prediction:
        single_prediction = pdb_utils.get_single_structure_prediction_deviations(
            os.path.join(output_directory, sequence + ".pdb"), bond_deviations)
        
        pdb_utils.output_structure(full_coords[0][single_prediction:single_prediction+1], 
            atom_names, 
            sequence, 
            amino_acid_pos,
            os.path.join(output_directory, sequence + "_single_prediction.pdb"))
    

    

        
    
    with open(os.path.join(output_directory, sequence + "_report.out"), 'w') as f:
        header = "Model\tBond deviation\tLikelihoods\n"
        
        
        f.writelines(header)
        
        for index in range(num_samples):
            
            row = str(index+1)
            
            row = row + "\t" + str(bond_deviations[-1, 0, index])
            
            row = row + "\t" + str(logps[0][index])
            
            row = row + "\n"

            f.writelines(row)       
                   
    

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
                        help="File with output", required=True, type=str)
       
    parser.add_argument("-n", dest="num_samples", type=int, 
                        help="Number of samples to generate",
                        required=False, default=1)
    parser.add_argument("-c", dest="chunk_size", type=int, 
                        help="Chunk size to use when generating samples",
                        required=False, default=None)
    parser.add_argument("-s", dest="sequence", type=str, 
                        help="Peptide sequence for which to generate samples",
                        required=True, default=None)
    
    parser.add_argument("-st", dest="num_steps", type=int, 
                        help="Number of steps",
                        required=False, default=250)
    
    parser.add_argument("--f", dest="fix_chirality", help="To not fix the chirality",
                    required=False, default=True,
                    action="store_false")
    parser.add_argument("--d", dest="d_peptide", help="Whether to generate d_peptides",
                        required=False, default=False,
                        action="store_true")
    
    parser.add_argument("-b", dest="bonds_list",
                    help="Bond file", required=True, type=str)

 

    parser.add_argument("--s", dest="single_structure_prediction", help="Whether to compute energies",
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
    
    bonds = []
    
    with open(args.bonds_list, "r") as f:
        
        for line in f.readlines():
            
            bonds.append(line.strip().split(","))
            
     

    generate_samples(model, args.sequence, bonds, backbone_sde, rotamer_sde, hydrogen_sde,
                     args.output_directory, num_samples=args.num_samples,
                     chunk_size=args.chunk_size, single_structure_prediction=args.single_structure_prediction,
                     num_steps=args.num_steps, fix_chirality=args.fix_chirality, d_peptide=args.d_peptide)
    
    
