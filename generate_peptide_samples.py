import sys
sys.path.append('../pepflow/')

from pepflow.model.model import BackboneModel
from pepflow.model.dynamics import DynamicsRotamer, DynamicsHydrogen
from pepflow.model.model_configs import config_rotamer, config_hydrogen,\
    config_backbone_bert
from pepflow.model.ema import ExponentialMovingAverage
from pepflow.utils.dataset_utils import get_features_from_seq
from pepflow.utils import pdb_utils
from time import time
import numpy as np
import torch
import argparse
import os

def to_cuda(features):
    
    for feature in features:
        
        if "atom_names" not in feature and "cyclic" not in feature:
            
            features[feature] = features[feature].cuda()
            
    
    return features


def generate_backbone_samples(sequence, model, backbone_sde, 
                            num_samples, likelihood, output_directory, 
                            chunksize,
                            single_structure_prediction):
    
    
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
        
    features = get_features_from_seq(sequence)
    
    all_likelihoods = []
    
    all_coords = []
    
    all_energies = []
    
    start_time = time()
    
    for num in range(num_samples//chunksize):
        print("Currently on chunk %i of %i, elapsed time %f" %(num+1, num_samples//chunksize, 
                                                               time()-start_time))
        sys.stdout.flush()
        full_coords, likelihoods, _ = model.sample_backbone(to_cuda(features), backbone_sde, 
                                                                      num_samples=num_samples,
                                                                      likelihood=likelihood)
        #     def sample_backbone(self, features, sde, num_samples=1, rtol=1e-3, atol=1e-3, solver='RK23',
        #                likelihood=True, exact_trace=False, differentiable=False, multiple_points=False):

        all_coords.append(full_coords)
                
        all_likelihoods.append(likelihoods)
        
    full_coords = torch.cat(all_coords, dim=1)
    

    atom_names = features["atom_names_backbone"][0] 
    length = len(features["atom_names_backbone"][0])
    amino_acid_pos = features["amino_acid_pos_backbone"][0, :length].cpu().numpy()
        
    
    pdb_utils.output_structure(full_coords[0], 
        atom_names, 
        sequence, 
        amino_acid_pos,
        os.path.join(output_directory, sequence + ".pdb"))
    
    if single_structure_prediction:
        single_prediction = pdb_utils.get_single_structure_prediction(os.path.join(output_directory, sequence + ".pdb"))
        
        pdb_utils.output_structure(full_coords[0][single_prediction:single_prediction+1], 
            atom_names, 
            sequence, 
            amino_acid_pos,
            os.path.join(output_directory, sequence + "_single_prediction.pdb"))
    
        likelihoods = torch.cat(all_likelihoods, dim=1)
        with open(os.path.join(output_directory, sequence + "_report.out"), 'w') as f:
            header = "Model"
            header = header + "\tLikelihood"                           
            header = header + "\n"
            f.writelines(header)
            
            for index in range(num_samples):
                row = str(index+1)
                row = row + "\t" + str(likelihoods.cpu().detach().numpy()[0][index])                
                row = row + "\n"
                f.writelines(row)


def generate_samples(sequence, model, backbone_sde, rotamer_sde, protonation_sde, 
                     num_samples, likelihood, compute_energies, output_directory, 
                     protonate, sample_rotamers, fix_chirality, d_peptide, chunksize,
                     single_structure_prediction):
    
    
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
        
    features = get_features_from_seq(sequence)
    
    all_likelihoods = []
    
    all_coords = []
    
    all_energies = []
    
    start_time = time()
    
    for num in range(num_samples//chunksize):
        print("Currently on chunk %i of %i, elapsed time %f" %(num+1, num_samples//chunksize, 
                                                               time()-start_time))
        sys.stdout.flush()
        full_coords, likelihoods, energies, _ = model.sample_all_atoms(to_cuda(features), backbone_sde, 
                                                               rotamer_sde, protonation_sde, chunksize,
                                                               likelihood, compute_energies=compute_energies,
                                                               protonate=protonate, sample_rotamers=sample_rotamers,
                                                               fix_chirality=fix_chirality,
                                                               d_peptide=d_peptide,
                                                               unnormalized_energy=True)
        
        all_coords.append(full_coords)
        
        all_energies.append(energies)
        
        all_likelihoods.append(likelihoods)
        
    full_coords = torch.cat(all_coords, dim=1)
    
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
    
    if single_structure_prediction:
        single_prediction = pdb_utils.get_single_structure_prediction(os.path.join(output_directory, sequence + ".pdb"))
        
        pdb_utils.output_structure(full_coords[0][single_prediction:single_prediction+1], 
            atom_names, 
            sequence, 
            amino_acid_pos,
            os.path.join(output_directory, sequence + "_single_prediction.pdb"))
    
    if likelihood or compute_energies:
        if likelihood:
            likelihoods = torch.cat(all_likelihoods, dim=1)
        if compute_energies:
            energies = torch.cat(all_energies, dim=1)

        with open(os.path.join(output_directory, sequence + "_report.out"), 'w') as f:
            header = "Model"
            
            if likelihood:
                header = header + "\tLikelihood"
                
            if compute_energies:
                header = header + "\tEnergy (KJ/mol)"
           
            header = header + "\n"
            f.writelines(header)
            
            for index in range(num_samples):
                
                row = str(index+1)
                
                if likelihood:
                    
                    row = row + "\t" + str(likelihoods.cpu().detach().numpy()[0][index])
                
                if compute_energies:
                    
                    row = row + "\t" + str(energies.cpu().detach().numpy()[0][index])
                
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

    parser.add_argument("--r", dest="sample_rotamers", help="To not sample rotamers",
                        required=False, default=True,
                        action="store_false")
    parser.add_argument("--p", dest="protonate", help="To not protonate molecules",
                    required=False, default=True,
                    action="store_false")
    
    parser.add_argument("--f", dest="fix_chirality", help="To not fix the chirality",
                    required=False, default=True,
                    action="store_false")
    parser.add_argument("--d", dest="d_peptide", help="Whether to generate d_peptides",
                        required=False, default=False,
                        action="store_true")
    
    parser.add_argument("--l", dest="likelihood", help="Whether to compute likelihoods",
                        required=False, default=False,
                        action="store_true")
    parser.add_argument("--e", dest="compute_energies", help="Whether to compute energies",
                        required=False, default=False,
                        action="store_true")
    

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
        
        
        
        # config = config_rotamer
        # rotamer_dynamics = DynamicsRotamer(config.model_config)
        
        # rotamer_sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
        #                         beta_max=config.sde_config.beta_max,
        #                         temperature=0.4)
        
        # if config.training.ema != None and "ema_state_dict" in torch.load(args.rotamer_model):
        #     ema = ExponentialMovingAverage(rotamer_dynamics.parameters(), decay=config.training.ema)
        #     ema.load_state_dict(torch.load(args.rotamer_model)["ema_state_dict"])
            
        #     ema.copy_to(rotamer_dynamics.parameters())
        
        # else:
            
        #     rotamer_dynamics.load_state_dict(torch.load(args.rotamer_model)["model_state_dict"])
        
        
    
        # config = config_hydrogen
        # hydrogen_dynamics = DynamicsHydrogen(config.model_config)
        
        # hydrogen_sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
        #                         beta_max=config.sde_config.beta_max,
        #                         temperature=1.0)
        
        # if config.training.ema != None and "ema_state_dict" in torch.load(args.hydrogen_model):
        #     ema = ExponentialMovingAverage(hydrogen_dynamics.parameters(), decay=config.training.ema)
        #     ema.load_state_dict(torch.load(args.hydrogen_model)["ema_state_dict"])
            
        #     ema.copy_to(hydrogen_dynamics.parameters())
        
        # else:
            
        #     hydrogen_dynamics.load_state_dict(torch.load(args.hydrogen_model)["model_state_dict"])
        
        
        # model.hydrogen_dynamics = hydrogen_dynamics
        
        # model.rotamer_dynamics = rotamer_dynamics
        
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
    # generate_samples(args.sequence, model, backbone_sde, rotamer_sde, hydrogen_sde, 
    #                  args.num_samples, args.likelihood, args.compute_energies, args.output_directory,
    #                  args.protonate, args.sample_rotamers, args.fix_chirality, args.d_peptide,
    #                  args.chunk_size, args.single_structure_prediction)
    
generate_backbone_samples(args.sequence,model, backbone_sde, 
                          args.num_samples, args.likelihood, args.output_directory, 
                          args.chunk_size,
                          args.single_structure_prediction)