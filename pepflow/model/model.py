from pepflow.model.dynamics import DynamicsBackbone
from pepflow.model.hypernet import HyperBERT, HyperResNet
from pepflow.model.sampling import get_sampling_likelihood_fn, get_ode_sampler,\
    get_differentiable_ode_sampler
from pepflow.model.openmm.openmmwrap import OpenMMBridge, OpenMMEnergy
from pepflow.utils.pdb_utils import create_system
from pepflow.utils.constants import residue_order, backbone_atom_indices, atom_order
from pepflow.utils.tensor_utils import cdist
from pepflow.model.sampling import TimeOutException
from openmm.app.forcefield import ForceField
from simtk import unit
from time import time
from contextlib import nullcontext
import openmm
import torch.nn as nn
import torch
import numpy as np
import sys
import copy

def to_cuda(features):
    
    for feature in features:
        
        if "atom_names" not in feature and "cyclic_bond_indices" not in feature:
            
            features[feature] = features[feature].cuda()
            
    
    return features




class BackboneModel(nn.Module):

    def __init__(self, config, rotamer_dynamics=None, hydrogen_dynamics=None,
                 temperature=310):

        super(BackboneModel, self).__init__()
    
        
        if not config.hyperbert:
            self.hypernetwork = HyperResNet(**config["hypernetwork_config"])
        else:
            self.hypernetwork = HyperBERT(**config["hypernetwork_config"])
            
        self.backbone_dynamics = DynamicsBackbone(config.mainnet_config)

        self.mainnet_config = config.mainnet_config
        
        self.rotamer_dynamics = rotamer_dynamics
        
        self.hydrogen_dynamics = hydrogen_dynamics
        
        self.forcefield = ForceField('amber99sbildn.xml', 'implicit/gbn2.xml')

        self.temperature = temperature
        
    def forward(self, features, t, perturbed_data, sde):
        
        '''
        Compute dynamics parameters
        '''
        batch_size = features["fragment_seq"].shape[0]
        
        sequence, positions = self.hypernetwork.process_tensors(features["fragment_seq"])

        w_x, b_x, w_t, w_b_t = self.hypernetwork(sequence, positions, return_embedding=False)
        
        w_x_reshaped = []
        b_x_reshaped = []
        w_t_reshaped = []
        w_b_t_reshaped = []

        prev_idx_w_x = 0
        prev_idx_b_x = 0
        prev_idx_w_t = 0
        prev_idx_w_b_t = 0

        for layer in self.mainnet_config.dims:
            
            w_x_reshaped.append(w_x[prev_idx_w_x].reshape(batch_size, layer[0], layer[1]))
            prev_idx_w_x += 1

            b_x_reshaped.append(b_x[prev_idx_b_x].reshape(batch_size, layer[1]))
            prev_idx_b_x += 1

            w_t_reshaped.append(w_t[prev_idx_w_t].reshape(batch_size, 1, layer[1]))
            prev_idx_w_t += 1

            w_b_t_reshaped.append(w_b_t[prev_idx_w_b_t].reshape(batch_size, 1, layer[1]))
            prev_idx_w_b_t += 1
                
        
        '''
        Reshape network parameters based on mainnet config
        and type of layers
        '''
        
        _, std = sde.marginal_prob(torch.zeros(perturbed_data.shape).to(
                features["fragment_seq"].device), t)
                
        prediction = self.backbone_dynamics(t, perturbed_data, 
                                            features["fragment_seq"], 
                                            features["atoms_backbone"], 
                                            features["amino_acid_pos_backbone"], 
                                            features["bond_matrix_backbone"],
                                            features["edge_mask_backbone"], 
                                            features["atom_mask_backbone"],
                                            features["restore_indices"],
                                            w_x_reshaped, b_x_reshaped,
                                            w_t_reshaped, w_b_t_reshaped)
        
        prediction = -1*prediction/std[:, None, None, None].to(prediction.device)
        
        return prediction
    
    def get_hypernet_embedding(self, sequence, device='cuda'):
        
        sequence_input = torch.LongTensor(np.array([residue_order[i] for i in sequence]))\
            .unsqueeze(0).to(device)
       

        sequence, positions = self.hypernetwork.process_tensors(sequence_input)

        w_x, b_x, w_t, w_b_t, embedding = self.hypernetwork(sequence, positions, return_embedding=True)

        return embedding
    
    

              
    def _compute_bond_deviations(self, all_bond_indices, coords, backbone=True):
        
        if backbone:
            ideal_bond_length = 1.32
        else:
            ideal_bond_length = 1.5
            
        all_bond_deviations = []
        for index, bond_indices in enumerate(all_bond_indices):
                
            bond_length_deviation_sum = 0
            for bond in bond_indices:
                
                bond_length = torch.sqrt(torch.sum(torch.square(coords[index, :, bond[0]]\
                                                     - coords[index, :, bond[1]]), dim=-1) + 1e-12)
                    
                bond_length_deviation_sum += torch.abs(ideal_bond_length-bond_length)
                
            all_bond_deviations.append(bond_length_deviation_sum.flatten())
                
        all_bond_deviations = torch.stack(all_bond_deviations, dim=0)
        
        return all_bond_deviations
        
    def _compute_clashes(self, coords, bond_matrix):
        _, all_dists = cdist(coords, coords)
        
        # increase diagonal values to avoid reporting as clash
        all_dists[:, :, range(all_dists.shape[2]), range(all_dists.shape[2])] = 1000
        
        all_dists = torch.sum(torch.logical_and(all_dists < 1.6,
                                                1 - bond_matrix.unsqueeze(1)), dim=(-1, -2))
        
        num_clashes = all_dists//2 
        
        return num_clashes
    
        
    def _mcmc_bond_step(self, sampling_fn, score_fn, z, all_bond_deviations_curr, 
                        features, atom_mask, num_samples, atom_indices, 
                        all_bond_indices, step_size, curr_clashes = 0, clash_penalty=True,
                        sample=True):
        
          
        
        z_init = z.clone()
        
        completed = False
        
        while not completed:
            
            
            noise = torch.randn((atom_mask.shape[0], num_samples, torch.sum(atom_indices),
                             3), requires_grad=True, device=atom_mask.device)*step_size
        
            try:
                if sample:
                    z = z_init + noise
                else:
                    z = z_init
                    
                coords, _ = sampling_fn(score_fn, z)
                
                completed = True
            except TimeOutException:
                
                print("Sampling timed out")
                sys.stdout.flush()
                
                
            
        coords =  torch.sum(coords.unsqueeze(2).repeat(1, 1, coords.shape[2], 1, 1)*\
               features["restore_indices"][:, None, :, :, None], axis=3)
            
       
        if clash_penalty:
            
            num_clashes = self._compute_clashes(coords, features["bond_matrix_backbone"])
            
            clash_difference = num_clashes - curr_clashes
            
        all_bond_deviations = self._compute_bond_deviations(all_bond_indices, coords)
        
        
        change = all_bond_deviations - all_bond_deviations_curr
        
        accept_p = torch.clip(torch.exp(-4*change), max=1)
        
        if clash_penalty:
            
            accept_p = accept_p*torch.exp(-1*clash_difference)
        
        accept_mask = accept_p > torch.rand_like(accept_p)
        
        z = z_init*(~accept_mask[..., None, None]) + z*(accept_mask[..., None, None])
        
        all_bond_deviations_curr = all_bond_deviations_curr*(~accept_mask)\
            + all_bond_deviations*(accept_mask)
            
        
        if clash_penalty:
            
            curr_clashes = curr_clashes*(~accept_mask)\
                + num_clashes*(accept_mask)
                
            return z, all_bond_deviations_curr, curr_clashes
        
        else:
            
            return z, all_bond_deviations_curr
 
    def _mcmc_bond_step_side_chain(self, sampling_fn, score_fn, z, sampling_fn_side,
                                   sde_rotamer, z_side, all_bond_deviations_curr,
                                   features, atom_mask, num_samples, atom_indices, atom_mask_side, 
                                   atom_indices_side, means_side, indices_side, all_bond_indices, 
                                   step_size, step_size_side, curr_clashes = 0, clash_penalty=True, tol_curr=1e-2,
                                   sample=True):
        

            
        z_init = z.clone()
        
        
        
        z_init_side = z_side.clone()
        
        completed = False
        
        while not completed:
            
            noise = torch.randn((atom_mask.shape[0], num_samples, torch.sum(atom_indices),
                             3), requires_grad=True, device=atom_mask.device)*step_size
        
            _, noise_side = sde_rotamer.prior_sampling((atom_mask_side.shape[0],
                                                     num_samples,
                                                     torch.sum(atom_indices_side),
                                                     3), 
                                                   atom_mask_side,
                                                   means_side, 
                                                   indices=indices_side)
            
            noise_side = noise_side.to(atom_mask.device)*step_size_side
            
            try:
                if sample:
                    z = z_init + noise
                else:
                    z = z_init
                
                if sample:
                    z_side = z_init_side + noise_side
                else:
                    z_side = z_side
                
                
                relative_coords, _ = sampling_fn(score_fn, z)
                    
                coords =  torch.sum(relative_coords.unsqueeze(2).repeat(1, 1, relative_coords.shape[2], 1, 1)*\
                           features["restore_indices"][:, None, :, :, None], axis=3)
                    
                features["coordinates_backbone"] = relative_coords
                
                features = self._compute_rotamer_edge_masks(coords, features) 
                
                means_rotamer = torch.stack(
                    [features["centroid_coordinates"]\
                     [i][:,features['centroid_indices'][i]\
                         [features['amino_acid_pos_rotamer'][i]-1], :]\
                         for i in range(len(features["centroid_coordinates"]))])
        
                means_rotamer = means_rotamer[:, :, atom_indices_side[0], :]
                
                
                sampling_fn_side = get_ode_sampler(sde_rotamer, (atom_mask_side.shape[0],
                                                            num_samples,
                                                            torch.sum(atom_indices_side),
                                                            3), inverse_scaler=lambda x: x,
                        denoise=False, rtol=tol_curr, atol=tol_curr,
                        method='RK23', eps=1e-5, device='cuda', atom_mask=atom_mask_side, 
                        indices=indices_side,
                        means=means_rotamer)
                
                score_fn_rotamer = self.rotamer_dynamics.get_score_fn(features, sde_rotamer, side_chain_only=True)
                
                coords_rotamer, _ = sampling_fn_side(score_fn_rotamer, z_side + means_rotamer)   
                
                completed = True
                
            except TimeOutException:
                
                print("Sampling timed out")
                sys.stdout.flush()
                
        
        full_mask = torch.logical_not((features["atoms_rotamer"][..., None] ==\
                                       torch.LongTensor(backbone_atom_indices).to(features["atom_mask_rotamer"].device)).any(-1)
                                      ).unsqueeze(-1).repeat(1,1,3)
        features["coordinates_rotamer"] = features["coordinates_rotamer"].masked_scatter(full_mask.unsqueeze(1).repeat(1, 
            coords_rotamer.shape[1], 1, 1), coords_rotamer)
       
        if clash_penalty:
            
            num_clashes = self._compute_clashes(features["coordinates_rotamer"],
                                                features["bond_matrix_rotamer"])
            
            clash_difference = num_clashes - curr_clashes
            
        all_bond_deviations = self._compute_bond_deviations(all_bond_indices, features["coordinates_rotamer"],
                                                            backbone=False)
        
        
        change = all_bond_deviations - all_bond_deviations_curr
        
        accept_p = torch.clip(torch.exp(-4*change), max=1)
        
        if clash_penalty:
            
            accept_p = accept_p*torch.exp(-1*clash_difference)
        
        accept_mask = accept_p > torch.rand_like(accept_p)
        
        z = z_init*(~accept_mask[..., None, None]) + z*(accept_mask[..., None, None])
        
        z_side = z_init_side*(~accept_mask[..., None, None]) + z_side*(accept_mask[..., None, None])
                
        all_bond_deviations_curr = all_bond_deviations_curr*(~accept_mask)\
            + all_bond_deviations*(accept_mask)
            
        
        if clash_penalty:
            
            curr_clashes = curr_clashes*(~accept_mask)\
                + num_clashes*(accept_mask)
                
            return z, z_side, all_bond_deviations_curr, curr_clashes
        
        else:
            
            return z, z_side, all_bond_deviations_curr
        
    def sample_with_bond_constraint(self, features, sde, rotamer_sde, protonation_sde,
                                    num_samples, num_steps=500, fix_chirality=True, d_peptide=False):
        
        atom_mask = features["atom_mask_backbone"]
        
        atom_indices = torch.sum(atom_mask, dim=-1) > 0
        
        indices = None
        
        means = None
 
            
        all_bond_indices = features["backbone_cyclic_bond_indices"]
        
        
        score_fn = self.get_score_fn(features, sde)
        
        # get initial samples
        
        sampling_fn = get_ode_sampler(sde, (atom_mask.shape[0],
                                                            num_samples,
                                                            torch.sum(atom_indices),
                                                            3), inverse_scaler=lambda x: x,
                        denoise=False, rtol=1e-2, atol=1e-2,
                        method='RK23', eps=1e-5, device='cuda', atom_mask=atom_mask, indices=indices,
                        means=means)
            

        
        # get initial values
        
        z = torch.randn((atom_mask.shape[0], num_samples, torch.sum(atom_indices),
                             3), requires_grad=False, device=atom_mask.device)*sde.temperature
        
        coords, _ = sampling_fn(score_fn, z)
            
        coords =  torch.sum(coords.unsqueeze(2).repeat(1, 1, coords.shape[2], 1, 1)*\
                   features["restore_indices"][:, None, :, :, None], axis=3)
            
                
        all_bond_deviations_curr = self._compute_bond_deviations(all_bond_indices, coords)
        
        curr_clashes = self._compute_clashes(coords, features["bond_matrix_backbone"],
                                             )

        
        bond_deviations_all_steps = []
        
        tolerances_all_steps = []
        
        curr_tolerance = 1e-2
        
        sampling_fn_curr = sampling_fn 
        
        start_time = time()
        
        step = 0
        
        while step < num_steps:
            
            
            tolerances_all_steps.append(curr_tolerance)
            
            z, all_bond_deviations_curr, curr_clashes = self._mcmc_bond_step(sampling_fn_curr, score_fn, z, 
                                                               all_bond_deviations_curr,
                                                               features, atom_mask, 
                                                               num_samples, atom_indices,
                                                               all_bond_indices,
                                                               2e-1*sde.temperature,
                                                               curr_clashes=curr_clashes)
                
            
            if step != num_steps - 1:
                bond_deviations_all_steps.append(all_bond_deviations_curr.cpu().detach().numpy())
            
            if (step+1) % 10 == 0 or step == 0:
                print("Step %i of %i, min bond deviation %f, elapsed time: %f, tolerance: %f" %(step+1, num_steps,
                                                                torch.min(all_bond_deviations_curr).cpu().detach().numpy(),
                                                                time()-start_time,
                                                                curr_tolerance))
            
                
                sys.stdout.flush()
                
                
            if (step+1) % 25 == 0 and not curr_tolerance > 1e-4:
                
                
                sampling_fn_lower_tolerance = get_ode_sampler(sde, (atom_mask.shape[0],
                                                            num_samples,
                                                            torch.sum(atom_indices),
                                                            3), inverse_scaler=lambda x: x,
                        denoise=False, rtol=curr_tolerance*0.1, atol=curr_tolerance*0.1,
                        method='RK23', eps=1e-5, device='cuda', atom_mask=atom_mask, indices=indices,
                        means=means)
                
                z_low, all_bond_deviations_curr_low, curr_clashes_low = self._mcmc_bond_step(sampling_fn_lower_tolerance, score_fn, z, 
                                                                   all_bond_deviations_curr,
                                                                   features, atom_mask, 
                                                                   num_samples, atom_indices,
                                                                   all_bond_indices,
                                                                   2e-1*sde.temperature,
                                                                   curr_clashes=curr_clashes,
                                                                   sample=False)
                                                                   
                differences = torch.mean(torch.abs(all_bond_deviations_curr\
                                                   - all_bond_deviations_curr_low))
                    
                if differences > 0.25:
                    
                    curr_tolerance = curr_tolerance*0.1
                    
                    sampling_fn_curr = sampling_fn_lower_tolerance
                    
            step += 1
            
        likelihood_sampling_fn = get_sampling_likelihood_fn(sde,
                                                         (atom_mask.shape[0],
                                                         num_samples,
                                                         torch.sum(atom_indices),
                                                         3), 
                                                          lambda x: x, device=atom_mask.device, 
                                                          hutchinson_type='Rademacher',
                                                          rtol=1e-3, atol=1e-3, 
                                                          method="RK23", eps=1e-5, 
                                                          atom_mask=atom_mask, indices=indices,
                                                          means=means, num_centers=None,
                                                          exact_trace=False)
            
        logp, relative_coords, _, _, _ = likelihood_sampling_fn(score_fn, z)
            
        coords =  torch.sum(relative_coords.unsqueeze(2).repeat(1, 1, relative_coords.shape[2], 1, 1)*\
                   features["restore_indices"][:, None, :, :, None], axis=3)
    
        if fix_chirality:
            coords = self._fix_chirality(coords, features["amino_acid_pos_backbone"],
                                                    features["atoms_backbone"], not d_peptide)
            
            coords = torch.FloatTensor(coords).to(features["atoms_backbone"].device)


        all_bond_deviations_curr = self._compute_bond_deviations(all_bond_indices, coords)
        
        curr_clashes = self._compute_clashes(coords, features["bond_matrix_backbone"])
        
        bond_deviations_all_steps.append(all_bond_deviations_curr.cpu().detach().numpy())
            
        
        
        '''
        sample rotamer and hydrogen
        '''
        
        for feature in features:

            if "names" not in feature and feature.endswith("_backbone"):

                features[feature] = features[feature].cpu().detach()


        features["coordinates_backbone"] = relative_coords
        
        
        features = self._compute_rotamer_edge_masks(coords, features)
        
         
        rotamer_coords, logp_rotamer = self.sample_rotamers(features, rotamer_sde, num_samples=num_samples,
                                                            likelihood=True, multiple_points=False)
        
          
        
        features = self._compute_hydrogen_edge_masks(rotamer_coords, features)
 
        features["coordinates_rotamer"] = features["coordinates_heavy_h"].clone()
        

        for feature in features:

            if "names" not in feature and feature.endswith("_rotamer"):

                features[feature] = features[feature].cpu().detach()

        torch.cuda.empty_cache()

        
       
                
        all_coords, lopg_hydrogen, energies, relative_coords =  self.sample_hydrogens(features, protonation_sde,
                                                                  num_samples=num_samples,
                                                                  likelihood=True,
                                                                  compute_energies=False, 
                                                                  multiple_points=True)

        
        features["coordinates_h"] = relative_coords
        
        
      
        
        return all_coords, bond_deviations_all_steps, tolerances_all_steps,\
            logp.cpu().detach().numpy() + logp_rotamer.cpu().detach().numpy() +\
                lopg_hydrogen.cpu().detach().numpy()
        

    def sample_with_bond_constraint_sidechain(self, features, sde, sde_rotamer, protonation_sde,
                                              num_samples, num_steps=1000, fix_chirality=True, d_peptide=False):
        
        atom_mask = features["atom_mask_backbone"]
        
        atom_indices = torch.sum(atom_mask, dim=-1) > 0
        
        indices = None
        
        means = None
        
        atom_mask_rotamer = features["atom_mask_rotamer"]
        
        atom_indices_rotamer = torch.sum(atom_mask_rotamer, dim=-1) > 0
        
        atom_mask_rotamer = atom_mask_rotamer[:, atom_indices_rotamer[0], :]
        
        indices_rotamer = features["amino_acid_pos_rotamer"][:, atom_indices_rotamer[0]] - 1
        
        num_centers = torch.sum(torch.logical_and(features["fragment_seq"] != 20, 
                                features["fragment_seq"] != residue_order["G"]))

        
        all_bond_indices = features["rotamer_cyclic_bond_indices"]
        
        score_fn = self.get_score_fn(features, sde)
        
        # get initial samples
        
        sampling_fn = get_ode_sampler(sde, (atom_mask.shape[0],
                                                            num_samples,
                                                            torch.sum(atom_indices),
                                                            3), inverse_scaler=lambda x: x,
                        denoise=False, rtol=1e-2, atol=1e-2,
                        method='RK23', eps=1e-5, device='cuda', atom_mask=atom_mask, indices=indices,
                        means=means)
            

        
        # get initial values
        
        z = torch.randn((atom_mask.shape[0], num_samples, torch.sum(atom_indices),
                             3), requires_grad=False, device=atom_mask.device)
        

        
        relative_coords, _ = sampling_fn(score_fn, z)
            
        coords =  torch.sum(relative_coords.unsqueeze(2).repeat(1, 1, relative_coords.shape[2], 1, 1)*\
                   features["restore_indices"][:, None, :, :, None], axis=3)
        
        features["coordinates_backbone"] = relative_coords
        
        features = self._compute_rotamer_edge_masks(coords, features)
        
        
        means_rotamer = torch.stack(
            [features["centroid_coordinates"]\
             [i][:,features['centroid_indices'][i]\
                 [features['amino_acid_pos_rotamer'][i]-1], :]\
                 for i in range(len(features["centroid_coordinates"]))])

        means_rotamer = means_rotamer[:, :, atom_indices_rotamer[0], :]
        
        
        sampling_fn_rotamer = get_ode_sampler(sde_rotamer, (atom_mask_rotamer.shape[0],
                                                    num_samples,
                                                    torch.sum(atom_indices_rotamer),
                                                    3), inverse_scaler=lambda x: x,
                denoise=False, rtol=1e-2, atol=1e-2,
                method='RK23', eps=1e-5, device='cuda', atom_mask=atom_mask_rotamer, 
                indices=indices_rotamer,
                means=means_rotamer)
            
            
        
        score_fn_rotamer = self.rotamer_dynamics.get_score_fn(features, sde_rotamer, side_chain_only=True)
        
        _, z_rotamer = sde_rotamer.prior_sampling((atom_mask_rotamer.shape[0],
                                                            num_samples,
                                                            torch.sum(atom_indices_rotamer),
                                                            3),
                                               atom_mask_rotamer,
                                               means_rotamer, 
                                               indices=indices_rotamer)
        
        z_rotamer = z_rotamer.to(atom_mask.device)
        
        coords_rotamer, _ = sampling_fn_rotamer(score_fn_rotamer, z_rotamer + means_rotamer)   
        
        
        
        full_mask = torch.logical_not((features["atoms_rotamer"][..., None] ==\
                                       torch.LongTensor(backbone_atom_indices).to(features["atom_mask_rotamer"].device)).any(-1)
                                      ).unsqueeze(-1).repeat(1,1,3)
        features["coordinates_rotamer"] = features["coordinates_rotamer"].masked_scatter(full_mask.unsqueeze(1).repeat(1, 
            coords_rotamer.shape[1], 1, 1), coords_rotamer)

         

        
        all_bond_deviations_curr = self._compute_bond_deviations(all_bond_indices, 
                                                                 features["coordinates_rotamer"],
                                                                 backbone=False)
        
        curr_clashes = self._compute_clashes(features["coordinates_rotamer"], features["bond_matrix_rotamer"])

        
        bond_deviations_all_steps = []
        
        tolerances_all_steps = []
        
        curr_tolerance = 1e-2
        
        sampling_fn_curr = sampling_fn 
        
        sampling_fn_curr_rotamer = sampling_fn_rotamer
        
        start_time = time()
        
        step = 0
           
        while step < num_steps:
            
            tolerances_all_steps.append(curr_tolerance)
            
            z, z_rotamer, all_bond_deviations_curr, curr_clashes = self._mcmc_bond_step_side_chain(sampling_fn_curr, score_fn, z, 
                                                                                        sampling_fn_curr_rotamer,
                                                                                        sde_rotamer, z_rotamer, all_bond_deviations_curr,
                                                                                        features, atom_mask, num_samples, atom_indices,
                                                                                        atom_mask_rotamer, atom_indices_rotamer, means_rotamer, indices_rotamer,
                                                                                        all_bond_indices, 
                                                               2e-1*sde.temperature,2e-1*sde_rotamer.temperature,
                                                               curr_clashes=curr_clashes, tol_curr=curr_tolerance)
                
            
            bond_deviations_all_steps.append(all_bond_deviations_curr.cpu().detach().numpy())
            
            if (step+1) % 10 == 0 or step == 0:
                print("Step %i, min bond deviation %f, elapsed time: %f, tolerance: %f" %(step+1,
                                                                torch.min(all_bond_deviations_curr).cpu().detach().numpy(),
                                                                time()-start_time,
                                                                curr_tolerance))
            
                
                sys.stdout.flush()
                
                
            if (step+1) % 25 == 0 and not curr_tolerance > 1e-4:
                
                
                sampling_fn_lower_tolerance = get_ode_sampler(sde, (atom_mask.shape[0],
                                                            num_samples,
                                                            torch.sum(atom_indices),
                                                            3), inverse_scaler=lambda x: x,
                        denoise=False, rtol=curr_tolerance*0.1, atol=curr_tolerance*0.1,
                        method='RK23', eps=1e-5, device='cuda', atom_mask=atom_mask, indices=indices,
                        means=means)
                
                sampling_fn_rotamer_lower_tolerance = get_ode_sampler(sde_rotamer, (atom_mask_rotamer.shape[0],
                                                            num_samples,
                                                            torch.sum(atom_indices_rotamer),
                                                            3), inverse_scaler=lambda x: x,
                        denoise=False, rtol=curr_tolerance*0.1, atol=curr_tolerance*0.1,
                        method='RK23', eps=1e-5, device='cuda', atom_mask=atom_mask_rotamer, 
                        indices=indices_rotamer,
                        means=means_rotamer, num_centers=num_centers)
                
                z_low, z_rotamer_low, all_bond_deviations_curr_low, curr_clashes_low = self._mcmc_bond_step_side_chain(sampling_fn_lower_tolerance, score_fn, z, 
                                                                                        sampling_fn_rotamer_lower_tolerance,
                                                                                        sde_rotamer, z_rotamer, all_bond_deviations_curr,
                                                                                        features, atom_mask, num_samples, atom_indices,
                                                                                        atom_mask_rotamer, atom_indices_rotamer, means_rotamer, indices_rotamer,
                                                                                        all_bond_indices, 
                                                               2e-1*sde.temperature,2e-1*sde_rotamer.temperature,
                                                               curr_clashes=curr_clashes, 
                                                               tol_curr=curr_tolerance,
                                                               sample=False)
                                                                   
                differences = torch.mean(torch.abs(all_bond_deviations_curr\
                                                   - all_bond_deviations_curr_low))
                    
                if differences > 0.25:
                    
                    curr_tolerance = curr_tolerance*0.1
                    
                    sampling_fn_curr = sampling_fn_lower_tolerance
                
            step += 1
                        
        likelihood_sampling_fn = get_sampling_likelihood_fn(sde,
                                                         (atom_mask.shape[0],
                                                         num_samples,
                                                         torch.sum(atom_indices),
                                                         3), 
                                                          lambda x: x, device=atom_mask.device, 
                                                          hutchinson_type='Rademacher',
                                                          rtol=1e-3, atol=1e-3, 
                                                          method="RK23", eps=1e-5, 
                                                          atom_mask=atom_mask, indices=indices,
                                                          means=means, num_centers=None,
                                                          exact_trace=False)
            
        logp, relative_coords, _, _, _ = likelihood_sampling_fn(score_fn, z)
            
        coords =  torch.sum(relative_coords.unsqueeze(2).repeat(1, 1, coords.shape[2], 1, 1)*\
                   features["restore_indices"][:, None, :, :, None], axis=3)

    
        if fix_chirality:
            coords = self._fix_chirality(coords, features["amino_acid_pos_backbone"],
                                                    features["atoms_backbone"], not d_peptide)
            
            coords = torch.FloatTensor(coords).to(features["atoms_backbone"].device)
            
        features["coordinates_backbone"] = relative_coords
        
        features = self._compute_rotamer_edge_masks(coords, features)
        
        means_rotamer = torch.stack(
            [features["centroid_coordinates"]\
             [i][:,features['centroid_indices'][i]\
                 [features['amino_acid_pos_rotamer'][i]-1], :]\
                 for i in range(len(features["centroid_coordinates"]))])

        means_rotamer = means_rotamer[:, :, atom_indices_rotamer[0], :]
        
        score_fn_rotamer = self.rotamer_dynamics.get_score_fn(features, sde_rotamer, side_chain_only=True)
        
        likelihood_sampling_fn = get_sampling_likelihood_fn(sde_rotamer,
                                                         (atom_mask_rotamer.shape[0],
                                                         num_samples,
                                                         torch.sum(atom_indices_rotamer),
                                                         3), 
                                                          lambda x: x, device=atom_mask.device, 
                                                          hutchinson_type='Rademacher',
                                                          rtol=1e-3, atol=1e-3, 
                                                          method="RK23", eps=1e-5, 
                                                          atom_mask=atom_mask_rotamer, indices=indices_rotamer,
                                                          means=means_rotamer, num_centers=num_centers,
                                                          exact_trace=False)        
        
        
        logp_rotamer, coords_rotamer, _, _, _ = likelihood_sampling_fn(score_fn_rotamer, z_rotamer + means_rotamer)   
        
        full_mask = torch.logical_not((features["atoms_rotamer"][..., None] ==\
                                       torch.LongTensor(backbone_atom_indices).to(features["atom_mask_rotamer"].device)).any(-1)
                                      ).unsqueeze(-1).repeat(1,1,3)
       

        features = self._compute_hydrogen_edge_masks(coords_rotamer, features)

        features["coordinates_rotamer"] = features["coordinates_heavy_h"].clone()
                
        
        
        all_bond_deviations_curr = self._compute_bond_deviations(all_bond_indices, features["coordinates_rotamer"],
                                                                 backbone=False)
        
        curr_clashes = self._compute_clashes(features["coordinates_rotamer"], features["bond_matrix_rotamer"])
        
        bond_deviations_all_steps.append(all_bond_deviations_curr.cpu().detach().numpy())
       

        
        for feature in features:

            if "names" not in feature and feature.endswith("_rotamer"):

                features[feature] = features[feature].cpu().detach()

        torch.cuda.empty_cache()

        
       
                
        all_coords, lopg_hydrogen, energies, relative_coords =  self.sample_hydrogens(features, protonation_sde,
                                                                  num_samples=num_samples,
                                                                  likelihood=True,
                                                                  compute_energies=False, 
                                                                  multiple_points=True)

        
        features["coordinates_h"] = relative_coords
        
        
      
        
        return all_coords, bond_deviations_all_steps, tolerances_all_steps,\
            logp.cpu().detach().numpy() + logp_rotamer.cpu().detach().numpy() +\
                lopg_hydrogen.cpu().detach().numpy()
                
        
   
    
    def get_score_fn(self, features, sde, return_embedding=False):
                
        sequence, positions = self.hypernetwork.process_tensors(features["fragment_seq"])

        if return_embedding:
            w_x, b_x, w_t, w_b_t, embedding = self.hypernetwork(sequence, positions, return_embedding)
        else:
            w_x, b_x, w_t, w_b_t = self.hypernetwork(sequence, positions, return_embedding)
        
        batch_size = features["fragment_seq"].shape[0]
        
        w_x_reshaped = []
        b_x_reshaped = []
        w_t_reshaped = []
        w_b_t_reshaped = []

        prev_idx_w_x = 0
        prev_idx_b_x = 0
        prev_idx_w_t = 0
        prev_idx_w_b_t = 0

        for layer in self.mainnet_config.dims:
            
            w_x_reshaped.append(w_x[prev_idx_w_x].reshape(batch_size, layer[0], layer[1]))
            prev_idx_w_x += 1

            b_x_reshaped.append(b_x[prev_idx_b_x].reshape(batch_size, layer[1]))
            prev_idx_b_x += 1

            w_t_reshaped.append(w_t[prev_idx_w_t].reshape(batch_size, 1, layer[1]))
            prev_idx_w_t += 1

            w_b_t_reshaped.append(w_b_t[prev_idx_w_b_t].reshape(batch_size, 1, layer[1]))
            prev_idx_w_b_t += 1
                
        

        def score_backbone(perturbed_data, t):
                
            _, std = sde.marginal_prob(torch.zeros(perturbed_data.shape).to(
                features["fragment_seq"].device), t)
            
            prediction = self.backbone_dynamics(t, perturbed_data, 
                                    features["fragment_seq"], 
                                    features["atoms_backbone"], 
                                    features["amino_acid_pos_backbone"], 
                                    features["bond_matrix_backbone"],
                                    features["edge_mask_backbone"], 
                                    features["atom_mask_backbone"],
                                    features["restore_indices"],
                                    w_x_reshaped, b_x_reshaped,
                                    w_t_reshaped, w_b_t_reshaped)
            
            prediction = -1*prediction/std[:, None, None, None].to(prediction.device)
            
            return prediction

                                    
            
        return score_backbone


    def sample_backbone(self, features, sde, num_samples=1, rtol=1e-3, atol=1e-3, solver='RK23',
                        likelihood=True, exact_trace=False, differentiable=False, multiple_points=False):
        
        atom_mask = features["atom_mask_backbone"].clone()
        

        atom_indices = torch.sum(atom_mask, dim=-1) > 0
        
        
        indices = None
        
        means = None
        
        if differentiable:
        
            score_fn_backbone = get_wrapped_backbone_score_fn(self, features, sde, means, atom_mask)
            
        else:
            score_fn_backbone = self.get_score_fn(features, sde)
        
        if differentiable:
            
            sampling_fn_backbone = get_differentiable_ode_sampler(sde,
                                                         (atom_mask.shape[0],
                                                         num_samples,
                                                         atom_mask.shape[1],
                                                         3), 
                                                          rtol=rtol, atol=atol, 
                                                          eps=1e-5, likelihood=True,
                                                          atom_mask=atom_mask, indices=indices,
                                                          means=means, num_centers=None)
            
            z, delta_logp, prior_logp = sampling_fn_backbone(score_fn_backbone)
        
        elif likelihood:
            
            sampling_fn_backbone = get_sampling_likelihood_fn(sde,
                                                         (atom_mask.shape[0],
                                                         num_samples,
                                                         torch.sum(atom_indices),
                                                         3), 
                                                          lambda x: x, device=atom_mask.device, 
                                                          hutchinson_type='Rademacher',
                                                          rtol=rtol, atol=atol, 
                                                          method=solver, eps=1e-5, 
                                                          atom_mask=atom_mask, indices=indices,
                                                          means=means, num_centers=None,
                                                          exact_trace=exact_trace)
            
            logp_backbone, z, nfe_backbone, _, _ = sampling_fn_backbone(score_fn_backbone)   
            
        else:
            sampling_fn_backbone = get_ode_sampler(sde, (atom_mask.shape[0],
                                                         num_samples,
                                                         torch.sum(atom_indices),
                                                         3), 
                                                   lambda x: x, device=atom_mask.device,
                                                   denoise=False, rtol=1e-4, atol=1e-4,
                                                   method='RK23', eps=1e-5, 
                                                   atom_mask=atom_mask, indices=indices,
                                                   means=means, multiple_points=multiple_points)
            
            z, nfe_backbone = sampling_fn_backbone(score_fn_backbone)   
            
            logp_backbone = None
            
        z_init = z.clone()
        
        z =  torch.sum(z.unsqueeze(2).repeat(1, 1, z.shape[2], 1, 1)*\
           features["restore_indices"][:, None, :, :, None], axis=3)
        
        if not differentiable:
            return z, logp_backbone, z_init
        else:
            return z, delta_logp, prior_logp
             
    def compute_likelihood_backbone(self, features, sde, rtol=1e-3, atol=1e-3):
        
        atom_mask = features["atom_mask_backbone"].clone()
        
        indices = None
        
        means = None
        
        score_fn_backbone = get_wrapped_backbone_score_fn(self, features, sde, means, atom_mask)      
        
        
        sampling_fn_backbone = get_differentiable_ode_sampler(sde,
                                                     (atom_mask.shape[0],
                                                     features["coordinates_backbone"].shape[1],
                                                     atom_mask.shape[1],
                                                     3), 
                                                      rtol=rtol, atol=atol, 
                                                      eps=1e-5, only_likelihood=True,
                                                      atom_mask=atom_mask, indices=indices,
                                                      means=means, num_centers=None)
        
        logp = sampling_fn_backbone(score_fn_backbone, features["coordinates_backbone"])
        
        
        return logp
            
        
    def sample_rotamers(self, features, sde, num_samples=1, rtol=1e-3, atol=1e-3, solver='RK23',
                        likelihood=True, exact_trace=False, differentiable = False, multiple_points=False):
        
        
        atom_mask = features["atom_mask_rotamer"]
        
        atom_indices = torch.sum(atom_mask, dim=-1) > 0
      
        atom_mask = atom_mask[:, atom_indices[0], :]
        
        indices = features["amino_acid_pos_rotamer"][:, atom_indices[0]] - 1
        
        num_centers = torch.sum(torch.logical_and(features["fragment_seq"] != 20, 
                                features["fragment_seq"] != residue_order["G"]))
        
        means = torch.stack(
            [features["centroid_coordinates"]\
             [i][:,features['centroid_indices'][i]\
                 [features['amino_acid_pos_rotamer'][i]-1], :]\
                 for i in range(len(features["centroid_coordinates"]))])

        means = means[:, :, atom_indices[0], :]
        
        # get sampling function
        
        if not differentiable:
            score_fn_rotamer = self.rotamer_dynamics.get_score_fn(features, sde, side_chain_only=True)
        
        else:
            
            score_fn_rotamer = get_wrapped_rotamer_score_fn(self.rotamer_dynamics, features, sde, 
                                                            means, atom_mask,
                                                            side_chain_only=True)
        
        if differentiable:
            
            sampling_fn_rotamer = get_differentiable_ode_sampler(sde,
                                                          (atom_mask.shape[0],
                                                         num_samples,
                                                         torch.sum(atom_indices),
                                                         3), 
                                                          rtol=rtol, atol=atol, 
                                                          eps=1e-5, likelihood=True,
                                                          atom_mask=atom_mask, indices=indices,
                                                          means=means, num_centers=num_centers)
          
            
            z, delta_logp, prior_logp = sampling_fn_rotamer(score_fn_rotamer)       
            
        elif likelihood:
            
            sampling_fn_rotamer = get_sampling_likelihood_fn(sde,
                                                          (atom_mask.shape[0],
                                                         num_samples,
                                                         torch.sum(atom_indices),
                                                         3), 
                                                          lambda x: x, device=atom_mask.device, 
                                                          hutchinson_type='Rademacher',
                                                          rtol=rtol, atol=atol, 
                                                          method=solver, eps=1e-5, 
                                                          atom_mask=atom_mask, indices=indices,
                                                          means=means, num_centers=num_centers,
                                                          exact_trace=exact_trace)
            
            logp_rotamer, z, nfe_rotamer, _, _ = sampling_fn_rotamer(score_fn_rotamer)   
            
        else:
            sampling_fn_rotamer = get_ode_sampler(sde,  (atom_mask.shape[0],
                                                         num_samples,
                                                         torch.sum(atom_indices),
                                                         3), 
                                                   lambda x: x, device=atom_mask.device,
                                                   denoise=False, rtol=1e-4, atol=1e-4,
                                                   method='RK23', eps=1e-5, 
                                                   atom_mask=atom_mask, indices=indices,
                                                   means=means, multiple_points=multiple_points)
            
            z, nfe_rotamer = sampling_fn_rotamer(score_fn_rotamer)   
            
            logp_rotamer = None
            
        if not differentiable:
            return z, logp_rotamer
        else:
            return z, delta_logp, prior_logp
            
    
    def set_temperature(self, temperature):
        
        self.temperature = temperature
        
    def sample_hydrogens(self, features, sde, num_samples=1, 
               rtol=1e-3, atol=1e-3, solver='RK23', likelihood=True, compute_energies=True,
               exact_trace=False, differentiable = False, compute_gradients=False, multiple_points=False,
               unnormalized_energy=False):
        


        
        atom_mask = features["atom_mask_h"]
            
        # get sampling function
        if not differentiable:
            score_fn_hydrogen = self.hydrogen_dynamics.get_score_fn(features, sde, 
                                                                )
        else:
            
        
            score_fn_hydrogen = get_wrapped_hydrogen_score_fn(self.hydrogen_dynamics, features, sde, means=None, atom_mask=atom_mask)
        
        if differentiable:
            
            sampling_fn_hydrogen = get_differentiable_ode_sampler(sde,
                                                                  (atom_mask.shape[0],
                                                                   num_samples, atom_mask.shape[1], 3),
                                                          rtol=1e-3, atol=1e-3, 
                                                          eps=1e-5, likelihood=True,
                                                          atom_mask=atom_mask, indices=None,
                                                          means=None, num_centers=None)
            
            z, delta_logp, prior_logp = sampling_fn_hydrogen(score_fn_hydrogen)   
            
        elif likelihood:
            
            sampling_fn_hydrogen = get_sampling_likelihood_fn(sde,
                                                          (atom_mask.shape[0],
                                                             num_samples, atom_mask.shape[1], 3), 
                                                          lambda x: x, device=atom_mask.device, 
                                                          hutchinson_type='Rademacher',
                                                          rtol=rtol, atol=atol, 
                                                          method=solver, eps=1e-5, 
                                                          atom_mask=atom_mask, indices=None,
                                                          means=None, num_centers=None,
                                                          exact_trace=exact_trace)
            
            logp_hydrogen, z, nfe_hydrogen, delta_logp, prior_logp = sampling_fn_hydrogen(score_fn_hydrogen)   
            
        else:
            sampling_fn_hydrogen = get_ode_sampler(sde, (atom_mask.shape[0],
                                                             num_samples, atom_mask.shape[1], 3), 
                                                   lambda x: x, device=atom_mask.device,
                                                   denoise=False, rtol=1e-4, atol=1e-4,
                                                   method='RK23', eps=1e-5, 
                                                   atom_mask=atom_mask, indices=None,
                                                   means=None, multiple_points=multiple_points)
            
    
            z, nfe_hydrogen = sampling_fn_hydrogen(score_fn_hydrogen) 
            
            logp_hydrogen = None
        
        # fill in sampled protons
        
        sampled_coords = z
        
        z_hydrogen = sampled_coords + torch.stack([features["coordinates_heavy_h"]\
                                               [i, :, features["bound_atom"][i], :]\
                                                   for i in range(len(z))])
            
       
        peptide_seq, x_heavy, atom_mask_heavy, atom_mask_h, atom_names_heavy,\
            atom_names_h, amino_acid_pos_heavy, amino_acid_pos_h = features["fragment_seq"],  features["coordinates_heavy_h"],\
                features["atom_mask_heavy"], features["atom_mask_h"], features["atom_names_heavy"],\
                    features["atom_names_h"], features["amino_acid_pos_heavy"],\
                        features["amino_acid_pos_h"]
            
        atom_mask = torch.cat([atom_mask_heavy, atom_mask_h], dim=1)
            
        if x_heavy.shape[1] != z_hydrogen.shape[1]:
            x_heavy = x_heavy.repeat(1, z_hydrogen.shape[1], 1, 1)
        
        x = torch.cat([x_heavy, z_hydrogen], dim = 2)
            
       
        integrators = []
        
        if compute_energies:
                    
            energy_modules = nn.ModuleList()
            
            for pep in range(x.shape[0]):
                integrator = openmm.LangevinIntegrator(self.temperature*unit.kelvin,
                       1.0/unit.picoseconds,
                        2.0*unit.femtoseconds)
        
                atom_names = atom_names_heavy[pep] + atom_names_h[pep]
                
                amino_acid_pos = torch.cat((amino_acid_pos_heavy[pep, :len(atom_names_heavy[pep])], 
                                        amino_acid_pos_h[pep, :len(atom_names_h[pep])]),
                                       dim=0)
             
                if not peptide_seq.is_cuda:
                    system = create_system(peptide_seq[pep].numpy(), 
                                           amino_acid_pos.numpy(), 
                                           atom_names, self.forcefield)  
                else:
                    
                    system = create_system(peptide_seq[pep].cpu().numpy(), 
                                           amino_acid_pos.cpu().numpy(), 
                                           atom_names, self.forcefield) 
                    
                bridge = OpenMMBridge(system, integrator, platform_name="CPU", n_workers=1)
            
                energy = OpenMMEnergy(bridge = bridge)

                energy_modules.append(energy)
                
                integrators.append(integrator)
                    
                 
    
            energy_values_all = []
                
            all_coords_full = []
            
            with torch.no_grad() if not differentiable and not compute_gradients else nullcontext():
                for pep in range(x.shape[0]):
                    
                    atom_names = atom_names_heavy[pep] + atom_names_h[pep]
                    
                    atom_mask_x = atom_mask[pep].unsqueeze(0).repeat(x.shape[1], 1, 1)
                    
                    x_full = torch.masked_select(x[pep], atom_mask_x.bool()).view(atom_mask_x.shape[0], -1, 3)
                    
                    
                    all_coords_full.append(x_full)
                    
                    re_order = [torch.concat((torch.nonzero(amino_acid_pos_heavy[pep, :len(atom_names_heavy[pep])]-1 == i).flatten(),
                                             torch.nonzero(amino_acid_pos_h[pep, :len(atom_names_h[pep])] -1 == i).flatten()+len(atom_names_heavy[pep])))\
                                for i in range(peptide_seq.shape[1])]
                    re_order = torch.concat(re_order)
                    
                    x_full = x_full[:, re_order, :]
                    
    
                    energy_values = energy_modules[pep].energy(x_full.view(-1, len(atom_names),
                        3)/10).to(x.device)
                    
                    atom_mask_x = atom_mask[pep].unsqueeze(0)
    
                  
                        
                    if unnormalized_energy:
                        
                        energy_values = energy_values*(310*unit.kelvin*unit.MOLAR_GAS_CONSTANT_R).\
                            value_in_unit(unit.kilojoules_per_mole)
    
                    energy_values_all.append(energy_values)
                    
                  
                        
                        
                energy_values = torch.stack(energy_values_all, dim=0).squeeze(-1)
        
        else:
            all_coords_full = []
            
            with torch.no_grad() if not differentiable and not compute_gradients else nullcontext():
                for pep in range(x.shape[0]):
                    
                    atom_names = atom_names_heavy[pep] + atom_names_h[pep]
                    
                    atom_mask_x = atom_mask[pep].unsqueeze(0).repeat(x.shape[1], 1, 1)
                    
                    x_full = torch.masked_select(x[pep], atom_mask_x.bool()).view(atom_mask_x.shape[0], -1, 3)
                    
                    all_coords_full.append(x_full)
                   
                    
            energy_values = None
        
        if not differentiable and not compute_gradients:
            return torch.stack(all_coords_full), logp_hydrogen, energy_values, sampled_coords
        else:
            return torch.stack(all_coords_full), delta_logp, prior_logp, energy_values, sampled_coords
        

    def _compute_rotamer_edge_masks(self, backbone_coords, features):
        
        all_coordinates_centroid = []
        
        all_coordinates_backbone = []
        
        all_edge_masks = []
        
        for index, item in enumerate(features["atoms_backbone"]):
            coordinates_centroid = backbone_coords[index, :, item == 4]
            coordinates_centroid = coordinates_centroid[:, :torch.sum(torch.logical_and(features["fragment_seq"][index] != residue_order["G"],
                                                                      features["fragment_seq"][index] != 20))]
            coordinates_backbone = backbone_coords[index, :, item != 4]
    
                
            coordinates_centroid_rotamer = torch.zeros((coordinates_centroid.shape[0], features["fragment_seq"].shape[1],
                                                        3), device=coordinates_centroid.device)
            coordinates_centroid_rotamer[:, :coordinates_centroid.shape[1], :] = coordinates_centroid
            
            all_coordinates_centroid.append(coordinates_centroid_rotamer)
              
            # compute edge mask based on centroid distances
            _, centroid_distances = cdist(coordinates_centroid_rotamer, 
                                       coordinates_centroid_rotamer)
            
            edge_cutoff_mask = centroid_distances[:, features['centroid_indices'][index]\
                                                  [features["amino_acid_pos_rotamer"][index] - 1], :]\
                                                  [:, :, features['centroid_indices'][index][features["amino_acid_pos_rotamer"][index] - 1]] < 8.0
            edge_mask = torch.ones((edge_cutoff_mask.shape[0], features["amino_acid_pos_rotamer"].shape[1], 
                                 features["amino_acid_pos_rotamer"].shape[1]), device=centroid_distances.device)
            edge_mask[:, len(features["atom_names_rotamer"][index]):, :] = 0
            edge_mask[:, :, len(features["atom_names_rotamer"][index]):] = 0
            edge_mask[:, features["atoms_rotamer"][index] < 4, :] = 0
            edge_mask[:, :edge_cutoff_mask.shape[1], :edge_cutoff_mask.shape[1]]\
                    = edge_mask[:, :edge_cutoff_mask.shape[1], :edge_cutoff_mask.shape[1]] * edge_cutoff_mask
            
            all_edge_masks.append(edge_mask)
            
            coordinates_backbone_rotamer = torch.zeros((coordinates_centroid.shape[0],
                                                        features["amino_acid_pos_rotamer"].shape[1],
                                                        3), device=coordinates_backbone.device)
        
            mask = torch.zeros(coordinates_backbone_rotamer.shape,  device=coordinates_backbone.device,
               dtype=torch.bool)
            mask[:, features["atoms_rotamer"][index] < 4, :] = 1
            mask[:, len(features["atom_names_rotamer"][index]):, :] = 0
            
            num_backbone = len([i for i in features["atom_names_backbone"][index] if i != "CEN"])
            
            coordinates_backbone_rotamer = coordinates_backbone_rotamer.masked_scatter(mask, coordinates_backbone[:, :num_backbone, :])
            
            all_coordinates_backbone.append(coordinates_backbone_rotamer)
            
        all_coordinates_centroid = torch.stack(all_coordinates_centroid, dim=0)
        
        all_coordinates_backbone = torch.stack(all_coordinates_backbone, dim=0)
        all_edge_masks = torch.stack(all_edge_masks, dim=0)
        
        features["coordinates_rotamer"] = all_coordinates_backbone
        features["centroid_coordinates"] = all_coordinates_centroid
        features["edge_mask_rotamer"] = all_edge_masks
        
        return features

    
    def _compute_hydrogen_edge_masks(self, rotamer_coords, features):
        
        full_mask = torch.logical_not((features["atoms_heavy"][..., None] ==\
                                       torch.LongTensor(backbone_atom_indices).to(features["atom_mask_rotamer"].device)).any(-1)
                                      ).unsqueeze(-1).repeat(1,1,3)
        features["coordinates_heavy_h"] = features["coordinates_rotamer"].masked_scatter(full_mask.unsqueeze(1).repeat(1, 
            rotamer_coords.shape[1], 1, 1), rotamer_coords)

        
        all_edge_masks_full = []
        all_edge_masks_h_heavy_full = []
        
        for index_2 in range(features["coordinates_heavy_h"].shape[1]):
            
            all_edge_masks = []

            all_edge_masks_h_heavy = []

            for index, coords in enumerate(features["coordinates_heavy_h"][:, index_2, ...]):
                
                _, distances = cdist(coords, coords)
                neighbors = torch.nonzero(distances <= 4)
            
                #generate edge mask between heavy atoms
                edge_mask = torch.zeros((len(features["atoms_rotamer"][index]), 
                                         len(features["atoms_rotamer"][index])), device=rotamer_coords.device)
                
                edge_mask[neighbors[:, 0], neighbors[:, 1]] = 1
            
                edge_mask[len(features["atom_names_rotamer"][index]):, :] = 0
                edge_mask[:, len(features["atom_names_rotamer"][index]):] = 0
                
                all_edge_masks.append(edge_mask)
            
                # generate edge mask between hydrogens and heavy atoms
            
                edge_mask_h_heavy = edge_mask[features["bound_atom"][index], :]
                edge_mask_h_heavy_full = torch.zeros((len(features["bound_atom"][index]), 
                                                   edge_mask_h_heavy.shape[1]), device=rotamer_coords.device)
                edge_mask_h_heavy_full[:edge_mask_h_heavy.shape[0], :edge_mask_h_heavy.shape[1]] = edge_mask_h_heavy
                edge_mask_h_heavy_full[len(features["atom_names_h"][index]):, :] = 0
                edge_mask_h_heavy_full[:, len(features["atom_names_rotamer"][index]):] = 0
            
                all_edge_masks_h_heavy.append(edge_mask_h_heavy_full)
            
            all_edge_masks_full.append(torch.stack(all_edge_masks, dim=0))

            all_edge_masks_h_heavy_full.append(torch.stack(all_edge_masks_h_heavy, dim=0))

        all_edge_masks = torch.stack(all_edge_masks_full, dim=1)
        all_edge_masks_h_heavy = torch.stack(all_edge_masks_h_heavy_full, dim=1)

        features["edge_mask_heavy_neighbors"] = all_edge_masks
        features["edge_mask_h_heavy"] = all_edge_masks_h_heavy
    
        return features
    
    def sample_all_atoms(self, features, backbone_sde, rotamer_sde, protonation_sde, num_samples, 
                         likelihood=False, compute_energies=True, protonate=True, sample_rotamers=True,
                         multiple_points=False, fix_chirality=False, d_peptide=False, unnormalized_energy=False):
        
        complete = False
        while not complete:
            
            try:
                backbone_coords, logp_backbone, relative_coords = self.sample_backbone(features, backbone_sde, num_samples=num_samples,
                                                                                       likelihood=likelihood, multiple_points=multiple_points)
                
            except TimeOutException:
                print("Backbone generation timed out")
                sys.stdout.flush()
                continue
                
            else:
                complete = True
                    
        if fix_chirality:
            backbone_coords = self._fix_chirality(backbone_coords, features["amino_acid_pos_backbone"],
                                                    features["atoms_backbone"], not d_peptide)
            
            backbone_coords = torch.FloatTensor(backbone_coords).to(features["atoms_backbone"].device)

        
        features["coordinates_backbone"] = relative_coords
        
        

       
        for feature in features:

            if "names" not in feature and feature.endswith("_backbone"):

                features[feature] = features[feature].cpu().detach()


        torch.cuda.empty_cache()

        if not multiple_points:
            features = self._compute_rotamer_edge_masks(backbone_coords, features)
            
        else:
            features = self._compute_rotamer_edge_masks(backbone_coords[:, -1:, :, :], features)
        
        if not sample_rotamers:
             return backbone_coords, logp_backbone, None, features
         
        complete = False
        while not complete:
           
            
            try:
                rotamer_coords, logp_rotamer = self.sample_rotamers(features, rotamer_sde, num_samples=num_samples,
                                                                    likelihood=likelihood, multiple_points=multiple_points)
        
            except TimeOutException:
                print("Rotamer generation timed out")
                sys.stdout.flush()
                continue
                
            else:
                complete = True   
        
        
            
        if not multiple_points:
            features = self._compute_hydrogen_edge_masks(rotamer_coords, features)
        else:
            features = self._compute_hydrogen_edge_masks(rotamer_coords[:, -1:, :, :], features)
            
            alt_features = self._compute_hydrogen_edge_masks(rotamer_coords, copy.deepcopy(features))
            
        features["coordinates_rotamer"] = features["coordinates_heavy_h"].clone()

        for feature in features:

            if "names" not in feature and feature.endswith("_rotamer"):

                features[feature] = features[feature].cpu().detach()

        torch.cuda.empty_cache()

        if not protonate and likelihood:
             return features["coordinates_rotamer"], logp_backbone + logp_rotamer, None, features
        elif not protonate:
            return features["coordinates_rotamer"], None, None, features
        
        complete = False
        while not complete:
            
            try:
                
                all_coords, lopg_hydrogen, energies, relative_coords =  self.sample_hydrogens(features, protonation_sde,
                                                                          num_samples=num_samples,
                                                                          likelihood=likelihood,
                                                                          compute_energies=compute_energies, 
                                                                          multiple_points=multiple_points,
                                                                          unnormalized_energy=unnormalized_energy)
                
            except TimeOutException:
                print("Hydrogen generation timed out")
                sys.stdout.flush()
                continue
                
            else:
                complete = True  
        
        features["coordinates_h"] = relative_coords
        
        
        if multiple_points:
            return backbone_coords, alt_features["coordinates_heavy_h"], all_coords

        
        if likelihood:
            
            return all_coords, logp_backbone + logp_rotamer +  lopg_hydrogen, energies, features
        
        else:
            return all_coords, None, energies, features

    def sample_all_atoms_differentiable(self, features, backbone_sde, rotamer_sde, protonation_sde, num_samples,
            differentiable_protonation=False):
        
        backbone_coords, delta_logp_backbone, prior_logp_backbone = self.sample_backbone(features, backbone_sde, num_samples=num_samples,
                                                                               differentiable=True)
        
        torch.cuda.empty_cache()
        
        features = self._compute_rotamer_edge_masks(backbone_coords, features)
       

        rotamer_coords, delta_logp_rotamer, prior_logp_rotamer = self.sample_rotamers(features, rotamer_sde, num_samples=num_samples,
                                                            differentiable=True)
        
        features = self._compute_hydrogen_edge_masks(rotamer_coords, features)
        
        features["coordinates_rotamer"] = features["coordinates_heavy_h"].clone()
        


        torch.cuda.empty_cache()

    
                
        all_coords, delta_logp_hydrogen, prior_logp_hydrogen, energies, relative_coords  =  self.sample_hydrogens(features, protonation_sde,
                                                                  num_samples=num_samples,
                                                                  differentiable=differentiable_protonation,
                                                                  compute_energies=True,
                                                                  likelihood=True,
                                                                  compute_gradients=True)

        features["coordinates_h"] = relative_coords
        
        
        permuted_coords  = torch.stack([torch.stack([relative_coords[i][:, j, ...]\
                                                     for j in features["all_permutations"][i]])\
                                        for i in range(features["all_permutations"].shape[0])], axis=0)
        
        features["permuted_coords"] = permuted_coords
        
        torch.cuda.empty_cache()
        
        return energies, delta_logp_backbone + delta_logp_hydrogen + delta_logp_rotamer
        
    def _fix_chirality(self, all_coords, amino_acid_pos, atoms, l):
        
        amino_acid_pos = amino_acid_pos.cpu().detach().numpy()

        atoms = atoms.cpu().detach().numpy()

        all_coords = all_coords.cpu().detach().numpy()

        for index, coords in enumerate(all_coords):
        
            for index_2, coord in enumerate(coords):
            
                num_d = 0
                
                total = 0
                
                for pos in np.unique(amino_acid_pos[index]):
                    
                    if 4 not in atoms[index][amino_acid_pos[index]==pos]:
                        continue
                    
                    ca = coord[np.logical_and(amino_acid_pos[index]==pos,
                                              atoms[index]==atom_order["CA"])][0]
                    n = coord[np.logical_and(amino_acid_pos[index]==pos,
                                              atoms[index]==atom_order["N"])][0]
                    c = coord[np.logical_and(amino_acid_pos[index]==pos,
                                              atoms[index]==atom_order["C"])][0]
                    cen = coord[np.logical_and(amino_acid_pos[index]==pos,
                                              atoms[index]==4)][0] 
                    
                    if np.dot(ca - n,
                          np.cross(ca - c, 
                                   ca - cen)) > 0:
                      
                    
                        num_d += 1
                        
                    total += 1
                
                is_l = num_d <= total/2
                
                if (l and not is_l) or (not l and is_l):
                    coord[:, 0] = -1*coord[:, 0]
                    all_coords[index, index_2] = coord
        
        return all_coords

    

        
        
def get_wrapped_backbone_score_fn(model, features, sde, means, atom_mask):

    sequence, positions = model.hypernetwork.process_tensors(features["fragment_seq"])

    
    w_x, b_x, w_t, w_b_t = model.hypernetwork(sequence, positions, False)
    
    batch_size = features["fragment_seq"].shape[0]
    
    w_x_reshaped = []
    b_x_reshaped = []
    w_t_reshaped = []
    w_b_t_reshaped = []

    prev_idx_w_x = 0
    prev_idx_b_x = 0
    prev_idx_w_t = 0
    prev_idx_w_b_t = 0

    for layer in model.mainnet_config.dims:
        
        w_x_reshaped.append(w_x[prev_idx_w_x].reshape(batch_size, layer[0], layer[1]))
        prev_idx_w_x += 1

        b_x_reshaped.append(b_x[prev_idx_b_x].reshape(batch_size, layer[1]))
        prev_idx_b_x += 1

        w_t_reshaped.append(w_t[prev_idx_w_t].reshape(batch_size, 1, layer[1]))
        prev_idx_w_t += 1

        w_b_t_reshaped.append(w_b_t[prev_idx_w_b_t].reshape(batch_size, 1, layer[1]))
        prev_idx_w_b_t += 1
            
    class ScoreFunction(nn.Module):

        def __init__(self, model):
    
            super(ScoreFunction, self).__init__()
            
            
            self.hypernetwork = model.hypernetwork
            
            self.backbone_dynamics = model.backbone_dynamics
            
            self.epsilon = None
            
        def drift_fn(self, score_fn, x, t):
          """Get the drift function of the reverse-time SDE."""
          rsde = sde.reverse(score_fn, probability_flow=True)
          return rsde.sde(x, t, means)[0]
         
        def div_fn(self, fn, x, t, eps, mask):
            
            fn_output = self.drift_fn(fn, x,t)*mask[:, None, :, :]
          
            fn_output_original = fn_output.clone()
            
            grad_fn_eps = torch.autograd.grad(fn_output, x, eps, create_graph=True)[0]
            
            jacobian_trace = torch.sum(grad_fn_eps * eps * mask[:, None, :, :], dim=tuple(range(2, len(x.shape))))
            
            return jacobian_trace, fn_output_original
        
        def score(self, perturbed_data, t):
            
            for feature in features:

                if "names" not in feature:
                    features[feature] = features[feature].detach()

            _, std = sde.marginal_prob(torch.zeros(perturbed_data.shape).to(
                features["fragment_seq"].device), t)
            
                    
            prediction = self.backbone_dynamics(t, perturbed_data, 
                                    features["fragment_seq"], 
                                    features["atoms_backbone"], 
                                    features["amino_acid_pos_backbone"], 
                                    features["bond_matrix_backbone"],
                                    features["edge_mask_backbone"], 
                                    features["atom_mask_backbone"],
                                    features["restore_indices"],
                                    w_x_reshaped, b_x_reshaped,
                                    w_t_reshaped, w_b_t_reshaped)
            
            prediction = -1*prediction/std[:, None, None, None].to(prediction.device)
            
            return prediction
            
        def forward(self, t, sample):
            
            with torch.set_grad_enabled(True):      
                x, delta_logp = sample
                x = torch.autograd.Variable(x, requires_grad=True)
                if self.epsilon is None:
                
                    self.epsilon = torch.randint_like(x, low=0, high=2).float() * 2 - 1.
            
                vec_t = torch.ones(x.shape[0], device=x.device) * t
            
                logp_grad, drift = self.div_fn(self.score, x, vec_t, self.epsilon, atom_mask)
            
            
                return (drift, logp_grad)
        
    return ScoreFunction(model)
            
            
                    
def get_wrapped_hydrogen_score_fn(model, features, sde, means, atom_mask):

            
    class ScoreFunction(nn.Module):

        def __init__(self, model):
    
            super(ScoreFunction, self).__init__()
            
            self.epsilon = None
            
            self.model = model

        def drift_fn(self, score_fn, x, t):
          """Get the drift function of the reverse-time SDE."""
          rsde = sde.reverse(score_fn, probability_flow=True)
          return rsde.sde(x, t, means)[0]
      
        def div_fn(self, fn, x, t, eps, mask):
            
            fn_output = self.drift_fn(fn, x,t)*mask[:, None, :, :]
          
            fn_output_original = fn_output.clone()
            
            grad_fn_eps = torch.autograd.grad(fn_output, x, eps, create_graph=True)[0]
            
            jacobian_trace = torch.sum(grad_fn_eps * eps * mask[:, None, :, :], dim=tuple(range(2, len(x.shape))))
        
            return jacobian_trace, fn_output_original
        
        def score(self, perturbed_data, t):
            
            _, std = sde.marginal_prob(torch.zeros(perturbed_data.shape).to(
                features["fragment_seq"].device), t)
               
            prediction = self.model(t*999, perturbed_data, features)
           
            prediction = -1*prediction/std[:, None, None, None].to(prediction.device)
       
     
            
            return prediction
            
        def forward(self, t, sample):
            x, delta_logp = sample
            with torch.set_grad_enabled(True):

                x = torch.autograd.Variable(x, requires_grad=True)

                if self.epsilon is None:
                
                    self.epsilon = torch.randint_like(x, low=0, high=2).float() * 2 - 1.
            
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                logp_grad, drift = self.div_fn(self.score, x, vec_t, self.epsilon, atom_mask)
            
            
                return (drift, logp_grad)
        
        
    return ScoreFunction(model)           
            

def get_wrapped_rotamer_score_fn(model, features, sde, means, atom_mask, side_chain_only):

            
    class ScoreFunction(nn.Module):

        def __init__(self, model):
    
            super(ScoreFunction, self).__init__()
            
            self.epsilon = None
            
            self.model = model
            

            self.backbone_coords = features["coordinates_rotamer"]

        def div_fn(self, fn, x, t, eps, mask):
            
            fn_output = self.drift_fn(fn, x,t)*mask[:, None, :, :]
          
            fn_output_original = fn_output.clone()
            
            grad_fn_eps = torch.autograd.grad(fn_output, x, eps, create_graph=True)[0]
            
            jacobian_trace = torch.sum(grad_fn_eps * eps * mask[:, None, :, :], dim=tuple(range(2, len(x.shape))))
        
            return jacobian_trace, fn_output_original
        
        def drift_fn(self, score_fn, x, t):
          """Get the drift function of the reverse-time SDE."""
          rsde = sde.reverse(score_fn, probability_flow=True)
          return rsde.sde(x, t, means)[0]

        def score(self, perturbed_data, t):
            
                    
            atom_indices = torch.logical_not((features["atoms_rotamer"][..., None] ==\
                                                           torch.LongTensor(backbone_atom_indices).to(features["atom_mask_rotamer"].device)).any(-1))
            
            full_mask = torch.logical_not((features["atoms_rotamer"][..., None] ==\
                                                                              torch.LongTensor(backbone_atom_indices).to(features["atom_mask_rotamer"].device)).any(-1)).unsqueeze(-1).repeat(1,1,3)
            
            if side_chain_only:
                _, std = sde.marginal_prob(torch.zeros(perturbed_data.shape).to(features["fragment_seq"].device), 
                       t, torch.stack([features["atom_mask_rotamer"][i, atom_indices[i], :] for i in range(len(atom_indices))]) , 
                       torch.stack([features["amino_acid_pos_rotamer"][i, atom_indices[i]] for i in range(len(features["amino_acid_pos_rotamer"]))]).long() - 1)
                     
                x = self.backbone_coords.masked_scatter(full_mask.unsqueeze(
                                   1).repeat(1, perturbed_data.shape[1], 1, 1), perturbed_data)
            else:
                
                _, std = sde.marginal_prob(torch.zeros(perturbed_data.shape).to(features["fragment_seq"].device),
                        t, features["atom_mask_rotamer"], features["amino_acid_pos_rotamer"].long() - 1)
                full_mask = full_mask.float()
                
                x = self.backbone_coords*(1-full_mask) + perturbed_data*full_mask
            
            prediction = self.model(t*999, x, features)
            if side_chain_only:
               
                prediction = torch.stack([prediction[i, :, atom_indices[i], :] for i in range(len(atom_indices))])
           
            return -1*prediction/std[:, None, None, None].to(prediction.device)
            
        def forward(self, t, sample):
            with torch.set_grad_enabled(True):
                x, delta_logp = sample
               
                x = torch.autograd.Variable(x, requires_grad=True)
                
                if self.epsilon is None:
                
                    self.epsilon = torch.randint_like(x, low=0, high=2).float() * 2 - 1.
            
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                logp_grad, drift = self.div_fn(self.score, x, vec_t, self.epsilon, atom_mask)
            
            
                return (drift, logp_grad)
        
        
    return ScoreFunction(model)       
