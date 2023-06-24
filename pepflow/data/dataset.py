from torch.utils.data import Dataset
from pepflow.utils.constants import backbone_atom_indices, residue_order, atom_order
from pepflow.utils.dataset_utils import convert_to_tensor, pad_features, get_features_from_seq,\
    permutations, MIN_LENGTH, MAX_LENGTH
from pepflow.data.data_pipeline import DataPipelinePED, DataPipelineMD
from scipy.spatial.distance import cdist
import torch
import numpy as np
import os

def _get_hydrogen_features(features):
    
    
    '''
    Returns processed features for protonation
    
    '''
    
    fragment_seq, coordinates, amino_acid_pos, atoms, atom_names,\
        bond_matrix = features
        
    mean_coords = np.sum(coordinates, axis=0)[None, :]/len(coordinates)
    coordinates = coordinates - \
        mean_coords
    
        
    # identify hydrogen atoms and heavy atoms
    
    is_h = atoms == atom_order["H"]
    
    coordinates_h = coordinates[is_h, :]
    
    coordinates_heavy = coordinates[np.logical_not(is_h), :]
    
    atoms_h = atoms[is_h]
    
    atoms_heavy = atoms[np.logical_not(is_h)]

    amino_acid_pos_heavy = amino_acid_pos[np.logical_not(is_h)] 
                                            
    amino_acid_pos_h = amino_acid_pos[is_h]
    
    # get the atom names for heavy and hydrogen atoms
    
    atom_names_heavy = [i for index, i in enumerate(atom_names) if np.logical_not(is_h)[index]]
    
    atom_names_h = [i for index, i in enumerate(atom_names) if is_h[index]]
    
    
    # get heavy atom distances
    
    neighbors = np.argwhere(cdist(coordinates_heavy, 
                                   coordinates_heavy) <= 4)
    

    #generate edge mask between heavy atoms
    
    edge_mask = np.zeros((len(coordinates_heavy), 
                        len(coordinates_heavy)))
    
    edge_mask[neighbors[:, 0], neighbors[:, 1]] = 1
    
    
    # generate edge mask between hydrogens and heavy atoms
    
    bound_atom = np.argwhere(bond_matrix[is_h, :][:, np.logical_not(is_h)] == 1)[:, 1]
    

    edge_mask_h_heavy = edge_mask[bound_atom, :]
    
    
    # generate edges and edge mask between hydrogens
    
    edges_h_h = np.argwhere((bound_atom[:, None] == bound_atom))
    
    edge_mask_h_h = np.ones(len(edges_h_h))
    
    
    # generate atom masks
     
    atom_mask = np.ones((len(bound_atom), 3))
    
    atom_mask_heavy = np.ones((len(atoms_heavy), 3))
    
    # get bond matrices
    
    bond_matrix_h_heavy = bond_matrix[is_h, :][:, np.logical_not(is_h)]
    
    
    # get permutations
    
    all_permutations = np.zeros((6, len(bound_atom)))
    
    unique_bound_atom = np.unique(bound_atom)
    
    unique_num_bound = []
    
    for atom in unique_bound_atom:

        num_bound = np.sum(bound_atom == atom)
        
        unique_num_bound.append(num_bound)


        for index, permutation in enumerate(permutations[num_bound]):
            all_permutations[index][bound_atom == atom]  = np.argwhere(bound_atom == atom)[permutation, 0]
            
    
    


    coordinates_h = coordinates_h - coordinates_heavy[bound_atom, :]
    
    permuted_coords  = np.stack([coordinates_h[i.astype(int)] for i in all_permutations], axis=0)
    
    computed_features = {}
    
    computed_features["fragment_seq"] = fragment_seq
    computed_features["coordinates_h"] = coordinates_h
    computed_features["coordinates_heavy_h"] = coordinates_heavy
    computed_features["atoms_h"] = atoms_h 
    computed_features["atoms_heavy"] = atoms_heavy
    computed_features["amino_acid_pos_h"] = amino_acid_pos_h
    computed_features["amino_acid_pos_heavy"] = amino_acid_pos_heavy
    computed_features["bond_matrix_h_heavy"] = bond_matrix_h_heavy
    computed_features["edges_h_h"] = edges_h_h
    computed_features["edge_mask_heavy_neighbors"] = edge_mask
    computed_features["edge_mask_h_heavy"] = edge_mask_h_heavy
    computed_features["edge_mask_h_h"] = edge_mask_h_h
    computed_features["atom_mask_h"] = atom_mask
    computed_features["atom_mask_heavy"] = atom_mask_heavy
    computed_features["bound_atom"] = bound_atom
    computed_features["atom_names_heavy"] = atom_names_heavy
    computed_features["atom_names_h"] = atom_names_h
    computed_features["all_permutations"] = all_permutations
    computed_features["permuted_coords"] = permuted_coords

    
  
        
    return computed_features
    
def _get_rotamer_features(features):
    
    fragment_seq, coordinates, amino_acid_pos, atoms, atom_names,\
            bond_matrix = features
            

        
    # compute centroid features
    
    is_backbone = np.isin(atoms, backbone_atom_indices)
    
    backbone_indices = np.flatnonzero(is_backbone)
   
    centroid_counter = 0
    
    coordinates_centroid = []

    centroid_indices = []
    
    for index, res in enumerate(fragment_seq):
        
        if res == residue_order["G"]:
            centroid_indices.append(-1)
            continue
        
        coordinates_side_chain = coordinates[np.logical_and(np.logical_not(is_backbone),
                                                            amino_acid_pos == index+1), :]
        
        centroid_coord = np.mean(coordinates_side_chain, axis=0)
        
        coordinates_centroid.append(centroid_coord)
        
      

        centroid_indices.append(centroid_counter)
        
        centroid_counter += 1
    
    centroid_indices = np.array(centroid_indices)
    
    coordinates_centroid = np.array(coordinates_centroid)

    # center coordinates
    
    mean_coords = np.sum(coordinates, axis=0)[None, :]/len(coordinates)
    
    coordinates = coordinates - \
        mean_coords
    
    if len(coordinates_centroid) > 0:
        
        coordinates_centroid = coordinates_centroid - mean_coords
   
    # generate masks
    centroid_distances = cdist(coordinates_centroid, coordinates_centroid)

    edge_cutoff_mask = centroid_distances[centroid_indices[amino_acid_pos - 1], :]\
        [:, centroid_indices[amino_acid_pos - 1]] < 8.0
    
    edge_mask = np.ones(edge_cutoff_mask.shape)
    edge_mask[backbone_indices, :] = 0
    edge_mask = edge_mask * edge_cutoff_mask

    atom_mask = np.ones((len(atoms), 3))
    atom_mask[backbone_indices, :] = 0

    # generate feature dict
    
    computed_features = {}
    
    computed_features['fragment_seq'] = fragment_seq
    computed_features['coordinates_rotamer'] = coordinates
    computed_features['bond_matrix_rotamer'] = bond_matrix
    computed_features['atoms_rotamer'] = atoms
    computed_features['amino_acid_pos_rotamer'] = amino_acid_pos
    computed_features['edge_mask_rotamer'] = edge_mask
    computed_features['atom_mask_rotamer'] = atom_mask
    computed_features['atom_names_rotamer'] = atom_names
    computed_features['centroid_coordinates'] = coordinates_centroid
    computed_features['centroid_indices'] = centroid_indices
    
    return computed_features

def _get_backbone_features(features):
    
    fragment_seq, coordinates, amino_acid_pos, atoms, atom_names,\
            bond_matrix = features
    
    is_backbone = np.isin(atoms, backbone_atom_indices)
 
    
    # get backbone atoms
   
    backbone_atoms = atoms[is_backbone]

    
    # compute centroid features
    
    coordinates_centroid = []
    
    amino_acid_pos_centroid = []
    
    atoms_centroid = []
    
    
    centroid_counter = 0
    
    centroid_indices = []
    
    for index, res in enumerate(fragment_seq):
                
        if res == residue_order["G"]:
            centroid_indices.append(-1)
            continue
        
        centroid_indices.append(centroid_counter + len(backbone_atoms))
        
        coordinates_side_chain = coordinates[np.logical_and(np.logical_not(is_backbone),
                                                            amino_acid_pos == index+1), :]
        
      
        
        centroid_coord = np.mean(coordinates_side_chain, axis=0)
        
        coordinates_centroid.append(centroid_coord)
        
        amino_acid_pos_centroid.append(index+1)
        
        atoms_centroid.append(4)
        
        centroid_counter += 1
        
    if centroid_counter > 0:
        coordinates = np.concatenate((coordinates[is_backbone,:], np.array(coordinates_centroid)),
                                     axis=0)
    
        amino_acid_pos = np.concatenate((amino_acid_pos[is_backbone], 
                                          np.array(amino_acid_pos_centroid)))
    
        atoms = np.concatenate((backbone_atoms, np.array(atoms_centroid)))
        
        
        bond_matrix = bond_matrix[is_backbone, :][:, is_backbone]
        bond_matrix_extended = np.zeros((len(bond_matrix) + centroid_counter,
                                     len(bond_matrix) + centroid_counter))
    
        bond_matrix_extended[:len(bond_matrix), :len(bond_matrix)] = bond_matrix
    

        bond_matrix = bond_matrix_extended
    
    atom_names = [atom_names[i] for i  in np.argwhere(is_backbone).flatten()]
        
    for i in range(centroid_counter):
        atom_names.append("CEN")

    length = len(atoms)

    # get masks
    edge_mask = np.ones((length, length))

    atom_mask = np.ones((length, 3))


    # get positional-reference atom for each atom in backbone

    reference_indices = np.array(list(range(len(atoms))))
    
    same_residue = amino_acid_pos[:, None] == amino_acid_pos[None, :]
    previous_residue = (amino_acid_pos[:, None] - 1) == amino_acid_pos[None, :]
    
    n_c = np.argwhere(np.logical_and(np.logical_and(previous_residue, atoms[:, None] == atom_order["N"]),
                         atoms[None, :] == atom_order["C"]))
    
    
    reference_indices[n_c[:, 0]] = n_c[:, 1]
    
    ca_n = np.argwhere(np.logical_and(np.logical_and(same_residue, atoms[:, None] == atom_order["CA"]),
                         atoms[None, :] == atom_order["N"]))
    
    
    reference_indices[ca_n[:, 0]] = ca_n[:, 1]
    
    
    c_ca = np.argwhere(np.logical_and(np.logical_and(same_residue, atoms[:, None] == atom_order["C"]),
                         atoms[None, :] == atom_order["CA"]))
    
    
    reference_indices[c_ca[:, 0]] = c_ca[:, 1]
    

    o_c = np.argwhere(np.logical_and(np.logical_and(same_residue, atoms[:, None] == atom_order["O"]),
                         atoms[None, :] == atom_order["C"]))
    
    
    reference_indices[o_c[:, 0]] = o_c[:, 1]
    
    cen_ca = np.argwhere(np.logical_and(np.logical_and(same_residue, atoms[:, None] == 4),
                         atoms[None, :] == atom_order["CA"]))
    
    
    reference_indices[cen_ca[:, 0]] = cen_ca[:, 1]
                                     

    
    restore_indices = np.eye(len(reference_indices))
    restore_indices[np.array(list(range(len(reference_indices)))), reference_indices] = 1
    restore_indices = np.linalg.matrix_power(restore_indices, len(atoms)) > 0
    
   
    # get relative coordinates
    
    coordinates_relative = coordinates - coordinates[reference_indices, :]
    
    computed_features = {}
    
    computed_features['fragment_seq'] = fragment_seq
    computed_features['coordinates_backbone'] = coordinates_relative
    computed_features['bond_matrix_backbone'] = bond_matrix
    computed_features['atoms_backbone'] = atoms
    computed_features['amino_acid_pos_backbone'] = amino_acid_pos
    computed_features['edge_mask_backbone'] = edge_mask
    computed_features['atom_mask_backbone'] = atom_mask
    computed_features['atom_names_backbone'] = atom_names
    computed_features['reference_indices'] = reference_indices
    computed_features['restore_indices'] = restore_indices
    
    return computed_features



                    
class FragmentDatasetHydrogen(Dataset):

    def __init__(self, mode, data_dir="../datasets/",
                 data_path="./", pad=True, length=None):
        
        self.mode = mode
       
        self.data_dir = data_dir
        
        if self.mode == "train":
            self.data_list = []
            
            for fragment_length in range(MIN_LENGTH, MAX_LENGTH+1):
                
                if length != None:
                    if fragment_length != length:
                        continue
                    
                new_list = np.load(
                    data_dir + "/training_fragments_" +\
                            str(fragment_length) + ".npy",
                            allow_pickle=True).item()
                

                for element in new_list:

                    self.data_list.append((fragment_length, new_list[element][1]))
            
            
        elif self.mode == "val":
            self.data_list = []
            
            for fragment_length in range(MIN_LENGTH, MAX_LENGTH+1):
                
                if length != None:
                    if fragment_length != length:
                        continue
                    
                new_list = np.load(
                        data_dir + "/val_fragments_" +\
                                str(fragment_length) + ".npy",
                                allow_pickle=True).item()

                

                for element in new_list:
                    
                    self.data_list.append((fragment_length, new_list[element][1]))           
                    
        elif self.mode == "val_subset":
            self.data_list = []
            
            for fragment_length in range(MIN_LENGTH, MAX_LENGTH+1):
                
                if length != None:
                    if fragment_length != length:
                        continue
                    
                new_list = np.load(
                        data_dir + "/val_fragments_" +\
                                str(fragment_length) + "_subset.npy",
                                allow_pickle=True).item()

                

                for element in new_list:
                    
                    self.data_list.append((fragment_length, new_list[element][1]))           
                    
        
        self.num_data = len(self.data_list)
        
        self.pad = pad
        
        
        self.data_path = data_path
        
    def __len__(self):
        return self.num_data
            
    def __getitem__(self, index):

        item = self.data_list[index]
        
        fragment_location = item[1]
    
        fragment_length = item[0]

        if not os.path.exists(os.path.join(self.data_path, "../n_terminal_feats/", fragment_location.split("_")[0]\
                              + "_" + fragment_location.split("_")[1] + ".npz")):
            return None
        
        
        n_terminal_features = np.load(os.path.join(self.data_path, "../n_terminal_feats/", fragment_location.split("_")[0]\
                              + "_" + fragment_location.split("_")[1] + ".npz"), allow_pickle=True)
            
        c_terminal_features = np.load(os.path.join(self.data_path, "../c_terminal_feats/", fragment_location.split("_")[0]\
                              + "_" + fragment_location.split("_")[1] + ".npz"), allow_pickle=True)
            
        non_terminal_features = np.load(os.path.join(self.data_path, "../non_terminal_feats/", fragment_location.split("_")[0]\
                              + "_" + fragment_location.split("_")[1] + ".npz"), allow_pickle=True)
            

        fragment_seq = []
        coordinates = []
        amino_acid_pos = []
        atoms = []
        atom_names = []
        
        total_length = 0
        previous_C = 0

        for res in range(fragment_length):
            
            if res == 0:
                
                res_index = fragment_location.split("_")[2]
                
                if res_index not in n_terminal_features:
                    return None
                
                
                res_features = n_terminal_features[res_index].item()
                
                fragment_seq.append(res_features["fragment_seq"])
                
                coordinates.append(res_features["coordinates"])
                
                amino_acid_pos.append(np.ones_like(res_features["atoms"]))
                
                atoms.append(res_features["atoms"])
                
                total_length = len(atoms[-1])
                previous_C = np.argmax(atoms[-1] == atom_order["C"])

                atom_names.append(res_features["atom_names"])

                if "bond_matrix" in res_features:
                    bond_matrix = res_features["bond_matrix"]
                else:
                    bond_matrix = res_features["bond matrix"]
            
            elif res == fragment_length - 1:
                
                res_index = str(int(fragment_location.split("_")[2]) + res)
                
                if res_index not in c_terminal_features:
                    return None
                
                res_features = c_terminal_features[res_index].item()

                fragment_seq.append(res_features["fragment_seq"])
                
                coordinates.append(res_features["coordinates"])
                
                amino_acid_pos.append(np.ones_like(res_features["atoms"])*(res+1))
                
                atoms.append(res_features["atoms"])
                
                atom_names.append(res_features["atom_names"])

                if "bond_matrix" in res_features:
                    bond_matrix_curr = res_features["bond_matrix"]
                else:
                    bond_matrix_curr = res_features["bond matrix"]
                
                bond_matrix_full = np.zeros((bond_matrix.shape[0] + bond_matrix_curr.shape[0], 
                                             bond_matrix.shape[1] + bond_matrix_curr.shape[0]))
                
                bond_matrix_full[:bond_matrix.shape[0], :][:, :bond_matrix.shape[0]] = bond_matrix
                
                bond_matrix_full[bond_matrix.shape[0]:, :][:, bond_matrix.shape[0]:] = bond_matrix_curr
                
                
                
                current_N = total_length + np.argmax(atoms[-1] == atom_order["N"])
                
                bond_matrix_full[previous_C, current_N] = 1
                
                bond_matrix_full[current_N, previous_C] = 1
                
                bond_matrix = bond_matrix_full

                
                
            else:
                
                res_index = str(int(fragment_location.split("_")[2]) + res)
                
                if res_index not in non_terminal_features:
                    return None
                
                res_features = non_terminal_features[res_index].item()

                fragment_seq.append(res_features["fragment_seq"])
                
                coordinates.append(res_features["coordinates"])
                
                amino_acid_pos.append(np.ones_like(res_features["atoms"])*(res+1))
                
                atoms.append(res_features["atoms"])
                
                atom_names.append(res_features["atom_names"])

                if "bond_matrix" in res_features:
                    bond_matrix_curr = res_features["bond_matrix"]
                else:
                    bond_matrix_curr = res_features["bond matrix"]
                
                bond_matrix_full = np.zeros((bond_matrix.shape[0] + bond_matrix_curr.shape[0], 
                                             bond_matrix.shape[1] + bond_matrix_curr.shape[0]))
                
                bond_matrix_full[:bond_matrix.shape[0], :][:, :bond_matrix.shape[0]] = bond_matrix
                
                bond_matrix_full[bond_matrix.shape[0]:, :][:, bond_matrix.shape[0]:] = bond_matrix_curr
                
                
                current_N = total_length + np.argmax(atoms[-1] == atom_order["N"])

                bond_matrix_full[previous_C, current_N] = 1
                
                bond_matrix_full[current_N, previous_C] = 1
                
                bond_matrix = bond_matrix_full
                
                previous_C = total_length + np.argmax(atoms[-1] == atom_order["C"])

                total_length = total_length + len(atoms[-1])

        fragment_seq = np.array(fragment_seq)
        coordinates = np.concatenate(coordinates, axis=0)
        amino_acid_pos = np.concatenate(amino_acid_pos, axis=0)
        atoms = np.concatenate(atoms, axis=0)
        atom_names_full = [inner for outer in atom_names for inner in outer]
        
        features = fragment_seq, coordinates, amino_acid_pos, atoms, atom_names_full,\
            bond_matrix
            
        if self.pad:
            return pad_features(_get_hydrogen_features(features))
        else:
            return convert_to_tensor(_get_hydrogen_features(features))

class FragmentDatasetRotamer(Dataset):

    def __init__(self, mode, data_dir="../datasets/",
                 data_path="./", pad=True, length=None):
        
        self.mode = mode
       
        self.data_dir = data_dir
        
        if self.mode == "train":
            self.data_list = []
            
            for fragment_length in range(MIN_LENGTH, MAX_LENGTH+1):
                
                if length != None:
                    if fragment_length != length:
                        continue
                
                new_list = np.load(
                    data_dir + "/training_fragments_" +\
                            str(fragment_length) + ".npy",
                            allow_pickle=True).item()
                

                for element in new_list:

                    #filter out all-glycine fragments, as they lack sidechains

                    if len(new_list[element][0]) != new_list[element][0].count('G'):
                        self.data_list.append((fragment_length, new_list[element][1]))
            
            
        elif self.mode == "val":
            self.data_list = []
            
            for fragment_length in range(MIN_LENGTH, MAX_LENGTH+1):
                
                if length != None:
                    if fragment_length != length:
                        continue
                    
                new_list = np.load(
                        data_dir + "/val_fragments_" +\
                                str(fragment_length) + ".npy",
                                allow_pickle=True).item()

                

                for element in new_list:
                    if len(new_list[element][0]) != new_list[element][0].count('G'):
                        self.data_list.append((fragment_length, new_list[element][1]))           
                    
        elif self.mode == "val_subset":
            self.data_list = []
            
            for fragment_length in range(MIN_LENGTH, MAX_LENGTH+1):
                
                if length != None:
                    if fragment_length != length:
                        continue
                    
                new_list = np.load(
                        data_dir + "/val_fragments_" +\
                                str(fragment_length) + "_subset.npy",
                                allow_pickle=True).item()

                

                for element in new_list:
                    
                    self.data_list.append((fragment_length, new_list[element][1]))   
                    
        self.num_data = len(self.data_list)
        
        self.pad = pad
        
        self.data_path = data_path
        
    def __len__(self):
        return self.num_data
            
    def __getitem__(self, index):

        item = self.data_list[index]
        
        fragment_location = item[1]
    
        fragment_length = item[0]

        if not os.path.exists(os.path.join(self.data_path, "../n_terminal_feats/", fragment_location.split("_")[0]\
                              + "_" + fragment_location.split("_")[1] + ".npz")):
            return None
        
        
        n_terminal_features = np.load(os.path.join(self.data_path, "../n_terminal_feats/", fragment_location.split("_")[0]\
                              + "_" + fragment_location.split("_")[1] + ".npz"), allow_pickle=True)
            
        c_terminal_features = np.load(os.path.join(self.data_path, "../c_terminal_feats/", fragment_location.split("_")[0]\
                              + "_" + fragment_location.split("_")[1] + ".npz"), allow_pickle=True)
            
        non_terminal_features = np.load(os.path.join(self.data_path, "../non_terminal_feats/", fragment_location.split("_")[0]\
                              + "_" + fragment_location.split("_")[1] + ".npz"), allow_pickle=True)
            

        fragment_seq = []
        coordinates = []
        amino_acid_pos = []
        atoms = []
        atom_names = []
        
        total_length = 0
        previous_C = 0

        for res in range(fragment_length):
            
            if res == 0:
                
                res_index = fragment_location.split("_")[2]
                
                if res_index not in n_terminal_features:
                    return None
                
                
                res_features = n_terminal_features[res_index].item()
                
                atom_indices = res_features["atoms"] != atom_order["H"]
                
                fragment_seq.append(res_features["fragment_seq"])
                
                coordinates.append(res_features["coordinates"][atom_indices, :])
                
                amino_acid_pos.append(np.ones_like(res_features["atoms"])[atom_indices])
                
                atoms.append(res_features["atoms"][atom_indices])
                
                total_length = len(atoms[-1])
                previous_C = np.argmax(atoms[-1] == atom_order["C"])

                atom_names.append([res_features["atom_names"][index] for index, i in enumerate(atom_indices) if i])

                if "bond_matrix" in res_features:
                    bond_matrix = res_features["bond_matrix"][atom_indices, :][:, atom_indices]
                else:
                    bond_matrix = res_features["bond matrix"][atom_indices, :][:, atom_indices]
            
            elif res == fragment_length - 1:
                
                res_index = str(int(fragment_location.split("_")[2]) + res)
                
                if res_index not in c_terminal_features:
                    return None
                
                res_features = c_terminal_features[res_index].item()
                
                atom_indices = res_features["atoms"] != atom_order["H"]

                fragment_seq.append(res_features["fragment_seq"])
                
                coordinates.append(res_features["coordinates"][atom_indices, :])
                
                amino_acid_pos.append(np.ones_like(res_features["atoms"])[atom_indices]*(res+1))
                
                atoms.append(res_features["atoms"][atom_indices])
                
                atom_names.append([res_features["atom_names"][index] for index, i in enumerate(atom_indices) if i])

                if "bond_matrix" in res_features:
                    bond_matrix_curr = res_features["bond_matrix"][atom_indices, :][:, atom_indices]
                else:
                    bond_matrix_curr = res_features["bond matrix"][atom_indices, :][:, atom_indices]
                
                bond_matrix_full = np.zeros((bond_matrix.shape[0] + bond_matrix_curr.shape[0], 
                                             bond_matrix.shape[1] + bond_matrix_curr.shape[0]))
                
                bond_matrix_full[:bond_matrix.shape[0], :][:, :bond_matrix.shape[0]] = bond_matrix
                
                bond_matrix_full[bond_matrix.shape[0]:, :][:, bond_matrix.shape[0]:] = bond_matrix_curr
                
                
                
                current_N = total_length + np.argmax(atoms[-1] == atom_order["N"])
                
                bond_matrix_full[previous_C, current_N] = 1
                
                bond_matrix_full[current_N, previous_C] = 1
                
                bond_matrix = bond_matrix_full

                
                
            else:
                
                res_index = str(int(fragment_location.split("_")[2]) + res)
                
                if res_index not in non_terminal_features:
                    return None
                
                res_features = non_terminal_features[res_index].item()
                
                atom_indices = res_features["atoms"] != atom_order["H"]

                fragment_seq.append(res_features["fragment_seq"])
                
                coordinates.append(res_features["coordinates"][atom_indices, :])
                
                amino_acid_pos.append(np.ones_like(res_features["atoms"])[atom_indices]*(res+1))
                
                atoms.append(res_features["atoms"][atom_indices])
                
                atom_names.append([res_features["atom_names"][index] for index, i in enumerate(atom_indices) if i])

                if "bond_matrix" in res_features:
                    bond_matrix_curr = res_features["bond_matrix"][atom_indices, :][:, atom_indices]
                else:
                    bond_matrix_curr = res_features["bond matrix"][atom_indices, :][:, atom_indices]
                
                bond_matrix_full = np.zeros((bond_matrix.shape[0] + bond_matrix_curr.shape[0], 
                                             bond_matrix.shape[1] + bond_matrix_curr.shape[0]))
                
                bond_matrix_full[:bond_matrix.shape[0], :][:, :bond_matrix.shape[0]] = bond_matrix
                
                bond_matrix_full[bond_matrix.shape[0]:, :][:, bond_matrix.shape[0]:] = bond_matrix_curr
                
                
                current_N = total_length + np.argmax(atoms[-1] == atom_order["N"])

                bond_matrix_full[previous_C, current_N] = 1
                
                bond_matrix_full[current_N, previous_C] = 1
                
                bond_matrix = bond_matrix_full
                
                previous_C = total_length + np.argmax(atoms[-1] == atom_order["C"])

                total_length = total_length + len(atoms[-1])

        fragment_seq = np.array(fragment_seq)
        coordinates = np.concatenate(coordinates, axis=0)
        amino_acid_pos = np.concatenate(amino_acid_pos, axis=0)
        atoms = np.concatenate(atoms, axis=0)
        atom_names_full = [inner for outer in atom_names for inner in outer]
        
        features = fragment_seq, coordinates, amino_acid_pos, atoms, atom_names_full,\
            bond_matrix
            
        if self.pad:
            return pad_features(_get_rotamer_features(features))
        else:
            return convert_to_tensor(_get_rotamer_features(features))
                
class CustomWeightedRandomSampler(torch.utils.data.sampler.WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        
        return iter(rand_tensor.tolist())

def get_weighted_sampler(mode, data_dir="../datasets/"):
    
    
    if mode == "val":
        
        weights = np.load(data_dir + "weights_ss_val.npy")
        
    elif mode == "train":
        
        weights = np.load(data_dir + "weights_ss_train.npy")
        
    return CustomWeightedRandomSampler(torch.FloatTensor(weights), len(weights),
		replacement=True)
        
class FragmentDatasetBackbone(Dataset):

    def __init__(self, mode, data_dir="../datasets/",
                 data_path="./", pad=True, length=None, full=False):
        
        self.mode = mode
       
        self.data_dir = data_dir
        
        if self.mode == "train":
            
            if not full:
                self.data_list = []
                
                for fragment_length in range(MIN_LENGTH, MAX_LENGTH+1):
    
                    if length != None:
                        if fragment_length != length:
                            continue
                        
                    new_list = np.load(
                        data_dir + "/training_fragments_" +\
                                str(fragment_length) + ".npy",
                                allow_pickle=True).item()
                    
    
                    for element in new_list:
    
                        self.data_list.append((fragment_length, new_list[element][1]))
            else:
                
                self.data_list = np.load(data_dir + "/datalist_train.npy")
                
            
            
            
        elif self.mode == "val":
            
            if not full:
                
                self.data_list = []
                
                for fragment_length in range(MIN_LENGTH, MAX_LENGTH+1):
                    
                    if length != None:
                        if fragment_length != length:
                            continue
                        
                    new_list = np.load(
                            data_dir + "/val_fragments_" +\
                                    str(fragment_length) + ".npy",
                                    allow_pickle=True).item()
    
                    
    
                    for element in new_list:
                        
                        self.data_list.append((fragment_length, new_list[element][1]))       
            else:
                
                self.data_list = np.load(data_dir + "/datalist_val.npy")
                    
        elif self.mode == "val_subset":
            
         
                
            self.data_list = []
            
            for fragment_length in range(MIN_LENGTH, MAX_LENGTH+1):
                
                if length != None:
                    if fragment_length != length:
                        continue
                    
                new_list = np.load(
                        data_dir + "/val_fragments_" +\
                                str(fragment_length) + "_subset.npy",
                                allow_pickle=True).item()

                

                for element in new_list:
                    
                    self.data_list.append((fragment_length, new_list[element][1]))       

                    
                
        self.num_data = len(self.data_list)
        
        self.pad = pad
        
        self.data_path = data_path
        
    def __len__(self):
        return self.num_data
            
    def __getitem__(self, index):

        item = self.data_list[index]
        
        fragment_location = item[1]
    
        fragment_length = int(item[0])

        if not os.path.exists(os.path.join(self.data_path, "../n_terminal_feats/", fragment_location.split("_")[0]\
                              + "_" + fragment_location.split("_")[1] + ".npz")):
            return None
        
        
        n_terminal_features = np.load(os.path.join(self.data_path, "../n_terminal_feats/", fragment_location.split("_")[0]\
                              + "_" + fragment_location.split("_")[1] + ".npz"), allow_pickle=True)
            
        c_terminal_features = np.load(os.path.join(self.data_path, "../c_terminal_feats/", fragment_location.split("_")[0]\
                              + "_" + fragment_location.split("_")[1] + ".npz"), allow_pickle=True)
            
        non_terminal_features = np.load(os.path.join(self.data_path, "../non_terminal_feats/", fragment_location.split("_")[0]\
                              + "_" + fragment_location.split("_")[1] + ".npz"), allow_pickle=True)
            

        fragment_seq = []
        coordinates = []
        amino_acid_pos = []
        atoms = []
        atom_names = []
        
        total_length = 0
        previous_C = 0

        for res in range(fragment_length):
            
            if res == 0:
                
                res_index = fragment_location.split("_")[2]
                
                if res_index not in n_terminal_features:
                    return None
                
                
                res_features = n_terminal_features[res_index].item()
                
                atom_indices = res_features["atoms"] != atom_order["H"]
                
                fragment_seq.append(res_features["fragment_seq"])
                
                coordinates.append(res_features["coordinates"][atom_indices, :])
                
                amino_acid_pos.append(np.ones_like(res_features["atoms"])[atom_indices])
                
                atoms.append(res_features["atoms"][atom_indices])
                
                total_length = len(atoms[-1])
                previous_C = np.argmax(atoms[-1] == atom_order["C"])

                atom_names.append([res_features["atom_names"][index] for index, i in enumerate(atom_indices) if i])

                if "bond_matrix" in res_features:
                    bond_matrix = res_features["bond_matrix"][atom_indices, :][:, atom_indices]
                else:
                    bond_matrix = res_features["bond matrix"][atom_indices, :][:, atom_indices]
            
            elif res == fragment_length - 1:
                
                res_index = str(int(fragment_location.split("_")[2]) + res)
                
                if res_index not in c_terminal_features:
                    return None
                
                res_features = c_terminal_features[res_index].item()
                
                atom_indices = res_features["atoms"] != atom_order["H"]

                fragment_seq.append(res_features["fragment_seq"])
                
                coordinates.append(res_features["coordinates"][atom_indices, :])
                
                amino_acid_pos.append(np.ones_like(res_features["atoms"])[atom_indices]*(res+1))
                
                atoms.append(res_features["atoms"][atom_indices])
                
                atom_names.append([res_features["atom_names"][index] for index, i in enumerate(atom_indices) if i])

                if "bond_matrix" in res_features:
                    bond_matrix_curr = res_features["bond_matrix"][atom_indices, :][:, atom_indices]
                else:
                    bond_matrix_curr = res_features["bond matrix"][atom_indices, :][:, atom_indices]
                
                bond_matrix_full = np.zeros((bond_matrix.shape[0] + bond_matrix_curr.shape[0], 
                                             bond_matrix.shape[1] + bond_matrix_curr.shape[0]))
                
                bond_matrix_full[:bond_matrix.shape[0], :][:, :bond_matrix.shape[0]] = bond_matrix
                
                bond_matrix_full[bond_matrix.shape[0]:, :][:, bond_matrix.shape[0]:] = bond_matrix_curr
                
                
                
                current_N = total_length + np.argmax(atoms[-1] == atom_order["N"])
                
                bond_matrix_full[previous_C, current_N] = 1
                
                bond_matrix_full[current_N, previous_C] = 1
                
                bond_matrix = bond_matrix_full

                
                
            else:
                
                res_index = str(int(fragment_location.split("_")[2]) + res)
                
                if res_index not in non_terminal_features:
                    return None
                
                res_features = non_terminal_features[res_index].item()
                
                atom_indices = res_features["atoms"] != atom_order["H"]

                fragment_seq.append(res_features["fragment_seq"])
                
                coordinates.append(res_features["coordinates"][atom_indices, :])
                
                amino_acid_pos.append(np.ones_like(res_features["atoms"])[atom_indices]*(res+1))
                
                atoms.append(res_features["atoms"][atom_indices])
                
                atom_names.append([res_features["atom_names"][index] for index, i in enumerate(atom_indices) if i])

                if "bond_matrix" in res_features:
                    bond_matrix_curr = res_features["bond_matrix"][atom_indices, :][:, atom_indices]
                else:
                    bond_matrix_curr = res_features["bond matrix"][atom_indices, :][:, atom_indices]
                
                bond_matrix_full = np.zeros((bond_matrix.shape[0] + bond_matrix_curr.shape[0], 
                                             bond_matrix.shape[1] + bond_matrix_curr.shape[0]))
                
                bond_matrix_full[:bond_matrix.shape[0], :][:, :bond_matrix.shape[0]] = bond_matrix
                
                bond_matrix_full[bond_matrix.shape[0]:, :][:, bond_matrix.shape[0]:] = bond_matrix_curr
                
                
                current_N = total_length + np.argmax(atoms[-1] == atom_order["N"])

                bond_matrix_full[previous_C, current_N] = 1
                
                bond_matrix_full[current_N, previous_C] = 1
                
                bond_matrix = bond_matrix_full
                
                previous_C = total_length + np.argmax(atoms[-1] == atom_order["C"])

                total_length = total_length + len(atoms[-1])

        fragment_seq = np.array(fragment_seq)
        coordinates = np.concatenate(coordinates, axis=0)
        amino_acid_pos = np.concatenate(amino_acid_pos, axis=0)
        atoms = np.concatenate(atoms, axis=0)
        atom_names_full = [inner for outer in atom_names for inner in outer]
        
        features = fragment_seq, coordinates, amino_acid_pos, atoms, atom_names_full,\
            bond_matrix
            
        if self.pad:
            return pad_features(_get_backbone_features(features))
        else:
            return convert_to_tensor(_get_backbone_features(features))



class PeptideDataset(Dataset):

    def __init__(self, mode="train", data_dir="../datasets",
                 data_path="./pep_features", pad=True, model="backbone"):
       
        self.pad = pad
        
        self.data_path = data_path
        
        self.model = model
        
        self.mode = mode
        
        
        if self.mode == "train":
            self.cluster_list = np.load(os.path.join(data_dir, "train_peptide_clusters.npy"))
        elif self.mode == "val":
            self.cluster_list = np.load(os.path.join(data_dir, "val_peptide_clusters.npy"))
       

        self.cluster_dict = np.load(os.path.join(data_dir, "peptide_clusters.npy"), allow_pickle=True).item()

        self.num_data = len(self.cluster_list)
        
        

    def __len__(self):
        return self.num_data
            
    def __getitem__(self, index):

        cluster = self.cluster_list[index]
        
        item = np.random.choice(self.cluster_dict[cluster])

        features = np.load(os.path.join(self.data_path, item + ".npy"),
                           allow_pickle=True).item()
            
       

        fragment_seq = features["fragment_seq"]
        
        # choose one model
        coordinates_index = np.random.choice(list(range(len(features["coordinates"]))))
        
        coordinates = features["coordinates"][coordinates_index]
        
        amino_acid_pos = features["amino_acid_pos"]
        atoms = features["atoms"]
        atom_names_full = features["atom_names"]
        bond_matrix = features["bond_matrix"]
        
        if self.model == "backbone" or self.model == "rotamer":
            
            atom_indices = atoms != atom_order["H"]
            
            coordinates = coordinates[atom_indices, :]
            
            amino_acid_pos = amino_acid_pos[atom_indices]
            
            atoms = atoms[atom_indices]
            
            atom_names_full = [atom_names_full[index] for index, i in enumerate(atom_indices) if i]
            
            bond_matrix = bond_matrix[atom_indices,  :][:, atom_indices]
        
        features = fragment_seq, coordinates, amino_acid_pos, atoms, atom_names_full,\
            bond_matrix
            
        
        if self.model == "backbone":
            if self.pad:
                return pad_features(_get_backbone_features(features))
            else:
                return convert_to_tensor(_get_backbone_features(features))
        elif self.model == "rotamer":
            if self.pad:
                return pad_features(_get_rotamer_features(features))
            else:
                return convert_to_tensor(_get_rotamer_features(features))
        elif self.model == "protonation":
            if self.pad:
                return pad_features(_get_hydrogen_features(features))
            else:
                return convert_to_tensor(_get_hydrogen_features(features))
            
            
class PeptideTestDataset(Dataset):

    def __init__(self, data_dir="../datasets",
                 data_path="./pep_features", linear_only=False, 
                 pad=True, model="backbone"):
       
        self.pad = pad
        
        self.data_path = data_path
        
        self.model = model
        
        
        
        if linear_only:
            self.data_list = np.load(os.path.join(data_dir, "test_pep_examples.npy"))
        else:
            self.data_list = np.concatenate([np.load(os.path.join(data_dir, "test_pep_examples.npy")),
                                             np.load(os.path.join(data_dir, "test_pep_examples_cyclic.npy"))])
                                             
       


        self.num_data = len(self.data_list)

    def __len__(self):
        return self.num_data
            
    def __getitem__(self, index):

      
        
        item = self.data_list[index]

        features = np.load(os.path.join(self.data_path, item + ".npy"),
                           allow_pickle=True).item()
            
       

        peptide_seq = features["fragment_seq"]
        coordinates_all = features["coordinates"]
        amino_acid_pos = features["amino_acid_pos"]
        atoms = features["atoms"]
        atom_names_full = features["atom_names"]
        bond_matrix = features["bond_matrix"]
        
        if self.model == "backbone" or self.model == "rotamer":
            
            atom_indices = atoms != atom_order["H"]
            
            coordinates_all = [coords[atom_indices, :] for coords in coordinates_all]
            
            amino_acid_pos = amino_acid_pos[atom_indices]
            
            atoms = atoms[atom_indices]
            
            atom_names_full = [atom_names_full[index] for index, i in enumerate(atom_indices) if i]
            
            bond_matrix = bond_matrix[atom_indices,  :][:, atom_indices]
        
       
            
        
        if self.model == "backbone":
            
            coordinates = coordinates_all[0]
            
            features = peptide_seq, coordinates, amino_acid_pos, atoms, atom_names_full,\
                bond_matrix
                
            if self.pad:
                return pad_features(_get_backbone_features(features))
            else:
                return convert_to_tensor(_get_backbone_features(features))
            
        elif self.model == "rotamer":
            
                
            all_features = []
            
            for coordinates in coordinates_all:
            
                features = peptide_seq, coordinates, amino_acid_pos,\
                    atoms, atom_names_full, bond_matrix
                    
                if self.pad:
                    all_features.append(pad_features(_get_rotamer_features(features)))
                else:
                    all_features.append(convert_to_tensor(_get_rotamer_features(features)))
                    
            return all_features
        
        elif self.model == "protonation":
            
                
            all_features = []
            
            for coordinates in coordinates_all:
            
                features = peptide_seq, coordinates, amino_acid_pos,\
                    atoms, atom_names_full, bond_matrix
                    
                if self.pad:
                    all_features.append(pad_features(_get_hydrogen_features(features)))
                else:
                    all_features.append(convert_to_tensor(_get_hydrogen_features(features)))
                    
            return all_features
            
    
            
class MDDataset(Dataset):

    def __init__(self, mode="train", data_dir="../datasets",
                 data_path="../md_pdbs/", pad=True, model="backbone", no_mod=False, 
                 num_repeats=100, sample_mds=1):
       
        self.pad = pad
        
        self.data_path = data_path
        
        self.model = model
        
        
        if mode == "train" and not no_mod:
            self.data_list = np.load(os.path.join(data_dir, "train_md_data.npy"))
        elif mode == "train":
            self.data_list = np.load(os.path.join(data_dir, "train_md_data_no_mod.npy"))
        elif mode == "val" and not no_mod:
            self.data_list = np.load(os.path.join(data_dir, "val_md_data.npy"))
        elif mode == "val":
            self.data_list = np.load(os.path.join(data_dir, "val_md_data_no_mod.npy"))
        elif mode == "test" and not no_mod:
            self.data_list = np.load(os.path.join(data_dir, "test_md_data.npy"))
        elif mode == "test":
            self.data_list = np.load(os.path.join(data_dir, "test_md_data_no_mod.npy"))
       

        self.mode = mode
    
        self.num_repeats = num_repeats

        self.num_data = len(self.data_list)
        
        if model != "protonation":
            ignore_hydrogens = True
        else:
            ignore_hydrogens = False
        if mode == "train" or mode == "val":
            self.pipeline = DataPipelineMD(md_path=data_path, sample_mds=sample_mds,
                                           ignore_hydrogens=ignore_hydrogens)
        else:
            self.pipeline = DataPipelineMD(md_path=data_path, sample_mds=None, ignore_hydrogens=ignore_hydrogens)

    def __len__(self):
        if self.mode == "train" or self.mode == "val":
            return int(self.num_data*self.num_repeats)
        else:
            return self.num_data
            
    def __getitem__(self, index):

      
        
        item = self.data_list[index % self.num_data]
        
        
      
        features = self.pipeline.compute_features(item)
        
    
        if self.model == "backbone":
            
            
            peptide_seq, coordinates_all, amino_acid_pos,\
                atoms, atom_names, bond_matrix = features
                
            all_features = []
            
            for coordinates in coordinates_all:
            
                features = peptide_seq, coordinates, amino_acid_pos,\
                    atoms, atom_names, bond_matrix
                    
                if self.pad:
                    all_features.append(pad_features(_get_backbone_features(features)))
                else:
                    all_features.append(convert_to_tensor(_get_backbone_features(features)))
                    
            return all_features
        elif self.model == "rotamer":
            
            peptide_seq, coordinates_all, amino_acid_pos,\
                atoms, atom_names, bond_matrix = features
                
            all_features = []
            
            for coordinates in coordinates_all:
            
                features = peptide_seq, coordinates, amino_acid_pos,\
                    atoms, atom_names, bond_matrix
                    
                if self.pad:
                    all_features.append(pad_features(_get_rotamer_features(features)))
                else:
                    all_features.append(convert_to_tensor(_get_rotamer_features(features)))
                    
            return all_features
        
        elif self.model == "protonation":
            
            peptide_seq, coordinates_all, amino_acid_pos,\
                atoms, atom_names, bond_matrix = features
                
            all_features = []
            
            for coordinates in coordinates_all:
            
                features = peptide_seq, coordinates, amino_acid_pos,\
                    atoms, atom_names, bond_matrix
                    
                if self.pad:
                    all_features.append(pad_features(_get_hydrogen_features(features)))
                else:
                    all_features.append(convert_to_tensor(_get_hydrogen_features(features)))
                    
            return all_features
        
        
     
class SequenceDataset(Dataset):

    def __init__(self, data_path="../datasets/all_training_seqs.npy", pad=True, temp=310):

        self.pad = pad   

        self.datalist = np.load(data_path)
        
        self.num_data = len(self.datalist)

        
        
        
    def __len__(self):
        return self.num_data

    
        
    def __getitem__(self, index):
        
        sequence = self.datalist[index]
        
        features = get_features_from_seq(sequence, batch_dim=False)
        
        return features
                            
                            
        

        
        
        
