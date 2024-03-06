from Bio import PDB
from pepflow.utils import constants
import itertools
import numpy as np
import torch

permutations = {}
permutations[3] = [np.array(i) for i in itertools.permutations([0,1,2])]
permutations[2] = [np.array(i) for i in itertools.permutations([0,1])]*3
permutations[1] = [np.array(i) for i in itertools.permutations([0])]*6

# functions to pad features

def _pad(array, target_shape, value):
    return np.pad(
        array,
        [(0, target_shape[i] - array.shape[i]) for i in range(len(array.shape))],
        "constant", constant_values=value
    )


def pad_features(features):
    
    features_padded = {}
    
    for feature in features:
        
        if "names" in feature or "cyclic_bond_indices" in feature:
            features_padded[feature] = features[feature]

        elif "permuted_coords" in feature:
            features_padded[feature] = torch.as_tensor(_pad(features[feature],
                feature_dims[feature][0],
                feature_dims[feature][2]),
                dtype=feature_dims[feature][1]).unsqueeze(1)
            
        elif "coordinates" in feature:
            features_padded[feature] = torch.as_tensor(_pad(features[feature],
                feature_dims[feature][0],
                feature_dims[feature][2]),
                dtype=feature_dims[feature][1]).unsqueeze(0)

        else:
            features_padded[feature] = torch.as_tensor(_pad(features[feature],
                                            feature_dims[feature][0],
                                            feature_dims[feature][2]),
                                            dtype=feature_dims[feature][1])
    
    return features_padded


def convert_to_tensor(features):
    
    features_padded = {}
    
    for feature in features:
        
        if "fragment_seq" in feature:
            features_padded[feature] = torch.as_tensor(_pad(features[feature],
                                            feature_dims[feature][0],
                                            feature_dims[feature][2]),
                                            dtype=feature_dims[feature][1])
        elif "names" in feature or "cyclic_bond_indices" in feature:
            features_padded[feature] = features[feature]
        elif "permuted_coords" in feature:
            features_padded[feature] = torch.as_tensor(features[feature],
                dtype=feature_dims[feature][1]).unsqueeze(1)    
        elif "coordinates" in feature:
            features_padded[feature] = torch.as_tensor(features[feature],
                dtype=feature_dims[feature][1]).unsqueeze(0)
        else:
            features_padded[feature] = torch.as_tensor(features[feature],
                                            dtype=feature_dims[feature][1])
    
    return features_padded    
    
# feature dimensions

MIN_LENGTH = 3
MAX_LENGTH = 15
MAX_LENGTH_ATOMS = MAX_LENGTH*15
MAX_LENGTH_BACKBONE = MAX_LENGTH*5 + 1

feature_dims = {'fragment_seq':[(MAX_LENGTH,), torch.long, 20],
    'coordinates_rotamer':[(MAX_LENGTH_ATOMS, 3), torch.float, 0],
    'bond_matrix_rotamer':[(MAX_LENGTH_ATOMS, MAX_LENGTH_ATOMS), torch.float, 0],
    'atoms_rotamer':[(MAX_LENGTH_ATOMS,), torch.long, 0],
    'amino_acid_pos_rotamer':[(MAX_LENGTH_ATOMS,), torch.long, 1],
    'edge_mask_rotamer':[(MAX_LENGTH_ATOMS, MAX_LENGTH_ATOMS), torch.float, 0],
    'atom_mask_rotamer':[(MAX_LENGTH_ATOMS, 3), torch.float, 0],
    'centroid_coordinates':[(MAX_LENGTH, 3), torch.float, 0],
    'centroid_indices':[(MAX_LENGTH,), torch.long, -1],
    'coordinates_h':[(MAX_LENGTH_ATOMS, 3), torch.float, 0],
    'coordinates_heavy_h':[(MAX_LENGTH_ATOMS, 3), torch.float, 0],
    'atoms_h':[(MAX_LENGTH_ATOMS,), torch.long, 0],
    'atoms_heavy':[(MAX_LENGTH_ATOMS,), torch.long, 0],
    'amino_acid_pos_h':[(MAX_LENGTH_ATOMS,), torch.long, 1],
    'amino_acid_pos_heavy':[(MAX_LENGTH_ATOMS,), torch.long, 1],
    'bond_matrix_h_heavy':[(MAX_LENGTH_ATOMS, MAX_LENGTH_ATOMS), torch.float, 0],
    'edges_h_h':[(MAX_LENGTH_ATOMS*3, 2), torch.long, 0],
    'edge_mask_heavy_neighbors':[(MAX_LENGTH_ATOMS, MAX_LENGTH_ATOMS), torch.float, 0],
    'edge_mask_h_heavy':[(MAX_LENGTH_ATOMS, MAX_LENGTH_ATOMS), torch.float, 0],
    'edge_mask_h_h':[(MAX_LENGTH_ATOMS*3,), torch.float, 0],
    'atom_mask_h':[(MAX_LENGTH_ATOMS, 3), torch.float, 0],
    'atom_mask_heavy':[(MAX_LENGTH_ATOMS, 3), torch.float, 0],
    'bound_atom':[(MAX_LENGTH_ATOMS,), torch.long, 0],
    'all_permutations':[(6, MAX_LENGTH_ATOMS), torch.long, MAX_LENGTH_ATOMS-1],
    'permuted_coords':[(6, MAX_LENGTH_ATOMS, 3), torch.float, 0],
    'coordinates_backbone':[(MAX_LENGTH_BACKBONE, 3), torch.float, 0],
    'bond_matrix_backbone':[(MAX_LENGTH_BACKBONE, MAX_LENGTH_BACKBONE), torch.float, 0],
    'atoms_backbone':[(MAX_LENGTH_BACKBONE,), torch.long, 0],
    'amino_acid_pos_backbone':[(MAX_LENGTH_BACKBONE,), torch.long, 1],
    'edge_mask_backbone':[(MAX_LENGTH_BACKBONE, MAX_LENGTH_BACKBONE), torch.float, 0],
    'atom_mask_backbone':[(MAX_LENGTH_BACKBONE, 3), torch.float, 0],
    'reference_indices':[(MAX_LENGTH_BACKBONE, ), torch.long, MAX_LENGTH_BACKBONE-1],
    'restore_indices':[(MAX_LENGTH_BACKBONE, MAX_LENGTH_BACKBONE), torch.float, 0],
    }


# collate function


def collate(batch):
    
    batch = [i for i in batch if i != None]
    
    if len(batch) == 0:
        return None

    features = {}
    
    for feature in batch[0]:
        
        
        if "atom_names" not in feature and "cyclic_bond_indices" not in feature:
            
            features[feature] = torch.stack([i[feature] for i in batch], dim=0)
            
            
        else:
            
            features[feature] = [i[feature] for i in batch]

    return features


def collate_processed(batch):
    
    batch = [i for i in batch if i != None]
    
    if len(batch) == 0:
        return None

    features = {}
    
    for feature in batch[0]:
        
        
        if "atom_names" not in feature and "cyclic_bond_indices" not in feature:
            
            features[feature] = torch.cat([i[feature] for i in batch], dim=0)
            
            
        else:
            
            features[feature] = [i[feature][0] for i in batch]

    return features


def collate_multiple_coords(batch):
    
    list_batchs = [i for i in batch if i != None]
    
    if len(list_batchs) == 0:
        return None

    batch = []
    
    
    for list_batch in list_batchs:
        
        batch_features = {}
        
        for feature in list_batch[0]:
            if feature == "permuted_coords":
            
                batch_features[feature] = torch.concat([i[feature] for i in list_batch], dim=1)

            elif "coordinates" in feature  or ("edge_mask" in feature and "heavy" in feature)\
                or ("edge_mask_rotamer" in feature):
                
                if "coordinates" in feature:
                    
                    batch_features[feature] = torch.concat([i[feature] for i in list_batch], dim=0)
                
                else:  
                    batch_features[feature] = torch.stack([i[feature] for i in list_batch], dim=0)
                
            else:
                batch_features[feature] = list_batch[0][feature]
                
        batch.append(batch_features)
        
        
    features = {}
    
    for feature in batch[0]:
        
        
        if "atom_names" not in feature and "cyclic_bond_indices" not in feature:
            
            features[feature] = torch.stack([i[feature] for i in batch], dim=0)
            
            
        else:
            
            features[feature] = [i[feature] for i in batch]

    return features


# function to get features from sequence

def get_features_from_seq(sequence, rotamer_feats=True, protonation_feats=True,
                          batch_dim=True, cyclic_bonds=None):
    
    
    '''
    
    compute backbone features
    
    '''
    
    encoded_pep_seq = []
    
    atoms  = []
    
    atom_names = []
    
    amino_acid_pos = []
    
    cyclic_bond_atoms = []
    
    if cyclic_bonds != None:
        for bond in cyclic_bonds:
        
            cyclic_bond_atoms.append(bond[0])
            cyclic_bond_atoms.append(bond[1])
        
        
    hydrogens_to_exclude = []
    
    for atom in cyclic_bond_atoms:
        
        aa = int(atom.split("_")[0]) - 1
        
        res = sequence[aa]
        
        all_bonds = constants.COVALENT_BOND_DICT[PDB.Polypeptide.one_to_three(res)]
        
        if atom.split("_")[1] == "N":
            continue

        for bond in all_bonds:
            
            if bond[0] == atom.split("_")[1]:
                
                if bond[1].startswith("H"):
                    
                    hydrogens_to_exclude.append(atom.split("_")[0] + "_" + bond[1])
                    
                    break
    
    for index, residue in enumerate(sequence):
        
        encoded_pep_seq.append(constants.residue_order[residue])
        

            
            
        for atom in constants.ATOM_DICT[PDB.Polypeptide.one_to_three(residue)]:
            if atom not in ["C", "O", "N", "CA", "OXT"]:
                continue

            if atom.startswith("H") or  (atom == "OXT" and index != len(sequence) - 1):
                continue
            
            elif (residue, atom) in constants.ambiguous_mapping:
            
                atom_names.append(atom)
            
                atoms.append(constants.atom_order[constants.ambiguous_mapping[(residue, atom)]])
            
                amino_acid_pos.append(index+1)
            elif atom == "OXT" and str(index+1) + "_C" not in cyclic_bond_atoms:
                                
                atoms.append(constants.atom_order["O"])
                atom_names.append(atom)
                amino_acid_pos.append(index+1)
            elif atom == "OXT" and str(index+1) + "_C" in cyclic_bond_atoms:
                continue
            else:
                atom_names.append(atom)
                
                if not atom.startswith("H"):
                    atoms.append(constants.atom_order[atom])
                else:
                    atoms.append(constants.atom_order["H"])
                
                amino_acid_pos.append(index+1)
                
        if residue != "G":

            atom_names.append("CEN")
            atoms.append(4)
            amino_acid_pos.append(index+1)
    
    bond_matrix = np.zeros((len(atoms), len(atoms)))

    for i in range(len(atoms)):
        for j in range(len(atoms)):

            if atoms[i] == 4 or atoms[j] == 4:
                continue

            if amino_acid_pos[i] == amino_acid_pos[j]:
                
                one_letter = constants.RESIDUE_LIST[encoded_pep_seq[amino_acid_pos[i]-1]]
                three_letter = PDB.Polypeptide.one_to_three(
                    one_letter)
                
                if (atom_names[i], atom_names[j]) in constants.COVALENT_BOND_DICT[three_letter.upper()]\
                        or (atom_names[j], atom_names[i]) in constants.COVALENT_BOND_DICT[three_letter.upper()]:

                    bond_matrix[i, j] = 1

            elif amino_acid_pos[i] == amino_acid_pos[j] - 1:
                if atom_names[j] == "N" and atom_names[i] == "C":

                    bond_matrix[i, j] = 1

            elif amino_acid_pos[i] == amino_acid_pos[j] + 1:
                if atom_names[j] == "C" and atom_names[i] == "N":

                    bond_matrix[i, j] = 1
    
    

    amino_acid_pos = np.array(amino_acid_pos)
    atoms = np.array(atoms)
    
    backbone_cyclic_bond_indices = []
    
    if cyclic_bonds != None:
        for bond in cyclic_bonds:
            
            if bond[0].split("_")[1] not in ["C", "O", "N", "CA"] or\
                bond[1].split("_")[1] not in ["C", "O", "N", "CA"]:
                    
                    continue
                
            indices = get_bond_atom_indices(bond, atoms, amino_acid_pos)
            
            bond_matrix[indices[0], indices[1]] = 1
            
            bond_matrix[indices[1], indices[0]] = 0
            
            
            backbone_cyclic_bond_indices.append(indices)
            
        
   
 
    
    reference_indices = np.array(list(range(len(atoms))))
    
    same_residue = amino_acid_pos[:, None] == amino_acid_pos[None, :]
    previous_residue = (amino_acid_pos[:, None] - 1) == amino_acid_pos[None, :]
    
    n_c = np.argwhere(np.logical_and(np.logical_and(previous_residue, atoms[:, None] == constants.atom_order["N"]),
                         atoms[None, :] == constants.atom_order["C"]))
    
    
    reference_indices[n_c[:, 0]] = n_c[:, 1]
    
    ca_n = np.argwhere(np.logical_and(np.logical_and(same_residue, atoms[:, None] == constants.atom_order["CA"]),
                         atoms[None, :] == constants.atom_order["N"]))
    
    
    reference_indices[ca_n[:, 0]] = ca_n[:, 1]
    
    
    c_ca = np.argwhere(np.logical_and(np.logical_and(same_residue, atoms[:, None] == constants.atom_order["C"]),
                         atoms[None, :] == constants.atom_order["CA"]))
    
    
    reference_indices[c_ca[:, 0]] = c_ca[:, 1]
    

    o_c = np.argwhere(np.logical_and(np.logical_and(same_residue, atoms[:, None] == constants.atom_order["O"]),
                         atoms[None, :] == constants.atom_order["C"]))
    
    
    reference_indices[o_c[:, 0]] = o_c[:, 1]
    
    cen_ca = np.argwhere(np.logical_and(np.logical_and(same_residue, atoms[:, None] == 4),
                         atoms[None, :] == constants.atom_order["CA"]))
    
    
    reference_indices[cen_ca[:, 0]] = cen_ca[:, 1]
    
    restore_indices = np.eye(len(reference_indices))
    
    restore_indices[np.array(list(range(len(reference_indices)))),
                    reference_indices] = 1
    restore_indices = np.linalg.matrix_power(restore_indices, len(atoms)) > 0
    

    
    edge_mask = np.ones((len(atoms), len(atoms)))
    atom_mask = np.ones((len(atoms), 3))

    features = {}

    features["atom_names_backbone"] = atom_names
    features["fragment_seq"] = np.array(encoded_pep_seq)
    features["atoms_backbone"] = np.array(atoms)
    features["amino_acid_pos_backbone"] = np.array(amino_acid_pos)
    features["bond_matrix_backbone"] = bond_matrix
    features["edge_mask_backbone"] = edge_mask
    features["atom_mask_backbone"] = atom_mask
    features["restore_indices"] = restore_indices
    features["reference_indices"] = reference_indices
    features["backbone_cyclic_bond_indices"] = backbone_cyclic_bond_indices
    
    if not rotamer_feats and not protonation_feats:
        if batch_dim:
            return collate([convert_to_tensor(features)])
        else:
            return pad_features(features)
    
    '''
    compute rotamer features
    '''


    centroid_counter = 0
    
    centroid_indices = []
    
    atoms  = []
    
    atom_names = []
    
    amino_acid_pos = []
    
    
    for index, residue in enumerate(sequence):
        

        
        for atom in constants.ATOM_DICT[PDB.Polypeptide.one_to_three(residue)]:
      

            if atom.startswith("H") or  (atom == "OXT" and index != len(sequence) - 1):
                continue
            elif residue == "E" and str(index+1) + "_CD" in cyclic_bond_atoms and atom == "OE1":
                continue
            elif residue == "D" and str(index+1) + "_CG" in cyclic_bond_atoms and atom == "OD1":
                continue
            elif (residue, atom) in constants.ambiguous_mapping:
            
                atom_names.append(atom)
            
                atoms.append(constants.atom_order[constants.ambiguous_mapping[(residue, atom)]])
            
                amino_acid_pos.append(index+1)
            elif atom == "OXT" and str(index+1) + "_C" not in cyclic_bond_atoms:
                                
                atoms.append(constants.atom_order["O"])
                atom_names.append(atom)
                amino_acid_pos.append(index+1)
            elif atom == "OXT" and str(index+1) + "_C" in cyclic_bond_atoms:
                
                continue

            elif atom == "OXT":
                atoms.append(constants.atom_order["O"])
                atom_names.append(atom)
                amino_acid_pos.append(index+1)
            else:
                atom_names.append(atom)
                
                atoms.append(constants.atom_order[atom])
                
                amino_acid_pos.append(index+1)
      
        if residue == "G":
            centroid_indices.append(-1)
        else:
            centroid_indices.append(centroid_counter)
        
            centroid_counter += 1
            
    bond_matrix = np.zeros((len(atoms), len(atoms)))

    for i in range(len(atoms)):
        for j in range(len(atoms)):


            if amino_acid_pos[i] == amino_acid_pos[j]:
                
                one_letter = constants.RESIDUE_LIST[encoded_pep_seq[amino_acid_pos[i]-1]]
                three_letter = PDB.Polypeptide.one_to_three(
                    one_letter)
                
                if (atom_names[i], atom_names[j]) in constants.COVALENT_BOND_DICT[three_letter.upper()]\
                        or (atom_names[j], atom_names[i]) in constants.COVALENT_BOND_DICT[three_letter.upper()]:

                    bond_matrix[i, j] = 1

            elif amino_acid_pos[i] == amino_acid_pos[j] - 1:
                if atom_names[j] == "N" and atom_names[i] == "C":

                    bond_matrix[i, j] = 1

            elif amino_acid_pos[i] == amino_acid_pos[j] + 1:
                if atom_names[j] == "C" and atom_names[i] == "N":

                    bond_matrix[i, j] = 1
    
    centroid_indices = np.array(centroid_indices)
    
    is_backbone = np.isin(atoms, constants.backbone_atom_indices)
    
    backbone_indices = np.flatnonzero(is_backbone)
   
    # generate masks
   
    atom_mask = np.ones((len(atoms), 3))
    atom_mask[backbone_indices, :] = 0

    # convert to numpy arrays
    
    atoms = np.array(atoms)

    amino_acid_pos = np.array(amino_acid_pos)
    
    
    # get cyclic bond coords
    
    rotamer_cyclic_bond_indices = []
    
    if cyclic_bonds != None:
        for bond in cyclic_bonds:
            
                
            indices = get_bond_atom_indices(bond, atoms, amino_acid_pos)
            
            bond_matrix[indices[0], indices[1]] = 1
            
            bond_matrix[indices[1], indices[0]] = 0
            
            
            rotamer_cyclic_bond_indices.append(indices)
            
            
    
    # add features to dict
    
    features['bond_matrix_rotamer'] = bond_matrix
    features['atoms_rotamer'] = np.array(atoms)
    features['amino_acid_pos_rotamer'] = amino_acid_pos
    features['edge_mask_rotamer'] = edge_mask
    features['atom_mask_rotamer'] = atom_mask
    features['atom_names_rotamer'] = atom_names
    features['centroid_indices'] = centroid_indices
    features['rotamer_cyclic_bond_indices'] = rotamer_cyclic_bond_indices
    
    if not protonation_feats:
        if batch_dim:
            return collate([convert_to_tensor(features)])
        else:
            return pad_features(features)
    
    '''
    compute hydrogen features
    
    '''
    
    atoms  = []
    
    atom_names = []
    
    amino_acid_pos = []
    
    
    for index, residue in enumerate(sequence):
        

        
        for atom in constants.ATOM_DICT[PDB.Polypeptide.one_to_three(residue)]:
      
            if (atom == "OXT" and index != len(sequence) - 1):
                continue
            elif (atom == "H2" or atom == "H3") and index != 0:
                continue
            elif (atom == "H2" or atom == "H3") and str(index+1) + "_N" in cyclic_bond_atoms:
                continue
            elif residue == "E" and str(index+1) + "_CD" in cyclic_bond_atoms and atom == "OE1":
                continue
            elif residue == "D" and str(index+1) + "_CG" in cyclic_bond_atoms and atom == "OD1":
                continue
            elif (residue, atom) in constants.ambiguous_mapping:
            
                atom_names.append(atom)
            
                atoms.append(constants.atom_order[constants.ambiguous_mapping[(residue, atom)]])
            
                amino_acid_pos.append(index+1)
                
            elif atom == "OXT" and str(index+1) + "_C" in cyclic_bond_atoms:
                
                continue

            elif atom == "OXT":
                atoms.append(constants.atom_order["O"])
                atom_names.append(atom)
                amino_acid_pos.append(index+1)
            else:
                
                if not atom.startswith("H"):
                    atoms.append(constants.atom_order[atom])
                else:
                    
                    if not str(index+1) + "_" + atom in hydrogens_to_exclude:
                        atoms.append(constants.atom_order["H"])
                    else:
                        continue

                atom_names.append(atom)
                amino_acid_pos.append(index+1)
      
            
    bond_matrix = np.zeros((len(atoms), len(atoms)))

    for i in range(len(atoms)):
        for j in range(len(atoms)):


            if amino_acid_pos[i] == amino_acid_pos[j]:
                
                one_letter = constants.RESIDUE_LIST[encoded_pep_seq[amino_acid_pos[i]-1]]
                three_letter = PDB.Polypeptide.one_to_three(
                    one_letter)
                
                if (atom_names[i], atom_names[j]) in constants.COVALENT_BOND_DICT[three_letter.upper()]\
                        or (atom_names[j], atom_names[i]) in constants.COVALENT_BOND_DICT[three_letter.upper()]:

                    bond_matrix[i, j] = 1

            elif amino_acid_pos[i] == amino_acid_pos[j] - 1:
                if atom_names[j] == "N" and atom_names[i] == "C":

                    bond_matrix[i, j] = 1

            elif amino_acid_pos[i] == amino_acid_pos[j] + 1:
                if atom_names[j] == "C" and atom_names[i] == "N":

                    bond_matrix[i, j] = 1
                    
    # identify hydrogen atoms and heavy atoms
    
    atoms = np.array(atoms)

    amino_acid_pos = np.array(amino_acid_pos)

    if cyclic_bonds != None:
        for bond in cyclic_bonds:
            
                
            indices = get_bond_atom_indices(bond, atoms, amino_acid_pos)
            
            bond_matrix[indices[0], indices[1]] = 1
            
            bond_matrix[indices[1], indices[0]] = 0
            
            
    is_h = np.array(atoms) == constants.atom_order["H"]
    
  
    
    atoms_h = atoms[is_h]
    
    atoms_heavy = atoms[np.logical_not(is_h)]

    amino_acid_pos_heavy = amino_acid_pos[np.logical_not(is_h)] 
                                            
    amino_acid_pos_h = amino_acid_pos[is_h]
    
    # get the atom names for heavy and hydrogen atoms
    
    atom_names_heavy = [i for index, i in enumerate(atom_names) if np.logical_not(is_h)[index]]
    
    atom_names_h = [i for index, i in enumerate(atom_names) if is_h[index]]
    
    
   
    
    # generate edge mask between hydrogens and heavy atoms
    
    bound_atom = np.argwhere(bond_matrix[is_h, :][:, np.logical_not(is_h)] == 1)[:, 1]
    
    
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
            
    features["atoms_h"] = atoms_h
    features["atoms_heavy"] = atoms_heavy
    features["amino_acid_pos_h"] = amino_acid_pos_h
    features["amino_acid_pos_heavy"] = amino_acid_pos_heavy
    features["bond_matrix_h_heavy"] = bond_matrix_h_heavy
    features["edges_h_h"] = edges_h_h
    features["edge_mask_h_h"] = edge_mask_h_h
    features["atom_mask_h"] = atom_mask
    features["atom_mask_heavy"] = atom_mask_heavy
    features["bound_atom"] = bound_atom
    features["atom_names_heavy"] = atom_names_heavy
    features["atom_names_h"] = atom_names_h
    features["all_permutations"] = all_permutations
    
    if batch_dim:
        return collate([convert_to_tensor(features)])
    else:
        return pad_features(features)

def get_bond_atom_indices(bond, atoms, amino_acid_pos):
    
    indices = [np.argwhere(np.logical_and(atoms == constants.atom_order[bond[0].split("_")[1]],
                                                       amino_acid_pos == int(bond[0].split("_")[0])))[0][0],
                            np.argwhere(np.logical_and(atoms == constants.atom_order[bond[1].split("_")[1]],
                                                       amino_acid_pos == int(bond[1].split("_")[0])))[0][0]]

            
            
    return indices
