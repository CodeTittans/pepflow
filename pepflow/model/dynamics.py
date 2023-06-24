from pepflow.model.layers.gnn_layers import GNNLayerTrainable, GNNLayer
from pepflow.utils.tensor_utils import cdist, get_edge_indices, get_extra_edge_indices
from pepflow.utils.constants import backbone_atom_indices 
from torch_scatter import scatter
import torch.nn.functional as F
import torch.nn as nn
import torch

class DynamicsBackbone(nn.Module):

    def __init__(self, model_config):

        super(DynamicsBackbone, self).__init__()

        self.num_layers = model_config.num_layers

        layers = []

        for i in range(model_config.num_layers):

            layers.append(GNNLayer())

        self.layers = nn.ModuleList(layers)

        
        layers_trainable = []
        
        for i in range(model_config.num_layers_trainable):

            layers_trainable.append(GNNLayerTrainable(model_config.dims_trainable\
                                                      [i*int(len(model_config.dims_trainable)/model_config.num_layers_trainable):
                                                 (i+1)*int(len(model_config.dims_trainable)/model_config.num_layers_trainable)]))

        self.layers_trainable = nn.ModuleList(layers_trainable)        
   

    def forward(self, t, x, peptide_seq, atom_labels, amino_acid_pos, bond_matrix, 
                   edge_mask, atom_mask, restore_indices, w_x, b_x, w_t, w_b_t):
        
        '''
        Generate single atom embedding
        '''
        
        atom_one_hot = F.one_hot(atom_labels, num_classes=5)
        
        amino_acid_one_hot = F.one_hot(torch.gather(peptide_seq, dim=1, index=amino_acid_pos-1),
                                       num_classes=20)
        
        position_one_hot = F.one_hot(amino_acid_pos-1, num_classes=15)
        
        embedding = torch.cat([atom_one_hot, amino_acid_one_hot, position_one_hot],
                              dim=-1)
        
        '''
        Get edge indices
        '''
        samples_per_seq = x.shape[1]
        
        atom_features = embedding.unsqueeze(1).repeat(1, samples_per_seq, 1, 1)

        edge_row, edge_col = get_edge_indices(atom_features.shape[2], 
                                              atom_features.device) 
        
        '''
        Compute absolute coordinates
        '''
        x =  torch.sum(x.unsqueeze(2).repeat(1, 1, x.shape[2], 1, 1)*\
               restore_indices[:, None, :, :, None], axis=3)
                    
        x_original = x.clone()
        
        if len(edge_mask.shape) < 4:
            
            edge_mask = edge_mask.unsqueeze(1)
            
        edge_mask = edge_mask.view(edge_mask.shape[0], edge_mask.shape[1], -1)
                       
        for index, layer in enumerate(self.layers):

            
            differences, distances = cdist(x, x)
            
            
            edge_features = torch.cat([atom_features[:, :, edge_row],
                                       atom_features[:, :, edge_col],
                                       distances[:, :, edge_row, edge_col].unsqueeze(-1),
                                       bond_matrix[:, edge_row, edge_col].unsqueeze(-1)\
                                           .unsqueeze(1).repeat(1, differences.shape[1], 1, 1)], dim=-1)
            
            differences = differences[:, :, edge_row, edge_col, :]
            
            x_out, h_out = layer(t, atom_features, differences, edge_features, 
                                 edge_row, edge_col, edge_mask, atom_mask,
                                 w_x[int(len(w_x)/self.num_layers*index)
                                     :int(len(w_x)/self.num_layers*(index+1))],
                                 b_x[int(len(b_x)/self.num_layers*index)
                                         :int(len(b_x)/self.num_layers*(index+1))],
                                 w_t[int(len(w_t)/self.num_layers*index)
                                         :int(len(w_t)/self.num_layers*(index+1))],
                                 w_b_t[int(len(w_b_t)/self.num_layers*index)
                                       :int(len(w_b_t)/self.num_layers*(index+1))])

            x = x + x_out
            
            atom_features = h_out
         
        for index, layer in enumerate(self.layers_trainable):

            
            differences, distances = cdist(x, x)
            
            
            edge_features = torch.cat([atom_features[:, :, edge_row],
                                       atom_features[:, :, edge_col],
                                       distances[:, :, edge_row, edge_col].unsqueeze(-1),
                                       bond_matrix[:, edge_row, edge_col].unsqueeze(-1)\
                                           .unsqueeze(1).repeat(1, differences.shape[1], 1, 1)], dim=-1)
            
            differences = differences[:, :, edge_row, edge_col, :]
            
            x_out, h_out = layer(t, atom_features, differences, edge_features, 
                                 edge_row, edge_col, edge_mask, atom_mask)

            x = x + x_out
            
            atom_features = h_out
            
        prediction = x - x_original
        
        prediction = prediction * atom_mask[:, None, :, :]

        return prediction
  
    
class DynamicsRotamer(nn.Module):

    def __init__(self, model_config):

        super(DynamicsRotamer, self).__init__()

        self.num_layers = model_config.num_layers

        layers = []

        for i in range(model_config.num_layers):

            layers.append(GNNLayerTrainable(model_config.dims[i*int(len(model_config.dims)/model_config.num_layers):
                                                 (i+1)*int(len(model_config.dims)/model_config.num_layers)]))

        self.layers = nn.ModuleList(layers)


   

    def forward(self, t, x, features):
        
        peptide_seq, atom_labels, amino_acid_pos, bond_matrix,\
           edge_mask, atom_mask = features["fragment_seq"],  features["atoms_rotamer"],\
                   features["amino_acid_pos_rotamer"], features["bond_matrix_rotamer"],\
                       features["edge_mask_rotamer"], features["atom_mask_rotamer"]
        
        '''
        Generate single atom embedding
        '''
        atom_one_hot = F.one_hot(atom_labels, num_classes=43)

        amino_acid_one_hot = F.one_hot(torch.gather(peptide_seq, dim=1, index=amino_acid_pos-1),
                                       num_classes=20)
        
        
        position_one_hot = F.one_hot(amino_acid_pos-1, num_classes=15)
        
        embedding = torch.cat([atom_one_hot, amino_acid_one_hot, position_one_hot],
                              dim=-1)
        
        '''
        Get edge indices
        '''
        samples_per_seq = x.shape[1]
        
        atom_features = embedding.unsqueeze(1).repeat(1, samples_per_seq, 1, 1)

        edge_row, edge_col = get_edge_indices(atom_features.shape[2], 
                                              atom_features.device)        
        
        x_original = x.clone()
        
        if len(edge_mask.shape) < 4:
            
            edge_mask = edge_mask.unsqueeze(1)
            
        edge_mask = edge_mask.view(edge_mask.shape[0], edge_mask.shape[1], -1)
                       
        for index, layer in enumerate(self.layers):

            
            differences, distances = cdist(x, x)
            
            
            edge_features = torch.cat([atom_features[:, :, edge_row],
                                       atom_features[:, :, edge_col],
                                       distances[:, :, edge_row, edge_col].unsqueeze(-1),
                                       bond_matrix[:, edge_row, edge_col].unsqueeze(-1)\
                                           .unsqueeze(1).repeat(1, differences.shape[1], 1, 1)], dim=-1)
            
            differences = differences[:, :, edge_row, edge_col, :]
            
            x_out, h_out = layer(t, atom_features, differences, edge_features, 
                                 edge_row, edge_col, edge_mask, atom_mask)

            x = x + x_out
            
            atom_features = h_out
            
        prediction = x - x_original
        
        prediction = prediction * atom_mask[:, None, :, :]
 
        curr_means = scatter(src=prediction, dim=2, index=amino_acid_pos.long().unsqueeze(1)-1,
                                          reduce="sum")
        
        curr_means = curr_means/(scatter(src=atom_mask[:, None, :, :], dim=2, index=amino_acid_pos.long().unsqueeze(1)-1,
                                                              reduce="sum") +1e-8)
        
        curr_means = torch.stack([curr_means[i][:, amino_acid_pos[i].long() - 1, :] for i in range(len(x))])
        
        prediction = (prediction - curr_means) * atom_mask[:, None, :, :]
        

        return prediction
  
    def get_score_fn(self, features, sde, side_chain_only=False):
        
        
        atom_indices = torch.logical_not((features["atoms_rotamer"][..., None] ==\
                                                       torch.LongTensor(backbone_atom_indices).to(features["atom_mask_rotamer"].device)).any(-1))
        
        full_mask = torch.logical_not((features["atoms_rotamer"][..., None] ==\
                                                                          torch.LongTensor(backbone_atom_indices).to(features["atom_mask_rotamer"].device)).any(-1)).unsqueeze(-1).repeat(1,1,3)
        def score_rotamer(perturbed_data, t):
           
            if side_chain_only:
                _, std = sde.marginal_prob(torch.zeros(perturbed_data.shape).to(features["fragment_seq"].device), 
                       t, torch.stack([features["atom_mask_rotamer"][i, atom_indices[i], :] for i in range(len(atom_indices))]) , 
                       torch.stack([features["amino_acid_pos_rotamer"][i, atom_indices[i]] for i in range(len(features["amino_acid_pos_rotamer"]))]).long() - 1)
                
                x = features["coordinates_rotamer"].masked_scatter(full_mask.unsqueeze(
                                   1).repeat(1, perturbed_data.shape[1], 1, 1), perturbed_data)
            else:
                
                _, std = sde.marginal_prob(torch.zeros(perturbed_data.shape).to(features["fragment_seq"].device),
                        t, features["atom_mask_rotamer"], features["amino_acid_pos_rotamer"].long() - 1)
                
                x = perturbed_data
           
            prediction = self(t*999, x, features)
           
            if side_chain_only:
               
                prediction = torch.stack([prediction[i, :, atom_indices[i], :] for i in range(len(atom_indices))])
           
           
            return -1*prediction/std[:, None, None, None].to(prediction.device)


        return score_rotamer 
    
    
    
class DynamicsHydrogen(nn.Module):

    def __init__(self, model_config):

        super(DynamicsHydrogen, self).__init__()

        self.num_layers = model_config.num_layers

        layers = []

        for i in range(model_config.num_layers):

            layers.append(GNNLayerTrainable(model_config.dims[i*int(len(model_config.dims)/model_config.num_layers):
                                                 (i+1)*int(len(model_config.dims)/model_config.num_layers)]))

        self.layers = nn.ModuleList(layers)


   

    def forward(self, t, x_h, features):
       
        x_heavy, peptide_seq, atom_labels_heavy, atom_labels_h,\
                amino_acid_pos_heavy, amino_acid_pos_h,\
                    bond_matrix_h_heavy, edges_h_h, edge_mask_h_heavy, edge_mask_h_h,\
                        atom_mask, bound_atom =  features["coordinates_heavy_h"],\
                            features["fragment_seq"], features["atoms_heavy"],\
                                features["atoms_h"], features["amino_acid_pos_heavy"],\
                                    features["amino_acid_pos_h"], features["bond_matrix_h_heavy"],\
                                        features["edges_h_h"], features["edge_mask_h_heavy"],\
                                            features["edge_mask_h_h"], features["atom_mask_h"], features["bound_atom"]
                    
        
        if x_heavy.shape[1] != x_h.shape[1]:

            x_heavy = x_heavy.repeat(1, x_h.shape[1], 1, 1)

        '''
        Generate single atom embedding
        '''
        atom_one_hot_heavy = F.one_hot(atom_labels_heavy, num_classes=44)
        
        amino_acid_one_hot_heavy = F.one_hot(torch.gather(peptide_seq, dim=1, 
                                                          index=amino_acid_pos_heavy-1),
                                             num_classes=20)
        
     
        position_one_hot_heavy = F.one_hot(amino_acid_pos_heavy-1, num_classes=15)

        embedding_heavy = torch.cat([atom_one_hot_heavy, amino_acid_one_hot_heavy,
                                     position_one_hot_heavy],
                              dim=-1)
        
        
        atom_one_hot_h = F.one_hot(atom_labels_h, num_classes=44)
        
        amino_acid_one_hot_h = F.one_hot(torch.gather(peptide_seq, dim=1, 
                                                          index=amino_acid_pos_h-1),
                                             num_classes=20)
        
        position_one_hot_h = F.one_hot(amino_acid_pos_h-1, num_classes=15)
        
        embedding_h = torch.cat([atom_one_hot_h, amino_acid_one_hot_h, position_one_hot_h],
                              dim=-1)
        
        '''
        Compute absolute coordinates
        '''
        num_h = x_h.shape[2]
        
        x_h = torch.stack([x_h[i] + x_heavy[i][:, bound_atom[i], :]\
                           for i in range(len(x_h))], dim=0)
        
        x = torch.cat((x_h, x_heavy), dim=-2)
        
        atom_features = torch.cat((embedding_h, embedding_heavy), dim=1)
        
        bond_matrix_h_h = torch.zeros((bond_matrix_h_heavy.shape[0],
                                       num_h, num_h)).to(bond_matrix_h_heavy.device)

        bond_matrix = torch.cat((bond_matrix_h_h, bond_matrix_h_heavy), dim=-1)
        
        samples_per_seq = x_h.shape[1]
        
        atom_features = atom_features.unsqueeze(1).repeat(1, samples_per_seq, 1, 1)

        edge_row, edge_col = get_extra_edge_indices(edges_h_h, 
                                                    num_h, 
                                                    atom_features.shape[2], 
                                                    atom_features.shape[0], 
                                                    bond_matrix_h_heavy.device)      
        
        x_original = x.clone()
        
        
        if len(edge_mask_h_heavy.shape) < 4:
            edge_mask = torch.cat((edge_mask_h_h, edge_mask_h_heavy.view(atom_features.shape[0], -1)), dim=-1)
        else:
            edge_mask = torch.cat((edge_mask_h_h.unsqueeze(1).repeat(1, 
                edge_mask_h_heavy.shape[1], 1), edge_mask_h_heavy.view(atom_features.shape[0],
                edge_mask_h_heavy.shape[1], -1)), dim=-1)
                                                                       
        if len(edge_mask.shape) < 3:
            
            edge_mask = edge_mask.unsqueeze(1)
            
                       
        for index, layer in enumerate(self.layers):

            
            differences, distances = cdist(x[..., :num_h, :], x)
            
            edge_features = torch.cat([torch.stack([atom_features[i, :, edge_row[i], :] for i in range(edge_row.shape[0])]), 
                         torch.stack([atom_features[i, :, edge_col[i], :] for i in range(edge_row.shape[0])]),
                         torch.stack([distances[i, :, edge_row[i], edge_col[i]] for i in range(edge_row.shape[0])]).unsqueeze(-1),
                         torch.stack([bond_matrix.unsqueeze(1).repeat(1, distances.shape[1], 1, 1)[i, :, edge_row[i], edge_col[i]] for i in range(edge_row.shape[0])]).unsqueeze(-1)], dim=-1)
            
            differences = torch.stack([differences[i, :, edge_row[i], edge_col[i], :]\
                                       for i in range(differences.shape[0])])
            
            x_out, h_out = layer(t, atom_features, differences, edge_features, 
                                 edge_row.unsqueeze(1), edge_col, edge_mask, atom_mask)

            x = torch.cat((x[..., :num_h, :] + x_out[..., :num_h, :], x[..., num_h:, :]), dim=2)
        
            atom_features = torch.cat((h_out, atom_features[..., num_h:, :]), dim=2)
            
        prediction = x[..., :num_h, :] - x_original[..., :num_h, :]
        
        prediction = prediction * atom_mask[:, None, :, :]
 
        

        return prediction
  
    def get_score_fn(self, features, sde):
        
        
        def score_hydrogen(perturbed_data, t):
            
            _, std = sde.marginal_prob(torch.zeros(perturbed_data.shape).to(
                features["fragment_seq"].device), t)
               
            prediction = self(t*999, perturbed_data, features)
           
            prediction = -1*prediction/std[:, None, None, None].to(prediction.device)
       
            return prediction


        return score_hydrogen
