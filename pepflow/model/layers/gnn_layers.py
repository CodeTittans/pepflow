from pepflow.model.layers.mainnet_layers import MLPConcatSquash
from pepflow.model.layers.trainable_layers import MLPConcatSquashTrainable, MLP
from torch_scatter.composite.softmax import scatter_softmax
from torch_scatter import scatter_add
import torch.nn as nn
import torch


class GNNLayer(nn.Module):

    def __init__(self):

        super(GNNLayer, self).__init__()

        self.MLP = MLPConcatSquash(num_layers=2, final_activation="softplus")

        self.gate = MLPConcatSquash(num_layers=1)

        self.PhiX = MLPConcatSquash(num_layers=2)

        self.PhiInf = MLPConcatSquash(num_layers=1, final_activation="sigmoid")

        self.PhiH = MLPConcatSquash(num_layers=2)

    def forward(self, t, atom_features, differences, edge_features,
                edge_row, edge_col, edge_mask, atom_mask, w_x, b_x, w_t, w_b_t):

        out = edge_features*edge_mask[..., None]
        
        gate = self.gate(t, out, w_x[:1], b_x[:1], w_t[:1], w_b_t[:1])
        
        gate = edge_mask.unsqueeze(-1).float().bool()*gate + (
            torch.logical_not(edge_mask.unsqueeze(-1)))*(-1*1e20)
        gate = scatter_softmax(gate, edge_row, 2)
        out = self.MLP(t, out, w_x[1:3], b_x[1:3], w_t[1:3], w_b_t[1:3]) * \
            edge_mask[..., None]
            
        
        x_out = self.PhiX(t, out, w_x[3:5], b_x[3:5],
                          w_t[3:5], w_b_t[3:5])*gate
        x_out = x_out*(differences/(torch.sqrt(torch.sum(torch.square(differences)+1e-12, 
            dim=-1))[..., None]+1))
        x_out_full = scatter_add(x_out, edge_row, 2)*atom_mask[:, None, :, :]
        
        
        m_out = self.PhiInf(t, out, w_x[5:6], b_x[5:6], w_t[5:6], w_b_t[5:6])
        m_out = m_out*out
        m_out_full = scatter_add(
            m_out*edge_mask[..., None], 
            edge_row, 2, None)
        
        h_out = self.PhiH(t, torch.cat([atom_features, m_out_full], dim=-1),
                  w_x[6:8], b_x[6:8], w_t[6:8], w_b_t[6:8]) + atom_features 
        
        return x_out_full, h_out
    
                
class GNNLayerTrainable(nn.Module):

    def __init__(self, dims):

        super(GNNLayerTrainable, self).__init__()

        self.gate = MLPConcatSquashTrainable(dims[:1])

        self.MLP = MLPConcatSquashTrainable(
            dims[1:3], final_activation="softplus")

        self.PhiX = MLPConcatSquashTrainable(dims[3:5])

        self.PhiInf = MLPConcatSquashTrainable(
            dims[5:6], final_activation="sigmoid")

        self.PhiH = MLPConcatSquashTrainable(dims[6:8])
        

    def forward(self, t, atom_features, differences, edge_features,
                edge_row, edge_col, edge_mask, atom_mask):
        
        out = edge_features*edge_mask[..., None]
        
        gate = self.gate(t, out)
        gate = edge_mask.unsqueeze(-1).float().bool()*gate + (
            torch.logical_not(edge_mask.unsqueeze(-1)))*(-1*1e20)
        gate = scatter_softmax(gate, edge_row, 2)
        out = self.MLP(t, out) * \
            edge_mask[..., None]
        
        x_out = self.PhiX(t, out)*gate
        x_out = x_out*(differences/(torch.sqrt(torch.sum(torch.square(differences)+1e-12, 
            dim=-1))[..., None]+1))
        x_out_full = scatter_add(x_out, edge_row, 2)*atom_mask[:, None, :, :]

        m_out = self.PhiInf(t, out)

        m_out = m_out*out

        m_out_full = scatter_add(
            m_out*edge_mask[..., None], 
            edge_row, 2, None)
        
        h_out = self.PhiH(t, torch.cat(
            [atom_features[..., :m_out_full.shape[-2], :], m_out_full], dim=-1))\
                    + atom_features[..., :m_out_full.shape[-2], :]

        return x_out_full, h_out

