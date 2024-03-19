from pepflow.model.layers.attention_layers import MultiHeadAttention, FFN
from pepflow.model.layers.hyperlinear_layers import HyperLinear
import torch.nn as nn
import torch

class ResNetBlock(nn.Module):
    
    def __init__(self, in_dim, out_dim, kernel, padding):
        
        super(ResNetBlock, self).__init__()
        
        self.in_dim = in_dim
        
        self.out_dim = out_dim
        
        
        if self.in_dim != self.out_dim:
            self.CNN1 = nn.Conv1d(in_dim, out_dim, kernel, padding=padding)
        
            self.layernorm1 = nn.LayerNorm(out_dim)
        
        
        self.CNN2 = nn.Conv1d(out_dim, out_dim, kernel, padding=padding)
        
        self.layernorm2 = nn.LayerNorm(out_dim)
        
        self.relu2 = nn.ReLU()
        
        
        self.CNN3 = nn.Conv1d(out_dim, out_dim, kernel, padding=padding)
        
        self.layernorm3 = nn.LayerNorm(out_dim)
        
        self.relu3 = nn.ReLU()
        
    def forward(self, x):
        
        if self.in_dim != self.out_dim:
            
            x = self.CNN1(x)
            
            x = self.layernorm1(x.transpose(-1, -2)).transpose(-1, -2)
            
        res = x
        
        x = self.CNN2(x)
        
        x = self.layernorm2(x.transpose(-1, -2)).transpose(-1, -2)
        
        x = self.relu2(x)
        
        x = self.CNN3(x)
        
        x = self.layernorm3(x.transpose(-1, -2)).transpose(-1, -2)
        
        x = x + res
        
        x = self.relu3(x)
        
        return x
        
class AttentionBlock(nn.Module):
    
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout):
        
        super().__init__()
        
        self.self_attention = MultiHeadAttention(n_head, d_model,
                                                 d_k, d_v, dropout)
        
        
        self.ffn = FFN(d_model, d_inner, dropout)
        
        
    def forward(self, decoder_input):
      
        decoder_output, decoder_self_attention = self.self_attention(decoder_input, 
                                                                     decoder_input,
                                                                     decoder_input)
        
      

        decoder_output = self.ffn(decoder_output)
        
        return decoder_output, decoder_self_attention
            
        

class HyperResNet(nn.Module):
    
    def __init__(self, dims, repeats, kernel,
                 hidden_dims_output, activation_fn_output, mainnet_config,
                 padding):
        
        super(HyperResNet, self).__init__()
        

        self.embedding = nn.Embedding(21, dims[0])
        
        self.blocks = nn.ModuleList()
        
        self.repeats = repeats
        
        
        for index, dim in enumerate(dims[:-1]):
            for i in range(repeats):
                 self.blocks.append(ResNetBlock(dim, dim, kernel, padding))
                 
            self.blocks.append(ResNetBlock(dim, dims[index+1], kernel, padding))
            
         
            
            
        w_x_layers = []
        b_x_layers = []
        w_t_layers = []
        w_b_t_layers = []
        
        for layer in mainnet_config:
            w_x_layer = nn.Sequential(nn.Linear(dims[-1], hidden_dims_output[0]),
                                      nn.ReLU())
            
            b_x_layer = nn.Sequential(nn.Linear(dims[-1], hidden_dims_output[0]),
                                      nn.ReLU())
            w_t_layer = nn.Sequential(nn.Linear(dims[-1], hidden_dims_output[0]),
                                      nn.ReLU())
            
            b_t_layer = nn.Sequential(nn.Linear(dims[-1], hidden_dims_output[0]),
                                      nn.ReLU())
            
            for index, dim_output in enumerate(hidden_dims_output[1:]):
                
                w_x_layer.add_module("layer_" + str(index), nn.Linear(hidden_dims_output[index],
                                               dim_output))
                
                w_x_layer.add_module("relu_" + str(index), nn.ReLU())
                
                b_x_layer.add_module("layer_" + str(index), nn.Linear(hidden_dims_output[index],
                                               dim_output))
                
                b_x_layer.add_module("relu_" + str(index), nn.ReLU())
                
                w_t_layer.add_module("layer_" + str(index), nn.Linear(hidden_dims_output[index],
                                               dim_output))
                
                w_t_layer.add_module("relu_" + str(index), nn.ReLU())
                
                b_t_layer.add_module("layer_" + str(index), nn.Linear(hidden_dims_output[index],
                                               dim_output))
                
                b_t_layer.add_module("relu_" + str(index), nn.ReLU())
                
            w_x_layer.add_module("layer_final", HyperLinear(hidden_dims_output[-1], layer[1], layer[2], bias=False,
                                          predict_bias=True))
            
            b_x_layer.add_module("layer_final", HyperLinear(hidden_dims_output[-1], layer[1], layer[2], bias=True,
                                          predict_bias=True))
            
            w_t_layer.add_module("layer_final", HyperLinear(hidden_dims_output[-1], 1, layer[2], bias=False,
                                          predict_bias=True))
            
            b_t_layer.add_module("layer_final", HyperLinear(hidden_dims_output[-1], 1, layer[2], bias=False,
                                          predict_bias=True))
 
            
            w_x_layers.append(w_x_layer)
            
            b_x_layers.append(b_x_layer)
            
            w_t_layers.append(w_t_layer)
            
            w_b_t_layers.append(b_t_layer)
            
        self.w_x_layers = nn.ModuleList(w_x_layers)
        self.b_x_layers = nn.ModuleList(b_x_layers)
        self.w_t_layers = nn.ModuleList(w_t_layers)
        self.w_b_t_layers = nn.ModuleList(w_b_t_layers)
        
        
        self.mainnet_config = mainnet_config
        
    def forward(self, sequence, return_embedding=False):
        
        sequence = self.embedding(sequence)
        
        sequence  = sequence.transpose(-1,-2)
        
       
        for index, block in enumerate(self.blocks):
            
            sequence = block(sequence)
        
        sequence = sequence.transpose(-1,-2)
        
        sequence = torch.mean(sequence, dim=1)
        
        w_x_outputs = []
        b_x_outputs = []
        w_t_outputs = []
        w_b_t_outputs = []
        
        batch_size = sequence.shape[0]
        
                
        for index, layer in enumerate(self.w_x_layers):
            w_x_outputs.append(layer(sequence).reshape(batch_size, 
                                                       self.mainnet_config[index][1], 
                                                       self.mainnet_config[index][2]))
        
        for index, layer in enumerate(self.b_x_layers):
            b_x_outputs.append(layer(sequence).reshape(batch_size, 
                                                       self.mainnet_config[index][2]))
            
        for index, layer in enumerate(self.w_t_layers):
            w_t_outputs.append(layer(sequence).reshape(batch_size, 1, 
                                                       self.mainnet_config[index][2]))
            
        for index, layer in enumerate(self.w_b_t_layers):
            w_b_t_outputs.append(layer(sequence).reshape(batch_size, 1, 
                                                       self.mainnet_config[index][2]))
        
        
      
        
        if return_embedding:
            return w_x_outputs, b_x_outputs, w_t_outputs, w_b_t_outputs, sequence
        else:
            
            return w_x_outputs, b_x_outputs, w_t_outputs, w_b_t_outputs 
        
class HyperBERT(nn.Module):
    
    def __init__(self, repeats, d_model, d_inner, n_head, d_k, d_v, 
                 dropout, hidden_dims_output, activation_fn_output, 
                 mainnet_config):
        
        super(HyperBERT, self).__init__()

        # self.embedding = nn.Embedding(22, d_model)
        self.embedding = nn.Embedding(25, d_model)
        
        self.positional_embedding = nn.Embedding(16, d_model)
        
        self.blocks = nn.ModuleList()
        
        
        self.repeats = repeats
        
     
        
        for i in range(repeats):
            self.blocks.append(AttentionBlock(d_model, d_inner, n_head, d_k, d_v, dropout))
                 
         
         
            
            
        w_x_layers = []
        b_x_layers = []
        w_t_layers = []
        w_b_t_layers = []
        
        for layer in mainnet_config:
            w_x_layer = nn.Sequential(nn.Linear(d_model, hidden_dims_output[0]),
                                      nn.ReLU())
            
            b_x_layer = nn.Sequential(nn.Linear(d_model, hidden_dims_output[0]),
                                      nn.ReLU())
            w_t_layer = nn.Sequential(nn.Linear(d_model, hidden_dims_output[0]),
                                      nn.ReLU())
            
            b_t_layer = nn.Sequential(nn.Linear(d_model, hidden_dims_output[0]),
                                      nn.ReLU())
            
            
            for index, dim_output in enumerate(hidden_dims_output[1:]):
                
                w_x_layer.add_module("layer_" + str(index), nn.Linear(hidden_dims_output[index],
                                               dim_output))
                
                w_x_layer.add_module("relu_" + str(index), nn.ReLU())
                
                b_x_layer.add_module("layer_" + str(index), nn.Linear(hidden_dims_output[index],
                                               dim_output))
                
                b_x_layer.add_module("relu_" + str(index), nn.ReLU())
                
                w_t_layer.add_module("layer_" + str(index), nn.Linear(hidden_dims_output[index],
                                               dim_output))
                
                w_t_layer.add_module("relu_" + str(index), nn.ReLU())
                
                b_t_layer.add_module("layer_" + str(index), nn.Linear(hidden_dims_output[index],
                                               dim_output))
                
                b_t_layer.add_module("relu_" + str(index), nn.ReLU())
                
            w_x_layer.add_module("layer_final", HyperLinear(hidden_dims_output[-1], layer[1], layer[2], bias=False,
                                          predict_bias=True))
            
            b_x_layer.add_module("layer_final", HyperLinear(hidden_dims_output[-1], layer[1], layer[2], bias=True,
                                          predict_bias=True))
            
            w_t_layer.add_module("layer_final", HyperLinear(hidden_dims_output[-1], 1, layer[2], bias=False,
                                          predict_bias=True))
            
            b_t_layer.add_module("layer_final", HyperLinear(hidden_dims_output[-1], 1, layer[2], bias=False,
                                          predict_bias=True))
            
            
            w_x_layers.append(w_x_layer)
            
            b_x_layers.append(b_x_layer)
            
            w_t_layers.append(w_t_layer)
            
            w_b_t_layers.append(b_t_layer)
            
        self.w_x_layers = nn.ModuleList(w_x_layers)
        self.b_x_layers = nn.ModuleList(b_x_layers)
        self.w_t_layers = nn.ModuleList(w_t_layers)
        self.w_b_t_layers = nn.ModuleList(w_b_t_layers)
        
        self.mainnet_config = mainnet_config
   


    def process_tensors(self, sequence):

        cls_token = torch.ones_like(sequence[..., :1]).to(sequence.device) * 24 # 21
                        
        sequence = torch.cat([cls_token, sequence], -1)
        
        positions = torch.arange(0, sequence.shape[1]).unsqueeze(0).to(sequence.device)
        
        return sequence, positions

    def forward(self, sequence, positions, return_embedding=False):
       
        #sequence = torch.randint_like(sequence, 0, 20).to(sequence.device)
        
           
        sequence = self.embedding(sequence)
        
        positional_embedding = self.positional_embedding(positions)

        sequence = sequence + positional_embedding
        
        for index, block in enumerate(self.blocks):
            sequence, attention = block(sequence)
        
        sequence = sequence[:, 0, :]
        
        w_x_outputs = []
        b_x_outputs = []
        w_t_outputs = []
        w_b_t_outputs = []
        
        batch_size = sequence.shape[0]
                
        for index, layer in enumerate(self.w_x_layers):
            w_x_outputs.append(layer(sequence).reshape(batch_size, 
                                                       self.mainnet_config[index][1], 
                                                       self.mainnet_config[index][2]))
        
        for index, layer in enumerate(self.b_x_layers):
            b_x_outputs.append(layer(sequence).reshape(batch_size, 
                                                       self.mainnet_config[index][2]))
            
        for index, layer in enumerate(self.w_t_layers):
            w_t_outputs.append(layer(sequence).reshape(batch_size, 1, 
                                                       self.mainnet_config[index][2]))
            
        for index, layer in enumerate(self.w_b_t_layers):
            w_b_t_outputs.append(layer(sequence).reshape(batch_size, 1, 
                                                       self.mainnet_config[index][2]))
        
            
        if return_embedding:
            return w_x_outputs, b_x_outputs, w_t_outputs, w_b_t_outputs, sequence
        else:
            
            return w_x_outputs, b_x_outputs, w_t_outputs, w_b_t_outputs 
        
