import torch
import torch.nn as nn
import torch.nn.functional as F


NONLINEARITIES = {
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid()
}

class ConcatSquashLinear(nn.Module):
    
    def __init__(self):
        super(ConcatSquashLinear, self).__init__()
 
    def forward(self, t, x, w_x, b_x, w_t, w_b_t):
        
        x = torch.matmul(x, w_x[:, None, :, :]) + b_x[:, None, None, :]
        
        
        t = t.view(x.shape[0], 1, 1)

        x = x*torch.sigmoid(torch.matmul(t, w_t).squeeze(1))[:, None, None, :]\
            + torch.tanh(torch.matmul(t, w_b_t)[:, None, :, :])
        
        return x
    
    
class ConcatLinear(nn.Module):
    
    def __init__(self):
        super(ConcatLinear, self).__init__()
 
    def forward(self, x, t, w_x, b_x):
        
        tt = torch.ones_like(x[..., :1]) * t
        
        ttx = torch.cat([tt, x], -1)
        
        x = F.linear(ttx, w_x.transpose(-1, -2), b_x)
        
        return x
    
class MLPConcatSquash(nn.Module):
    
    def __init__(self, num_layers, nonlinearity="softplus", final_activation=None):
        
        super(MLPConcatSquash, self).__init__()
 
        self.num_layers = num_layers
        
        layers = []
        
        activations = []
        
        
        for i in range(num_layers):
            
            layers.append(ConcatSquashLinear())
            
            if i < num_layers - 1:
                
                activations.append(NONLINEARITIES[nonlinearity])
              
        self.layers = nn.ModuleList(layers)
        
        self.activations = nn.ModuleList(activations)
        
        self.final_activation = final_activation
        
        if not final_activation is None:
            self.final_activation_layer = NONLINEARITIES[final_activation]

    def forward(self, t, x, w_x, b_x, w_t, w_b_t):
        for i in range(self.num_layers):
            
            x = self.layers[i](t, x, w_x[i], b_x[i], w_t[i], w_b_t[i])
            if i < self.num_layers - 1:
                
                x = self.activations[i](x)
        
        if self.final_activation != None:
            
            x = self.final_activation_layer(x)
            
        return x
                
