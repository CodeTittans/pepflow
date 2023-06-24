import torch.nn as nn
import torch
import math


def _hyper_fan_in(tensor, dim_in, dim_in_main, predict_bias):
    
    bound = math.sqrt(1/(2**(predict_bias)*dim_in*dim_in_main))
    
    tensor.uniform_(-bound, bound)
    
    return tensor

def _hyper_fan_in_bias(tensor, dim_in):
    
    bound = math.sqrt(1/(2*dim_in))
    
    tensor.uniform_(-bound, bound)
    
    return tensor
    
    
class HyperLinear(nn.Module):
    
    '''
    a linear layer initialized with hyperfan-in (Chang et al. 2020)
    for use in the hypernetwork
    '''
    
    def __init__(self, dim_in, dim_in_main, dim_out_main, bias=False, 
                 predict_bias=False):
        
        super(HyperLinear, self).__init__()
        
        if not bias:
            
            self.W = torch.zeros(dim_in, dim_in_main*dim_out_main)
            
            self.b = torch.zeros(dim_in_main*dim_out_main)
        
        else:
            
            self.W = torch.zeros(dim_in, dim_out_main)
            
            self.b = torch.zeros(dim_out_main)
            
            
        
        if not bias:
            
            self.W = _hyper_fan_in(self.W, dim_in, dim_in_main, predict_bias)
            
        else:
            
            self.W = _hyper_fan_in_bias(self.W, dim_in)
        
        
        self.W = nn.Parameter(self.W)
        self.b = nn.Parameter(self.b)
        
    def forward(self, x):
        
        x = torch.matmul(x, self.W) + self.b
        return x
        
