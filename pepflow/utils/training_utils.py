from torch_scatter import scatter, scatter_add
from pepflow.utils.tensor_utils import normal_prob
import torch


def sample_noise(sde, x, atom_mask, device, eps=1e-5):
    
    t = torch.rand(x.shape[0]) * (sde.T - eps) + eps

    z = torch.randn_like(x)
      
    mean, std = sde.marginal_prob(x, t )
    
    mean = torch.FloatTensor(mean)
    std = torch.FloatTensor(std)
    
    perturbed_data = mean + std[:, None, None, None] * z
    
    return z.to(device), t.to(device),\
        perturbed_data.to(device),\
            mean.to(device), std.to(device) 
    
def sample_centered_noise(sde, x, atom_mask, device, indices=None, eps=1e-5):
    
    t = torch.rand(x.shape[0]) * (sde.T - eps) + eps
    
    z = torch.randn_like(x) * atom_mask[:, None, :, :]
    
    if indices == None:
        
        centroids = torch.sum(z, 
                              dim=2)[:, :, None, :]/torch.sum(atom_mask, dim=1)[:, None, None, :]
        
    else:
        
        centroids = scatter(src=z, dim=2, 
                                 index=indices.unsqueeze(1),
                                          reduce="sum")
        centroids = centroids/(scatter(src=atom_mask[:, None, :, :], dim=2,
                                         index=indices.unsqueeze(1), reduce="sum") +1e-8)
        centroids = torch.stack([centroids[i][:, indices[i], :]\
                                  for i in range(len(indices))])
            
        z = (z - centroids) * atom_mask[:, None, :, :]
     
    mean, std = sde.marginal_prob(x, t, atom_mask, indices)
            
    perturbed_data = (mean + std[:, None, None, None] * z)*atom_mask[:, None, :, :] +\
        (1-atom_mask[:, None, :, :])*x
    
    return z.to(device), t.to(device),\
        perturbed_data.to(device),\
            mean.to(device), std.to(device)

def sample_noise_with_perturbations(sde, x, permuted_coords, atom_mask,  
                                    device, permutation_indices=None, clamp=1, eps=1e-5):
    
    if permuted_coords == None:
        
        permuted_coords = []
        
        for index, coords in enumerate(x):
            permuted_coords.append(torch.stack([coords[:, i] for i in permutation_indices[index]],
                                               dim=0))
            
        permuted_coords = torch.stack(permuted_coords, dim=0)
    
    t = torch.rand(x.shape[0], 
                   device=x.device) * (sde.T - eps) + eps
    
    z = torch.randn_like(x) * atom_mask[:, None, :, :]
    z = z * atom_mask[:, None, :, :]
    
    mean, std = sde.marginal_prob(x, t)
    
    p_z  = normal_prob(mean, std[:, None, None], z, 
                        atom_mask[:, None, :, :])
    
    perturbed_data = (mean + std[:, None, None, None] * z)*atom_mask[:, None, :, :] 
    
    means_permuted = []
    for index in range(permuted_coords.shape[1]):
        
        mean_permuted, std_permuted = sde.marginal_prob(permuted_coords[:, index, :, :, :],
                                        t)
        
        means_permuted.append(mean_permuted)
        
    means_permuted = torch.stack(means_permuted, dim=1)
    permuted_noise = (perturbed_data.unsqueeze(1).repeat(1, 6, 1, 1, 1)\
                                          - means_permuted)/std[:, None, None, None, None]\
                                          *atom_mask[:, None, None, :, :]
    p_z_permuted = normal_prob(means_permuted, std[:, None, None, None], permuted_noise,
                               atom_mask[:, None, None, :, :])
    p_ratio = torch.clamp(torch.exp(p_z_permuted - p_z[:, None, :]), max=clamp)
    
    return z.to(device), t.to(device),\
        perturbed_data.to(device),\
            mean.to(device), std.to(device),\
                permuted_noise.to(device), p_ratio.to(device)
                

            
def dsm(prediction, std, z, atom_mask):
    
    all_losses = torch.square(prediction * std[:, None, None, None] + z) *\
                atom_mask[:, None, :, :]
                

    loss = torch.mean(torch.sum(all_losses, dim=(-1, -2, -3))) 
    
    return all_losses, loss

def dsm_permuted(prediction, std, permuted_noise, p_ratio, bound_atom, atom_mask,
                 weights=None, sde=None, t=None, likelihood_weighting=False):
    
    if not likelihood_weighting:
        all_permuted_losses = torch.square(prediction.unsqueeze(1).repeat(1, 6, 1, 1, 1)\
                                           * std[:, None, None, None, None].cuda()\
                                               + permuted_noise.cuda())*atom_mask[:, None, None, :, :]
    else:
        g2 = sde.sde(torch.zeros_like(prediction), t)[1] ** 2
        all_permuted_losses = torch.square(prediction.unsqueeze(1).repeat(1, 6, 1, 1, 1)\
                + permuted_noise.cuda() / std[:, None, None, None, None])*atom_mask[:, None, None, :, :] * g2[:, None, None, None, None]
            
        
    all_losses = []
    
    all_losses_unweighted = []
    
    for index in range(len(all_permuted_losses)):
        aggregated_loss = scatter_add(all_permuted_losses[index],
                      bound_atom[index].unsqueeze(0)\
                          .unsqueeze(0).unsqueeze(-1).repeat(6,1,1,3), dim=-2)
        aggregated_loss = torch.sum(aggregated_loss, dim=-1)
        min_indices = torch.argmin(aggregated_loss, dim=0, keepdim=True)
        min_loss = all_permuted_losses[index]*p_ratio[index][..., None].cuda()
        
        if weights != None:
            min_loss = min_loss * weights[index][None, :, None, None]
            
        min_loss = scatter_add(min_loss,
                               bound_atom[index].unsqueeze(0)\
                                   .unsqueeze(0).unsqueeze(-1).repeat(6,1,1,3), dim=-2)
        min_loss = torch.sum(min_loss, dim=-1)
        min_loss = torch.gather(min_loss, dim=0, 
                index=min_indices)
        
        all_losses.append(torch.sum(min_loss))    
        if weights != None:
            aggregated_loss = aggregated_loss * weights[index][None, :, None]
        all_losses_unweighted.append(torch.sum(torch.gather(aggregated_loss, dim=0, 
            index=min_indices)))

    
    loss = torch.mean(torch.stack(all_losses))  
    
    return all_losses_unweighted, loss
