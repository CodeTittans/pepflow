from pepflow.model.model import BackboneModel
from pepflow.model.dynamics import DynamicsRotamer, DynamicsHydrogen
from pepflow.model.ema import ExponentialMovingAverage
from pepflow.model.model_configs import config_rotamer, config_hydrogen,\
    config_backbone_bert, config_training_by_energy 
from pepflow.data.dataset import SequenceDataset, MDDataset
from pepflow.utils.dataset_utils import collate, collate_multiple_coords
from pepflow.utils.training_utils import sample_noise, dsm
from time import time
import datetime
import torch
import sys
import argparse
import numpy as np
import os

torch.autograd.set_detect_anomaly(True)



def to_cuda(features):
    
    for feature in features:
        
        if "atom_names" not in feature and "cyclic_bond_indices" not in feature:
            
            features[feature] = features[feature].cuda()
            
    
    return features



def train(model, epochs, output_file, batch_size, batch_size_md, num_samples, lr, backbone_sde, rotamer_sde, hydrogen_sde,
          training_list, val_list, eps=1e-5, gradient_clip=None, optimizer_params=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if optimizer_params != None:
        optimizer.load_state_dict(torch.load(optimizer_params)["optimizer_state_dict"])
   
    dataset = SequenceDataset(data_path=training_list)

    validation_dataset = SequenceDataset(data_path=val_list)
            
  

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         num_workers=0, collate_fn=collate,
                                         shuffle=True)
    
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, 
                                         num_workers=0, collate_fn=collate,
                                         shuffle=True)


    dataset = MDDataset(mode="train", model="backbone", num_repeats=1e6, 
                                   no_mod=True, sample_mds=1)

    validation_dataset = MDDataset(mode="val", model="backbone", num_repeats=1e6, 
                                   no_mod=True, sample_mds=1)
    
    loader_md = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size_md, 
                                         num_workers=0, collate_fn=collate_multiple_coords,
                                         shuffle=True))
    
    validation_loader_md = iter(torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size_md, 
                                         num_workers=0, collate_fn=collate_multiple_coords,
                                         shuffle=True))
    
    start_time = time()

    log_step = 1
    
    validation_losses = []

    iters = len(loader)

    for e in range(epochs):

        model.eval()
        
        losses = []
        
        losses_dsm = []
        
        for value, features in enumerate(loader):

            

            optimizer.zero_grad()

            if features == None:
                continue
    

            
            try:
                energies, delta_log_p = model.sample_all_atoms_differentiable(to_cuda(features),
                                                                          backbone_sde,
                                                                          rotamer_sde,
                                                                          hydrogen_sde,
                                                                          num_samples=num_samples, differentiable_protonation=True )
            except:
                print("Generation failed")

                continue
            

            all_losses = energies - delta_log_p

            for index, i in enumerate(all_losses):
            
                losses.append(torch.mean(i).cpu().detach().numpy())
           
            
            loss = 0.5*torch.sum(torch.mean(all_losses, dim=1), dim=0)
            
                
            loss.backward()

            energies  = energies.cpu().detach().numpy()
            
            delta_log_p = delta_log_p.cpu().detach().numpy()

            if gradient_clip != None:

                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            torch.cuda.empty_cache()
      
            
            
            
            
            features_md = next(loader_md)
            
            z, t, perturbed_data, mean, std = sample_noise(backbone_sde, features_md["coordinates_backbone"], 
                                                                           features_md["atom_mask_backbone"],
                                                                           device="cuda")
            
            score_fn = model.get_score_fn(to_cuda(features_md), backbone_sde)
                    
    
                
            prediction = score_fn(perturbed_data, t)

            
            
            all_losses, loss = dsm(prediction, std, z,
                        features["atom_mask_backbone"]) 

            for index, i in enumerate(all_losses):
                losses_dsm.append(torch.sum(i).cpu().detach().numpy())

            loss = 0.5*loss

            loss.backward()
            
            
            for feature in features_md:

                if "names" not in feature:

                    features_md[feature] = features_md[feature].cpu().detach().numpy()


            loss = loss.cpu().detach().numpy()
            

            torch.cuda.empty_cache()
            
            optimizer.step()
        
            torch.cuda.empty_cache()

            if (value+1) % log_step == 0 or value == iters - 1:

                elapsed = time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))

                log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                    elapsed, e+1, epochs, value + 1, iters)
                log += ", {}: {:.5f}".format('Loss', np.mean(losses))
                log += ", {}: {:.5f}".format('DSM', np.mean(losses_dsm))
                
                print(log)
                sys.stdout.flush()

                losses = []
                losses_dsm = []
                
                        

            if (value + 1)  % 100 == 0:
                model.eval()
                
                losses = []
                losses_dsm = []
                
                
           
                    
                
                for value_val, features in enumerate(validation_loader):

                    if value_val >= 10:
                        break

                    if features == None:
                        continue

 
                    torch.cuda.empty_cache()
                    
                    with torch.no_grad():
                        
                        features_md = next(validation_loader_md)
                        
                        z, t, perturbed_data, mean, std = sample_noise(backbone_sde, features_md["coordinates_backbone"], 
                                                                                       features_md["atom_mask_backbone"],
                                                                                       device="cuda")
                        
                        score_fn = model.get_score_fn(to_cuda(features_md), backbone_sde)                            
                
                        prediction = score_fn(perturbed_data, t)
            
                        
                        
                        all_losses, loss = dsm(prediction, std, z,
                                    features_md["atom_mask_backbone"]) 
            
                        for index, i in enumerate(all_losses):
                            losses_dsm.append(torch.sum(i).cpu().detach().numpy())
            
                  
                

                        for feature in features_md:

                            if "names" not in feature:

                                features_md[feature] = features_md[feature].cpu().detach().numpy()


                        torch.cuda.empty_cache()
                        try:
                            energies, delta_log_p = model.sample_all_atoms_differentiable(to_cuda(features),
                                                                                      backbone_sde,
                                                                                      rotamer_sde,
                                                                                      hydrogen_sde,
                                                                                      num_samples=num_samples, differentiable_protonation=True)
                        
                        except:
                            print("Generation failed")
                            continue

            
                       
                       
                        all_losses = energies - delta_log_p
                        
                        for index, i in enumerate(all_losses):
                        
                            losses.append(torch.mean(i).cpu().detach().numpy())
                        
                        
            
                    
                validation_losses.append(np.mean(losses))

                log = "This is validation, Epoch [{}/{}]".format(
                    e + 1, epochs)

                log += ", {}: {:.5f}".format('Loss', np.mean(losses))
                log += ", {}: {:.5f}".format('DSM', np.mean(losses_dsm))
            
                if validation_losses[-1] == min(validation_losses):
                    

                    param_dict = {"model_state_dict": model.state_dict(),
                            "optimizer_state_dict":optimizer.state_dict()
                            }
                    
                        
                    
                 
                        
                    print("Saving model with new minimum validation loss")
                    torch.save(param_dict, output_file)
                    print("Saved model successfully!")

                
                print(log)
                
                
                if not os.path.isdir("checkpoints_energy"):
                    os.mkdir("checkpoints_energy")
                    
               
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict":optimizer.state_dict()
                            }, "checkpoints_energy/" +\
                           output_file.replace(".pth", "_" + str(value+1) + ".pth"))
                    
                    
                    
                losses = []
                
                losses_dsm = []
              
                sys.stdout.flush()


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
    parser.add_argument("-tl", dest="training_list",
                        help="Training list", required=False, type=str, default="dataset_split_skewed/all_training_seqs.npy")
    parser.add_argument("-vl", dest="validation_list",
                        help="Val list", required=False, type=str, default="dataset_split_skewed/all_validation_seqs.npy")
    parser.add_argument("-ep", dest="epochs", help="Number of epochs",
                        required=False, type=int, default=1)
    parser.add_argument("-o", dest="output_file",
                        help="Output file where params will be saved", required=True, type=str)
       

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
        
        
        
        config = config_rotamer
        rotamer_dynamics = DynamicsRotamer(config.model_config)
        
        rotamer_sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
                                beta_max=config.sde_config.beta_max,
                                temperature=0.4)
        
        if config.training.ema != None and "ema_state_dict" in torch.load(args.rotamer_model):
            ema = ExponentialMovingAverage(rotamer_dynamics.parameters(), decay=config.training.ema)
            ema.load_state_dict(torch.load(args.rotamer_model)["ema_state_dict"])
            
            ema.copy_to(rotamer_dynamics.parameters())
        
        else:
            
            rotamer_dynamics.load_state_dict(torch.load(args.rotamer_model)["model_state_dict"])
        
        
    
        config = config_hydrogen
        hydrogen_dynamics = DynamicsHydrogen(config.model_config)
        
        hydrogen_sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
                                beta_max=config.sde_config.beta_max,
                                temperature=1.0)
        
        if config.training.ema != None and "ema_state_dict" in torch.load(args.hydrogen_model):
            ema = ExponentialMovingAverage(hydrogen_dynamics.parameters(), decay=config.training.ema)
            ema.load_state_dict(torch.load(args.hydrogen_model)["ema_state_dict"])
            
            ema.copy_to(hydrogen_dynamics.parameters())
        
        else:
            
            hydrogen_dynamics.load_state_dict(torch.load(args.hydrogen_model)["model_state_dict"])
        
        
        model.hydrogen_dynamics = hydrogen_dynamics
        
        model.rotamer_dynamics = rotamer_dynamics
        
        torch.cuda.empty_cache()
        model.cuda()
        
    else:

        config = config_backbone_bert
        
        backbone_sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
                                    beta_max=config.sde_config.beta_max,
                                    temperature=1.0, mask_first_atom=True)
        
        
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

    train(model, args.epochs, args.output_file, config_training_by_energy.batch_size,
          config_training_by_energy.batch_size_md, config_training_by_energy.num_samples,
          config_training_by_energy.lr, backbone_sde, rotamer_sde, hydrogen_sde, args.training_list, 
          args.validation_list, gradient_clip=config_training_by_energy.gradient_clip, optimizer_params=args.full_model)
