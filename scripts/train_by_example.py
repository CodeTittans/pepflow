from pepflow.model.model import BackboneModel
from pepflow.model.dynamics import DynamicsRotamer, DynamicsHydrogen
from pepflow.model.model_configs import config_rotamer, config_hydrogen,\
    config_backbone_bert, config_backbone
from pepflow.model.ema import ExponentialMovingAverage
from pepflow.data.dataset import FragmentDatasetRotamer, FragmentDatasetHydrogen,\
    FragmentDatasetBackbone, MDDataset, get_weighted_sampler
from pepflow.utils.dataset_utils import collate, collate_multiple_coords
from pepflow.utils.training_utils import sample_centered_noise,\
    sample_noise, dsm
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



def train(model, model_type, epochs, output_file, batch_size, lr, sde, ema_decay, 
          gradient_clip = None, eps=1e-5, saved_params=None, data_path="./", weighted=False,
          dataset_type="fragment"):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if not saved_params == None:

        optimizer.load_state_dict(torch.load(saved_params)["optimizer_state_dict"])
        
    if ema_decay != None:
        
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        
        if saved_params != None:
            
            ema.load_state_dict(torch.load(saved_params)["ema_state_dict"])
    
    # ----
    dataset = MDDataset(mode="train", 
                        data_path=data_path, # aim to take the data_path from argparse.
                        data_dir=data_path, 
                        model=model_type) 
       
    validation_dataset = MDDataset(mode="val", model=model_type)
    
    collate_fn = collate_multiple_coords        
    
    if weighted:
        
        sampler = get_weighted_sampler("train")
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             num_workers=4, collate_fn=collate_fn,
                                             sampler=sampler)
        
        sampler = get_weighted_sampler("val")
        
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, 
                                             num_workers=4, collate_fn=collate_fn,
                                             sampler=sampler)
        
    else:
        batch_size = 8 # @@@hard-coded!!!!
        print(f"DBG: hardcoded batch size = {batch_size}")
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             num_workers=4, collate_fn=collate_fn,
                                             shuffle=True)

        # NOTE: Validation is off @30.12.2023       
        # validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, 
        #                                      num_workers=4, collate_fn=collate_fn,
        #                                      shuffle=True)

    start_time = time()

    log_step = 100
    
    validation_losses = []

    iters = len(loader)
    
#    assert iters == round(len(dataset) / batch_size), f"{iters} != {round(len(dataset) / batch_size)}, which is not expected."
    print(f"DBG: iter {iters} == {round(len(dataset) / batch_size)}")

    for e in range(epochs):

        model.train()
        losses = []
        
        
        for value, features in enumerate(loader):
            #print(f"DBG 126: value(iter)={value}, features.keys={features.keys()}")
            if features == None:
                continue

            torch.cuda.empty_cache()
          
            
            optimizer.zero_grad()
            if model_type == "rotamer":
                z, t, perturbed_data, mean, std = sample_centered_noise(sde, features["coordinates_rotamer"], 
                                                             features["atom_mask_rotamer"], device="cuda",
                                                             indices=features["amino_acid_pos_rotamer"].long() - 1)
            elif model_type == "protonation":
                z, t, perturbed_data, mean, std = sample_noise(sde, features["coordinates_h"], 
                                                               features["atom_mask_h"],
                                                               device="cuda")
            elif model_type == "backbone":
                z, t, perturbed_data, mean, std = sample_noise(sde, features["coordinates_backbone"], 
                                                               features["atom_mask_backbone"],
                                                               device="cuda")
            

            score_fn = model.get_score_fn(to_cuda(features), sde)
                    
    
                
            prediction = score_fn(perturbed_data, t)

            if model_type == "rotamer":
                all_losses, loss = dsm(prediction, std, z,
                                       features["atom_mask_rotamer"]) 
            elif model_type == "protonation":
                all_losses, loss = dsm(prediction, std, z,
                        features["atom_mask_h"]) 
            elif model_type == "backbone":
                all_losses, loss = dsm(prediction, std, z,
                        features["atom_mask_backbone"]) 
            
            
            for index, i in enumerate(all_losses):
            
                losses.append(torch.sum(i).cpu().detach().numpy())
            
      
            loss.backward()
            
            if gradient_clip != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
        
            if ema_decay != None:
                
                ema.update(model.parameters()) 

            if (value+1) % log_step == 0 or value == iters - 1:

                elapsed = time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))

                log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                    elapsed, e+1, epochs, value + 1, iters)
                log += ", {}: {:.5f}".format('Loss', np.mean(losses))
                log += ", {}: {:.5f}".format('Std', np.std(losses))
                
                print(log)
                sys.stdout.flush()

                losses = []
                
            continue # to inactivate validation loss calculation
        
            # @@@ Validation
            if (value+1) % 10000 == 0 or (value == iters - 1):
                model.eval()
                
                losses = []
                
                
                if ema_decay != None:
                    
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    
                
                for value_val, features in enumerate(validation_loader):
                    if features == None:
                        continue

                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        if value_val >= 1000:
                            break
                        if model_type == "rotamer":
                            z, t, perturbed_data, mean, std = sample_centered_noise(sde, features["coordinates_rotamer"], 
                                                                         features["atom_mask_rotamer"], device="cuda",
                                                                         indices=features["amino_acid_pos_rotamer"].long() - 1)
                        elif model_type == "protonation":
                            z, t, perturbed_data, mean, std = sample_noise(sde, features["coordinates_h"], 
                                                                           features["atom_mask_h"],
                                                                           device="cuda")
                        elif model_type == "backbone":
                            z, t, perturbed_data, mean, std = sample_noise(sde, features["coordinates_backbone"], 
                                                                           features["atom_mask_backbone"], 
                                                                           device="cuda")
                        score_fn = model.get_score_fn(to_cuda(features), sde)
                        
                        prediction = score_fn(perturbed_data, t)
            
                        if model_type == "rotamer":
                            all_losses, loss = dsm(prediction, std, z,
                                                   features["atom_mask_rotamer"]) 
                        elif model_type == "protonation":
                            all_losses, loss = dsm(prediction, std, z,
                                    features["atom_mask_h"]) 
                        elif model_type == "backbone":
                            all_losses, loss = dsm(prediction, std, z,
                                    features["atom_mask_backbone"]) 
                            
                        
                        for index, i in enumerate(all_losses):
                            losses.append(torch.sum(i).detach())
                        
            
                if ema_decay != None:
                    ema.restore(model.parameters())
                    
                losses = torch.stack(losses, dim=0)
                
                losses = losses.cpu().numpy()

                validation_losses.append(np.mean(losses))

                log = "This is validation, Epoch [{}/{}]".format(
                    e + 1, epochs)

                log += ", {}: {:.5f}".format('Loss', np.mean(losses))
                log += ", {}: {:.5f}".format('Std', np.std(losses))
            
                if validation_losses[-1] == min(validation_losses):
                    

                    param_dict = {"model_state_dict": model.state_dict(),
                            "optimizer_state_dict":optimizer.state_dict()
                            }
                    
                        
                    
                    if ema_decay != None:
                        
                        param_dict["ema_state_dict"] = ema.state_dict()
                        
                    print("Saving model with new minimum validation loss")
                    torch.save(param_dict, output_file)
                    print("Saved model successfully!")

                
                print(log)
                
                if (value+1) % 1000 == 0:
                    if not os.path.isdir("checkpoints_" + model_type):
                        os.mkdir("checkpoints_" + model_type)
                        
                   
                    torch.save({"model_state_dict": model.state_dict(),
                                "optimizer_state_dict":optimizer.state_dict()
                                }, "checkpoints_" + model_type +"/" +\
                               output_file.replace(".pth", "_" + str(value+1) + ".pth"))
                        
                    
                    
                losses = []
              
                sys.stdout.flush()
                model.train()

        # @@@ Save model parameters at each epoch
        param_dict = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict":optimizer.state_dict()
                     }                             
        if ema_decay != None:
            param_dict["ema_state_dict"] = ema.state_dict()
        print(f"Saving model at epoch {e}")
        torch.save(param_dict, output_file)
        print("Saved model successfully!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-sv", dest="use_saved", help="Flag for whether or not to use a saved model",
                        required=False, default=False, action='store_true')
    parser.add_argument("-sm", dest="saved_model", help="File containing saved params",
                        required=False, type=str, default="Saved.pth")
    parser.add_argument("-o", dest="output_file",
                        help="File for output of model parameters", required=True, type=str)
    parser.add_argument("-d", dest="data_path",
                        help="Directory where data is stored", required=False, 
                        type=str, default="./")
    parser.add_argument("-ep", dest="epochs", help="Number of epochs",
                        required=False, type=int, default=10)
    parser.add_argument("-m", dest="model", type=str, 
                        help="Model to train, backbone, rotamer or protonation",
                        required=False, default="rotamer")
    parser.add_argument("-ht", dest="hypernetwork_type", type=str, 
                        help="Type of hypernetwork to use, either bert or resnet",
                        required=False, default=None)
    parser.add_argument("-dt", dest="dataset", type=str, 
                        help="Dataset to train on, fragment or MD",
                        required=False, default="fragment")
    parser.add_argument("--w", dest="weighted", required=False,
                        action="store_true")


    
    args = parser.parse_args()

    if args.model != "backbone" and args.model != "protonation" and args.model != "rotamer":
        raise ValueError("Model argument is invalid, must be one of backbone, rotamer or protonation")
        
        
    if args.model == "rotamer":
        config = config_rotamer
        model = DynamicsRotamer(config.model_config)
    elif args.model == "protonation":
        config = config_hydrogen
        model = DynamicsHydrogen(config.model_config)
    elif args.model == "backbone":
        
        if args.hypernetwork_type == "bert":
            config = config_backbone_bert
        elif args.hypernetwork_type == "resnet":
            config = config_backbone
        else:
           raise ValueError("Hypernetwork type argument is invalid, must be one of bert or resnet")

        model = BackboneModel(config)

        print("The number of parameters in the hypernetwork is %i" 
              %(sum(p.numel()for p in model.hypernetwork.parameters() if p.requires_grad)))   
  
        # NOTE: Original implementation does not go through this if-statement even if args.dataset == 'MD', which is expected in the line 75-ish 
        if args.dataset.lower() == "md":   
        #if args.dataset == "md": 
            config.training = config.finetuning_md
            print("Batch size of config.training: ", config.training.batch_size)
    sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
                                beta_max=config.sde_config.beta_max)
    
    torch.cuda.empty_cache()

    if args.use_saved:
        model.load_state_dict(torch.load(args.saved_model)["model_state_dict"])
    

    model.cuda()
    
    # Shinji Added: 
    print("GPU usage: ",  torch.cuda.memory_allocated() / (1024 * 1024), "MiB")

    if args.use_saved:
        train(model, args.model, args.epochs, args.output_file, 
              config.training.batch_size, config.training.lr, sde, 
              ema_decay=config.training.ema, gradient_clip=config.training.gradient_clip,
              saved_params=args.saved_model, data_path=args.data_path, weighted=args.weighted,
              dataset_type=args.dataset)
    else:
        print("DBG: Batch size: ", config.training.batch_size)
        print("DBG: data_path: ", args.data_path)
        train(model, args.model, args.epochs, args.output_file, config.training.batch_size,
              config.training.lr, sde, ema_decay=config.training.ema, gradient_clip=config.training.gradient_clip,
              data_path=args.data_path, weighted=args.weighted, dataset_type=args.dataset)
