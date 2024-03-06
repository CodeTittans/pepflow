from pepflow.model.sde_lib import cmsubVPSDE, subVPSDE
import ml_collections as mlc

HIDDEN_SIZE_R = 128

HIDDEN_SIZE_H = 64

HIDDEN_SIZE_B = 256


gnn_concat_squash_layers_rotamer = [["concat_squash", 78*2 + 2, 1],
                                      ["concat_squash", 158, 2 * HIDDEN_SIZE_R ],
                                      ["concat_squash", 2 *
                                          HIDDEN_SIZE_R , HIDDEN_SIZE_R ],
                                      ["concat_squash", HIDDEN_SIZE_R , HIDDEN_SIZE_R ],
                                      ["concat_squash", HIDDEN_SIZE_R , 1],
                                      ["concat_squash", HIDDEN_SIZE_R , HIDDEN_SIZE_R ],
                                      ["concat_squash", HIDDEN_SIZE_R  +
                                          78, HIDDEN_SIZE_R ],
                                      ["concat_squash", HIDDEN_SIZE_R , 78], ]

gnn_concat_squash_layers_hydrogen = [["concat_squash", 79*2 + 2, 1],
                                      ["concat_squash", 160, 2 * HIDDEN_SIZE_H],
                                      ["concat_squash", 2 *
                                          HIDDEN_SIZE_H, HIDDEN_SIZE_H],
                                      ["concat_squash", HIDDEN_SIZE_H, HIDDEN_SIZE_H],
                                      ["concat_squash", HIDDEN_SIZE_H, 1],
                                      ["concat_squash", HIDDEN_SIZE_H, HIDDEN_SIZE_H],
                                      ["concat_squash", HIDDEN_SIZE_H +
                                          79, HIDDEN_SIZE_H],
                                      ["concat_squash", HIDDEN_SIZE_H, 79], ]


gnn_concat_squash_layers_backbone = [["concat_squash", 40*2 + 2, 1],
                            ["concat_squash", 82, 2 * HIDDEN_SIZE_B],
                            ["concat_squash", 2 * HIDDEN_SIZE_B, HIDDEN_SIZE_B],
                            ["concat_squash", HIDDEN_SIZE_B, HIDDEN_SIZE_B],
                            ["concat_squash", HIDDEN_SIZE_B, 1],
                            ["concat_squash", HIDDEN_SIZE_B, HIDDEN_SIZE_B],
                            ["concat_squash", HIDDEN_SIZE_B + 40, HIDDEN_SIZE_B],
                            ["concat_squash", HIDDEN_SIZE_B, 40], ]

config_training_by_energy = mlc.ConfigDict(
    {
     "lr":1e-5,
     "batch_size":1,
     "batch_size_md":16,
     "num_samples":8,
     "gradient_clip":100,
     }
)


config_backbone_bert = mlc.ConfigDict(
    {
        "training":
        {
            "lr": 0.0001,
            "batch_size":8,
            "ema":0.9999,
            "gradient_clip":1
        },
        "finetuning":
        {
            "lr": 0.0001,
            "batch_size":32,
            "ema":0.9999,
            "gradient_clip":1
        },

        "finetuning_md":
        {
            "lr": 0.0001,
            "batch_size":8,
            "ema":0.9999,
            "gradient_clip":1
        },
        "hypernetwork_config":
        {
            "repeats":3,
            "d_model":64, 
            "d_inner":64,
            "n_head":12, 
            "d_k":128, 
            "d_v":128, 
            "dropout":0.3,
            "hidden_dims_output": [64],
            "activation_fn_output": "relu",
            "mainnet_config": [gnn_concat_squash_layer for i in range(4)\
                               for gnn_concat_squash_layer in gnn_concat_squash_layers_backbone]
        },
        "mainnet_config": {
            "dims": [dim for j in range(4) for dim in [[i[1], i[2]] for i in gnn_concat_squash_layers_backbone]],
            "num_layers": 4,
            "dims_trainable": [dim for j in range(4) for dim in [[i[1], i[2]] for i in gnn_concat_squash_layers_backbone]],
            "num_layers_trainable":4,
        },
        "hyperbert": True,
        "sde_config":{
            "sde":subVPSDE,
            "beta_min":0.1,
            "beta_max":20,
            "eps":1e-5
        },
        "sampling":{
            "method":'ode',
            "rtol":1e-4,
            "atol":1e-4,
            "noise_removal":False,
            "probability_flow":True,
            "training":{
            "continuous":True
            }
        }
        
    }
)

config_backbone = mlc.ConfigDict(
    {
        "training":
        {
            "lr": 0.0001,
            "batch_size":8,
            "ema":0.9999
        },
        "hypernetwork_config":
        {
            "dims": [128, 256],
            "repeats": 2,
            "kernel": 3,
            "padding":1,
            "hidden_dims_output": [64],
            "activation_fn_output": "relu",
            "mainnet_config": [gnn_concat_squash_layer for i in range(6)\
                               for gnn_concat_squash_layer in gnn_concat_squash_layers_backbone]
        },
        "mainnet_config": {
            "dims": [dim for j in range(6) for dim in [[i[1], i[2]] for i in gnn_concat_squash_layers_backbone]],
            "num_layers": 6,
            "dims_trainable": [dim for j in range(2) for dim in [[i[1], i[2]] for i in gnn_concat_squash_layers_backbone]],
            "num_layers_trainable":2,
        },
        "hyperbert": False,
        "sde_config":{
            "sde":subVPSDE,
            "beta_min":0.1,
            "beta_max":20,
            "eps":1e-5
        },
        "sampling":{
            "method":'ode',
            "rtol":1e-4,
            "atol":1e-4,
            "noise_removal":False,
            "probability_flow":True,
            "training":{
            "continuous":True
            }
        }
        
    }
)

config_rotamer = mlc.ConfigDict(
    {
        "training":{
            "lr":0.0005,
            "batch_size":8,
            "ema":0.9999,
            "gradient_clip":1
        },
        "model_config":{
            "dims": [dim for j in range(4) for dim in [[i[1], i[2]] for i in gnn_concat_squash_layers_rotamer]],
            "num_layers": 4
        },
        "sde_config":{
            "sde":cmsubVPSDE,
            "beta_min":0.1,
            "beta_max":20,
            "eps":1e-5
        },
        "sampling":{
            "method":'ode',
            "rtol":1e-4,
            "atol":1e-4,
            "noise_removal":False,
            "probability_flow":True,
            "training":{
            "continuous":True
            }
        }

    }
)
    
config_hydrogen = mlc.ConfigDict(
    {
        "training":{
            "lr":0.0005,
            "batch_size":4,
            "ema":0.9999,
            "gradient_clip":None
        },
        "model_config":{
            "dims": [dim for j in range(4) for dim in [[i[1], i[2]] for i in gnn_concat_squash_layers_hydrogen]],
            "num_layers": 4
        },
        "sde_config":{
            "sde":subVPSDE,
            "beta_min":0.01,
            "beta_max":20,
            "eps":1e-5
        },
        "sampling":{
            "method":'ode',
            "rtol":1e-4,
            "atol":1e-4,
            "noise_removal":False,
            "probability_flow":True,
            "training":{
            "continuous":True
            }
        }

    }
)
    



    

    


    
    
    
