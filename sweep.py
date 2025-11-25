####################################################################################################
#                                            sweep.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 23/01/23                                                                                #
#                                                                                                  #
# Purpose: Sweep parameter definition for Weights & Biases.                                        #
#                                                                                                  #
####################################################################################################

if __name__ == '__main__':

    #*************#
    #   imports   #
    #*************#
    import pytorch_lightning as pl
    import wandb

    # own
    from train import Pipeline


    #**************************#
    #   eliminate randomness   #
    #**************************#
    pl.seed_everything(42)


    #************************************#
    #   configure sweep and parameters   #
    #************************************#
    sweep_config = {'method': 'grid'}

    metric = {
        'name': 'val_conc_loss',
        'goal': 'minimize',

        'additional_metrics': {
            'name': 'test_loss',
            'goal': 'minimize'
        }
    }

    sweep_parameters = {
        'skip_train': {'values': [False]},
        'online': {'values': [True]},

        # DD fitting
        'project': {'values': ['OoD-Robust-MRS-Quantification']},
        'model': {'values': ['nn']},   # 'nn', 'ls', 'gd'
        'loss': {'values': ['mse_specs']},  # 'mae_conc', 'mse_specs', 'mae_all_scale', ...
        'ppmlim': {'values': [(0.5, 4.0)]},  # ppm limits for the spectra (used if specType='auto')
        'basisFmt': {'values': ['7tslaser']},  # '', 'cha_philips', 'fMRSinPain', 'biggaba', '7tslaser', 'mrsi', ...
        'dataType': {'values': ['aumc2']},  # 'clean', 'norm', 'std', 'std_rw', 'std_rw_p', 'custom', ...
        'arch': {'values': ['cnn']},   # 'mlp', 'cnn'
        'lr': {'values': [1e-4]},  # learning rate
        'width': {'values': [512]},
        'depth': {'values': [3]},
        'conv_depth': {'values': [3]},
        'kernel_size': {'values': [3]},
        'stride': {'values': [1]},
        'activation': {'values': ['elu']},   # 'relu', 'elu', 'tanh', ...
        'dropout': {'values': [0.0]},
        'optimizer': {'values': ['theseus']},
        'initMethod': {'values': ['nn']},  # 'nn', 'rand', 'fsl'
        'stepSize': {'values': [0.1]},
        'maxIter': {'values': [10]},
        'adaptMode': {'values': ['per_spec']},  # 'model_only', 'full_ft', 'per_spec'
        'innerEpochs': {'values': [100]},  # number of epochs for the inner loop
        'innerLr': {'values': [1e-3]},  # learning rate for the inner loop
        'innerLoss': {'values': ['mse_specs']},  # loss function for the inner loop
        'callback': {'values': ['val_conc_loss']},
        'load_model': {'values': [False]},  # load model from path2trained
        'max_epochs': {'values': [-1]},
        'max_steps': {'values': [-1]},

    }

    sweep_config['name'] = 'model_sweep'   # sweep name
    sweep_config['parameters'] = sweep_parameters   # add parameters to sweep
    sweep_config['metric']= metric    # add metric to sweep

    # create sweep ID and name project
    wandb.login(key='')   # add your own key here
    sweep_id = wandb.sweep(sweep_config,
                           project=sweep_parameters['project']['values'][0],  # project name
                           entity='')   # your own entity
    # training the model
    pipeline = Pipeline()
    wandb.agent(sweep_id, pipeline.main)
