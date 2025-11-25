####################################################################################################
#                                             train.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/12/22                                                                                #
#                                                                                                  #
# Purpose: Train and evaluate neural models implemented in Pytorch Lightning.                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import numpy as np
import os
import pytorch_lightning as pl
import shutup; shutup.please()   # shut up warnings
import torch
import wandb

from pytorch_lightning.callbacks import (ModelCheckpoint, EarlyStopping, LearningRateMonitor,
                                         RichProgressBar)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import WandbLogger

# own
from utils.gpu_config import set_gpu_usage



#**************************************************************************************************#
#                                             Pipeline                                             #
#**************************************************************************************************#
#                                                                                                  #
# The pipeline allowing to load, augment, train, and test methods.                                 #
#                                                                                                  #
#**************************************************************************************************#
class Pipeline():

    #***************#
    #   main init   #
    #***************#
    def __init__(self):
        pl.seed_everything(42)

        self.default_config = {
            'checkpoint_path': '',   # path to a trained model
            'path2vae': '',   # path to a trained VAE model
            'path2basis': '../Data/BasisSets/7T_sLASER_OIT_TE34.basis',  # path to basis set

            'model': 'nn',   # 'nn', 'gd',
            'loss': 'mae',  # 'mae', 'mse', ...

            # setupq
            'specType': 'auto',  # 'auto', 'synth', 'invivo', 'fMRSinPain', 'all'
            'ppmlim': (0.5, 4.0),  # ppm limits for the spectra (used if specType='auto')
            'basisFmt': '7tslaser',  # '', 'cha_philips', 'fMRSinPain', 'biggaba', '7tslaser', 'mrsi', ...
            'dataType': 'aumc2',  # 'clean', 'norm', 'std', 'std_rw', 'std_rw_p', 'custom', ...

            # for nn models
            'arch': 'mlp',   # 'mlp', 'cnn' ...
            'activation': 'elu',   # 'relu', 'elu', 'tanh', 'sigmoid', 'softplus', ...
            'dropout': 0.0,
            'width': 512,
            'depth': 3,
            'conv_depth': 3,
            'kernel_size': 3,
            'stride': 1,

            # for gd model
            'adaptMode': 'per_spec',  # 'model_only', 'full_ft', 'per_spec'
            'innerEpochs': 100,   # number of epochs for the inner loop
            'innerLr': 1e-3,   # learning rate for the inner loop
            'innerLoss': 'mse_specs',   # loss function for the inner loop (e.g. 'mse_specs', 'mae_conc', ...)

            # training
            'val_size': 1024,   # number of validation samples
                                # (simulated new validation set each val_check_interval)

            'max_epochs': -1,  # limit number of epochs (not possible with generator)
            'max_steps': 10000,  # limit number of steps/batches (useful with generator)

            'val_check_interval': 256,   # None, if fixed training set
            'check_val_every_n_epoch': None,   # None, if trained with generator
            'trueBatch': 16,   # accumulates the gradients over trueBatch/batch
            'batch': 16,
            'lr': 1e-4,  # learning rate
            'reg_l1': 0.0,  # L1 regularization
            'reg_l2': 0.0,   # L2 regularization
            'callback': 'val_loss',  # loss to monitor for callbacks
            'gradient_clip_val': 0.0,   # gradient clipping (0.0 = no clipping)

            # hardware
            'num_workers': 4,   # adjust to roughly 4 times the number GPUs
            'shuffle': True,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'gpu_selection': set_gpu_usage() if torch.cuda.is_available() else 0,

            # mode
            'load_model': False,   # load model from path2trained
            'skip_train': False,   # skip the training procedure
        }

        # limit number of threads if run is CPU heavy
        torch.set_num_threads(self.default_config['num_workers'])

        # set device
        if str(self.default_config['device']) == 'cuda':
            self.default_config['device'] = torch.device(self.default_config['gpu_selection'])


    #**********************************#
    #   switch model based on config   #
    #**********************************#
    def getModel(self, config):
        if config.model.lower() == 'nn':
            from frameworks.frameworkNN import FrameworkNN
            model = FrameworkNN(**config)
            model.to(config.device)

        elif config.model.lower() == 'gd':
            from frameworks.frameworkGD import FrameworkGD
            model = FrameworkGD(**config)
            model.to(config.device)

        else:
            raise ValueError('model %s is not recognized' % config.model)
        return model


    #********************************#
    #   main pipeline for training   #
    #********************************#
    def main(self, config=None):
        # wandb init
        if (config is None) or config['online']:
            wandb.init(config=config)
            wandb_logger = WandbLogger(save_dir='./logs/')
        else:
            wandb.init(mode='disabled', config=config)
            wandb_logger = WandbLogger(save_dir='./logs/', offline=True)
        config = wandb.config

        # combine default configs and wandb config
        parser = argparse.ArgumentParser()
        for keys in self.default_config:
            parser.add_argument('--' + keys, default=self.default_config[keys],
                                type=type(self.default_config[keys]))
        args = parser.parse_known_args()[0]
        config.update(args, allow_val_change=False)

        # model inits
        self.model = self.getModel(config)

        # callbacks, etc.
        checkpoint_cb = ModelCheckpoint(monitor=config.callback, save_last=True,
                                        dirpath=os.path.join(wandb.run.dir, 'checkpoints'))
        early_stop_cb = EarlyStopping(monitor=config.callback, mode='min',
                                      min_delta=0.0, patience=10)
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # loading model...
        if config.load_model:
            self.model = type(self.model).load_from_checkpoint(**config, map_location=config.device)

        # ...train model
        if not config.skip_train:
            if torch.cuda.is_available():  # gpu acceleration
                try: torch.set_float32_matmul_precision('medium')    # matrix multiplications
                except: print('bfloat16 for matmul not supported')   # use the bfloat16

                trainer = pl.Trainer(max_epochs=config.max_epochs,
                                     max_steps=config.max_steps,
                                     accelerator='gpu',
                                     devices=[config.gpu_selection],  # select gpu by idx
                                     logger=wandb_logger,
                                     callbacks=[checkpoint_cb, lr_monitor, self.custom_progress_bar()],  # , early_stop_cb],
                                     log_every_n_steps=10,
                                     accumulate_grad_batches=config.trueBatch // config.batch,
                                     val_check_interval=config.val_check_interval,
                                     gradient_clip_val=config.gradient_clip_val,
                                     gradient_clip_algorithm="value"
                                     )
            else:
                trainer = pl.Trainer(max_epochs=config.max_epochs,
                                     max_steps=config.max_steps,
                                     logger=wandb_logger,
                                     callbacks=[checkpoint_cb, lr_monitor, self.custom_progress_bar()],  # , early_stop_cb],
                                     log_every_n_steps=10,
                                     accumulate_grad_batches=config.trueBatch // config.batch,
                                     val_check_interval=config.val_check_interval,
                                     gradient_clip_val=config.gradient_clip_val,
                                     gradient_clip_algorithm="value"
                                     )

            trainer.fit(self.model)

            # loading best model from callback
            self.model = type(self.model).load_from_checkpoint(checkpoint_cb.best_model_path,
                                                               **config)

        wandb.finish()
        self.config = config


    #*************************#
    #   custom progress bar   #
    #*************************#
    def custom_progress_bar(self):
        return RichProgressBar(
            theme=RichProgressBarTheme(
                progress_bar="cyan",  # main progress bar color
                progress_bar_finished="green",  # completed bar color
                progress_bar_pulse="magenta",  # pulsing effect for active progress
                batch_progress="bright_yellow",  # batch progress text color
                time="bright_blue",  # elapsed and remaining time color
                processing_speed="bright_magenta",  # speed color for better visibility
                metrics="bright_white",  # metrics color for clarity
                metrics_text_delimiter="\n",  # ensures clear separation of metrics
            )
        )


#**********#
#   main   #
#**********#
if __name__ == '__main__':
    Pipeline().main({'online': False})

