####################################################################################################
#                                             test.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/12/24                                                                                #
#                                                                                                  #
# Purpose: Testing of models with different data.                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutup; shutup.please()   # shut up warnings
import torch

from tqdm import tqdm

# own
from frameworks.framework import Framework
from utils.gpu_config import set_gpu_usage
from utils.structures import Map



#**************************************************************************************************#
#                                            Class Test                                            #
#**************************************************************************************************#
#                                                                                                  #
# The main class for testing the models.                                                           #
#                                                                                                  #
#**************************************************************************************************#
class Test():

        #*************************#
        #   initialize instance   #
        #*************************#
        def __init__(self, config):
            self.config = Map(config)  # make config mappable
            self.system = self.getSysModel(self.config)
            self.model = self.getModel(self.config)  # get the model
            self.dataloader = self.getDataLoader(self.config)  # get the data loader


        #****************#
        #   model init   #
        #****************#
        def getModel(self, config):
            config = Map(config)  # make config mappable
            if config.model == 'lcm':
                if config.method.lower() == 'newton' or config.method.lower() == 'mh':
                    from frameworks.frameworkFSL import FrameworkFSL
                    model = FrameworkFSL(**config)
                elif config.method.lower() == 'lcmodel':
                    from frameworks.frameworkLCM import FrameworkLCModel
                    model = FrameworkLCModel(**config)
                else:
                    raise ValueError('Method not recognized.')
            else:
                if config.model == 'nn':
                    from frameworks.frameworkNN import FrameworkNN
                    model = FrameworkNN(**config)
                elif config.model == 'ls':
                    from frameworks.frameworkLS import FrameworkLS
                    model = FrameworkLS(**config)
                elif config.model == 'gd':
                    from frameworks.frameworkGD import FrameworkGD
                    model = FrameworkGD(**config)
                elif config.model.lower() == 'snf':
                    from frameworks.frameworkSNF import FrameworkSNF
                    model = FrameworkSNF(**config)
                elif config.model.lower() == 'wand':
                    from frameworks.frameworkWAND import FrameworkWAND
                    model = FrameworkWAND(**config)
                else:
                    raise ValueError('Model %s is not recognized' % config.model)

                if config.get('slurm', False):
                    # auto GPU setting can select GPU outside of the SLURM allocation,
                    # so we set the GPU manually to 0
                    gpu = torch.device(0 if torch.cuda.is_available() else 'cpu')
                else:
                    gpu = torch.device(set_gpu_usage() if torch.cuda.is_available() else 'cpu')

                if config.load_model and config.checkpoint_path != '':
                    model = type(model).load_from_checkpoint(**config, map_location=gpu)
                else:
                    model.to(gpu)

                if config.state and config.state == 'eval':
                    model.eval()
                elif config.state and config.state == 'train':
                    model.train()

            return model


        #***********************#
        #   system model init   #
        #***********************#
        def getSysModel(self, config):
            return Framework(path2basis=config.path2basis, basisFmt=config.basisFmt,
                             specType=config.specType, dataType=config.dataType,
                             ppmlim=config.ppmlim)


        #**********************#
        #   data loader init   #
        #**********************#
        def getDataLoader(self, config, sigModel=None, params=None, concs=None):
            if sigModel is None: sigModel = self.system.sigModel
            if params is None: params = self.system.ps
            if concs is None: concs = self.system.concs

            if config.dataType[:3] == 'cha':
                from simulation.dataModules import ChallengeDataModule
                dataloader = ChallengeDataModule(basis_dir=config.path2basis,
                                                 nums_cha=config.test_size)

            else:
                from simulation.dataModules import SynthDataModule
                dataloader = SynthDataModule(basis_dir=config.path2basis,
                                             nums_test=config.test_size,
                                             sigModel=sigModel,
                                             params=params,
                                             concs=concs,
                                             basisFmt=config.basisFmt,
                                             specType=config.specType)
            return dataloader


        #***********************#
        #   run model on data   #
        #***********************#
        def run(self, model=None, data=None):
            if model is None: model = self.model
            if data is None: data = self.dataloader.test_dataloader()

            truths, preds = [], []
            specs, specs_sep = [], []
            with torch.no_grad():
                for x, y, t in tqdm(data):
                    specs.append(x)
                    specs_sep.append(y)
                    t_hat = model.forward(x.to(model.device) if hasattr(model, 'device') else x)
                    truths.append(t)
                    if isinstance(t_hat, tuple): t_hat = t_hat[0]   # only concentrations
                    if not isinstance(t_hat, torch.Tensor): t_hat = torch.Tensor(t_hat)
                    if not hasattr(model, 'basisObj'):
                        t_hat[:, :self.system.basisObj.n_metabs] = torch.stack([t_hat[:, model.basisFSL.names.index(m)]   # sort
                                                                                for m in self.system.basisObj.names], dim=1)
                    preds.append(t_hat)

            return torch.cat(truths), torch.cat(preds), torch.cat(specs), torch.cat(specs_sep)


        #*******************************#
        #   run model no ground truth   #
        #*******************************#
        def run_noGT(self, model=None, data=None):
            if model is None: model = self.model
            if data is None: data = self.dataloader.test_dataloader()

            specs, preds = [], []
            with torch.no_grad():
                for x in tqdm(data):
                    specs.append(x)
                    t_hat = model.forward(x.to(model.device) if hasattr(model, 'device') else x)
                    if isinstance(t_hat, tuple): t_hat = t_hat[0]   # only concentrations
                    if not isinstance(t_hat, torch.Tensor): t_hat = torch.Tensor(t_hat)
                    if not hasattr(model, 'basisObj'):
                        t_hat[:, :self.system.basisObj.n_metabs] = torch.stack([t_hat[:, model.basisFSL.names.index(m)]   # sort
                                                                                for m in self.system.basisObj.names], dim=1)
                    preds.append(t_hat)

            return torch.cat(preds), torch.cat(specs)



#*************#
#   testing   #
#*************#
if __name__ == '__main__':

    # initialize the configuration
    config = {
        # path to a trained model
        'checkpoint_path': './wandb/run-20250803_162007-r2wqdtwd/files/checkpoints/epoch=0-step=1867776.ckpt',

        # path to basis set
        # 'path2basis': '../Data/DataSets/2016_MRS_fitting_challenge/basis_sets/basisset_LCModelBASIS/press3T_30ms.BASIS',
        # 'path2basis': '../Data/BasisSets/FOCI_slaser_basis_MMshift_Insshift.basis',
        'path2basis': '../Data/BasisSets/7T_sLASER_OIT_TE34.basis',

        # path to data
        'path2data': '../Data/DataSets/7tExample/PreS_spectro_act.SDAT',
        # 'control': '../Data/Other/BrainBeats.control',
        'control': '../Data/Other/Synth.control',

        # model
        'model': 'gd',  # 'nn', 'ls', 'lcm', ...
        'loss': 'mse_specs',  # 'mae', 'mse', ...

        # setup
        'specType': 'auto',  # 'auto', 'synth', 'invivo', 'fMRSinPain', 'all'
        'ppmlim': (0.5, 4.0),  # ppm limits for the spectra (used if specType='auto')
        'basisFmt': '7tslaser',  # '', 'cha_philips', 'fMRSinPain', 'biggaba', '7tslaser', ...
        'dataType': 'aumc2_ms',  # 'clean', 'std', 'std_rw', 'std_rw_p', 'custom', ...
        'test_size': 1000,  # number of test samples

        # for nn model
        'arch': 'mlp',  # 'mlp', 'cnn' ...
        'weight_init': None,  # weight initialization mode ('xavier', 'kaiming', 'orthogonal', 'constant')
        'norm': True,  # normalize the input spectra (and scale gts if forwardNorm = False)
        'forwardNorm': True,  # push the norm forward (i.e. scale the predictions with the norm)
        'activation': 'elu',  # 'relu', 'elu', 'tanh', 'sigmoid', 'softplus', ...
        'dropout': 0.0,
        'width': 512,  # (for mlp)
        'depth': 3,
        'conv_depth': 3,  # (for cnn)
        'kernel_size': 3,
        'stride': 1,

        # for ls model
        'optimizer': 'scipy',   # 'fsl', 'scipy', ...
        'initMethod': 'nn',  # 'nn', 'rand', 'fsl', 'none'
        'stepSize': 0.1,  # step size for the optimizer
        'maxIter': 200,  # max number of iterations for the optimizer

        # for gd model
        'adaptMode': 'per_spec',  # 'model_only', 'full_ft', 'per_spec'
        'innerEpochs': 100,  # number of inner epochs
        'innerBatch': 1,  # batch size for the inner loop
        'innerLr': 1e-4,  # learning rate for the inner loop
        'innerLoss': 'mse_specs',  # loss function for the inner loop
        'bnState': 'train',  # 'train', 'eval' - batch normalization state for the inner loop
        'initMode': 'nn',  # 'nn', 'rand', 'fsl', ...

        # for lcm model
        'method': 'Newton',  # 'Newton', 'MH', 'LCModel'
        'bandwidth': 3000,  # bandwidth of the spectra
        'sample_points': 1024,  # number of sample points
        'include_params': True,  # include signal parameters in the output (only for FSL-MRS)
        'save_path': '', #'./testLCM/',  # path to save the results (empty for no saving)

        # mode
        'load_model': True,  # load model from path2trained
        'skip_train': True,  # skip the training procedure

        # visual settings
        'run': True,  # run the inference (will try to load results if False)
        'save': True,  # save the plots
        'error': 'msmae',  # 'mae', 'mse', 'mape', ...
    }

    # run a quick test
    test = Test(config)

    if test.config.dataType != 'invivo':
        truths, preds, _, _ = test.run()

        # exclude macromolecules
        idx = [i for i, name in enumerate(test.system.basisObj.names)
               if 'mm' in name.lower() or 'mac' in name.lower()]
        truths[:, idx] = 0
        preds[:, idx] = 0

        err = test.system.concsLoss(truths, preds.to(truths.device), type=test.config.error)
        print(f'{test.config.model} {test.config.method if test.config.model == "lcm" else ""} '
              f'error: {err.mean().item()}')

    else:   # run in-vivo test

        import numpy as np
        from loading.loadData import loadDataAsFSL

        # load and prepare the data
        data = loadDataAsFSL(config['path2data'], fmt='philips')

        x = np.fft.fft(data.FID)
        x = np.stack([x.real, x.imag])
        x = torch.tensor(x).unsqueeze(0).float()

        # run the model
        with torch.no_grad():
            test.model.save_path = test.config.save_path
            t_hat = test.model.forward(x)