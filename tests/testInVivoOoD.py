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

import json
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shutup; shutup.please()   # shut up warnings
import time
import torch

from scipy.stats import binned_statistic

from tqdm import tqdm

# own
from frameworks.frameworkFSL import FrameworkFSL
from frameworks.frameworkLCM import FrameworkLCModel
from modelDefs import get_models, get_colors
from simulation.simulationDefs import aumcConcs, aumcConcsMS
from other.brainbeats import BrainBeatsDataModule
from test import Test


#*************#
#   testing   #
#*************#
if __name__ == '__main__':

    # initialize the configuration
    config = {
        'slurm': True,  # use SLURM for running the tests

        # path to a trained model
        'checkpoint_path': '',

        # path to basis set
        'path2basis': '../Data/BasisSets/7T_sLASER_OIT_TE34.basis',

        # path to data
        'path2exec': '~/lcmodel-6.3-1N/bin/lcmodel',  # LCModel executable path
        'control': '../Data/Other/BrainBeats.control',
        # 'control': '../Data/Other/Synth.control',

        # in vivo
        'path2raw': '../../raw/PARREC_physlog_raw/',  # path to in vivo raw data
        'path2data': '../Data/DataSets/BrainBeatsFSLCA_TE34/',  # path to in vivo data
        'filter_out': ['1004', '1044', '1070', '1073'],   # filter out scans
        # 1004 - ringing artifact
        # 1044 - shimming issue
        # 1070 - lipids
        # 1073 - shimming issue

        # processing
        'process': 'custom',   # 'custom', 'fsl-mrs'
        'coil_comb': 'simple',   # (only if process='custom'): 'adaptive', 'simple', 'fls-mrs'

        # model
        'model': 'nn',  # 'nn', 'ls', 'lcm ', ...
        'specification': [   # model or list of models to test or 'all' for all models
            'super',
            'selfsuper',
            'ttia_super',
            'ttoa_super',
            'ttda_super',
            'own_lcm_gd',
            'newton',
            'lcmodel',
            #
            # 'cnn_super',
            # 'cnn_selfsuper',
            # 'cnn_ttia_super',
            # 'cnn_ttia_selfsuper',
            #
            # 'ttia_selfsuper',
            # 'ttia_scratch',
            # 'ttoa_selfsuper',
            # 'ttda_selfsuper',
            #
            # 'ttia_super_iter10',
            # 'ttia_super_iter100',
            # 'ttia_super_iter500',
        ],
        'loss': 'mse_specs',  # 'mae', 'mse', ...
        'state': 'eval',  # 'train', 'eval'

        # setup
        'specType': 'auto',  # 'auto', 'synth', 'invivo', 'fMRSinPain', 'all'
        'ppmlim': (0.5, 4.0),  # ppm limits for the spectra (used if specType='auto')
        'basisFmt': '7tslaser',  # '', 'cha_philips', 'fMRSinPain', 'biggaba', '7tslaser', ...
        'dataType': 'aumc2',  # 'clean', 'std', 'std_rw', 'std_rw_p', 'custom', ...

        # for nn model
        'arch': 'mlp',  # 'mlp', 'cnn' ...
        'norm': True,   # normalize the input spectra (and scale gts if forwardNorm = False)
        'forwardNorm': True,  # push the norm forward (i.e. scale the predictions with the norm)
        'activation': 'elu',  # 'relu', 'elu', 'tanh', 'sigmoid', 'softplus', ...
        'dropout': 0.0,
        'width': 512,   # (for mlp)
        'depth': 3,
        'conv_depth': 3,   # (for cnn)
        'kernel_size': 3,
        'stride': 1,

        # for gd model
        'adaptMode': 'per_spec_adapt',  # 'model_only', 'domain_adapt', 'stream_adapt', 'per_spec_adapt'
        'innerEpochs': 50,  # number of inner epochs
        'innerBatch': 1,  # batch size for the inner loop
        'innerLr': 1e-4,  # learning rate for the inner loop
        'innerLoss': 'mse_specs',  # loss function for the inner loop
        'bnState': 'train',  # 'train', 'eval' - batch normalization state for the inner loop
        'initMode': 'nn',  # 'nn', 'rand', 'fsl', ...
        'multiprocessing': False,  # use multiprocessing for the inner loop

        # for lcm model
        'method': 'Newton',  # 'Newton', 'MH', 'LCModel'
        'bandwidth': 3000,  # bandwidth of the spectra
        'sample_points': 1024,  # number of sample points
        'include_params': True,  # include signal parameters in the output (only for FSL-MRS)
        'save_path': '',  # path to save the results (empty for no saving)
        
        # system parameters
        'cf': 298029903/1E6,
        'bw': 3000,
        'TE': 0.036,
        'TR': 5.0,

        # mode
        'load_model': True,  # load model from path2trained
        'skip_train': True,  # skip the training procedure

        # scenario settings
        'scenarios': ['ID', 'Lor+Gau', 'Eps', 'Phi', 'Bas', 'SNR', 'RW', 'MM_CMR', 'OOD', 'Noise'],  # scenarios to test
        'addAllMeatbs': False,  # add all metabolites to the scenarios

        # visual settings
        'nsa_limits': [64, 32, 16, 8, 4],  # limits for the number of NSA blocks
        'run': False,  # run the inference (will try to load results if False)
        'save': True,   # save the results of the run
        'path2results': '../Test/paper/invivo/save_all/',  # path to the saved results
        'path2save': '../Test/paper/invivo/fsl_64_all/',  # path to save the results
        'gt_selection': 'fsl_64',  # 'fsl', 'lcm', 'mix' - pseudo ground truth selection
        'error': 'mae',  # 'mae', 'mse', 'mape', ...
        'exMM': True,  # exclude MM from the results, i.e. zero them
        'scale': 'off',  # 'on', 'off', 'select' - scale the concentrations optimally ('select' for LCModel and FSL-MRS)
        'ood_lines': True,  # add dashed lines for OoD ranges in the dists plots

        'plot_fits': False,  # plot the spectra and fits for each scenario
        'plot_performance': True,  # plot the performance of models in one
        'plot_ranges': False,  # plot the  ranges for each scenario
        'plot_distributions': True,  # plot the distributions of metabolites
        'plot_all_in_one': True,  # plot all scatter of dists in one figure
        'plot_heatmaps': True,  # plot the heatmaps of the errors
    }

    # place data type in path
    config['path2results'] = os.path.join(config['path2results'], config['dataType'])
    config['path2save'] = os.path.join(config['path2save'], config['dataType'])

    # checkpoint paths
    checkpoint_paths = {
        # aumc2_ms
        'super.mlp.aumc2_ms': './wandb/run-20250803_162000-g73hv30p/files/checkpoints/epoch=0-step=8738560.ckpt',
        'selfsuper.mlp.aumc2_ms': './wandb/run-20250803_162007-r2wqdtwd/files/checkpoints/epoch=0-step=1867776.ckpt',
        'hybrid.mlp.aumc2_ms': '',

        # aumc2
        'super.mlp.aumc2': './wandb/run-20250803_162127-wyoo5p40/files/checkpoints/epoch=0-step=7537664.ckpt',
        'selfsuper.mlp.aumc2': './wandb/run-20250803_162153-48ke5d6u/files/checkpoints/epoch=0-step=3572224.ckpt',
        'hybrid.mlp.aumc2': '',

        # aumc2_ms
        'super.cnn.aumc2_ms': './wandb/run-20250803_162112-64zynxew/files/checkpoints/epoch=0-step=2617600.ckpt',
        'selfsuper.cnn.aumc2_ms': './wandb/run-20250803_162057-v53fzp9m/files/checkpoints/epoch=0-step=1530624.ckpt',
        'hybrid.cnn.aumc2_ms': '',

        # aumc2
        'super.cnn.aumc2': './wandb/run-20250803_162250-mkasyalx/files/checkpoints/epoch=0-step=8883712.ckpt',
        'selfsuper.cnn.aumc2': './wandb/run-20250803_162322-fixfk343/files/checkpoints/epoch=0-step=2685184.ckpt',
        'hybrid.cnn.aumc2': '',
    }

    # set the checkpoint path
    if config['checkpoint_path'] == '':
        if not isinstance(config['specification'], list):
            ckpt_key = config['specification'] + '.' + config['arch'] + '.' + config['dataType']
            if ckpt_key in checkpoint_paths:
                config['checkpoint_path'] = checkpoint_paths[ckpt_key]

    # main setup (in distribution)
    test = Test(config)

    # define OoD scenarios
    scenarios = config['scenarios']
    scenarios = scenarios + test.system.basisObj.names if config['addAllMeatbs'] else scenarios
    nsa_limits = config.get('nsa_limits', [64])  # limits for the number of NSA blocks

    # define model(s)
    model_defs = get_models(config, checkpoint_paths)
    if config['specification'] == 'all':
        models = {}
        for key, model in model_defs.items():
            models[key] = test.getModel(model)
    elif isinstance(config['specification'], list):
        models = {}
        for model in config['specification']:
            if model in model_defs:
                models[model] = test.getModel(model_defs[model])
            else:
                print(f'Warning: Model {model} not found in model definitions. Skipping.')
    elif config['specification'] in model_defs:
        models = {test.config.specification: test.getModel(model_defs[test.config.specification])}
    else:
        raise ValueError(f"Specification '{config['specification']}' not found in model definitions.")

    # make sure 'newton' and 'lcmodel' are in the models
    if 'newton' not in models:  models['newton'] = test.getModel(model_defs['newton'])
    if 'lcmodel' not in models:  models['lcmodel'] = test.getModel(model_defs['lcmodel'])

    # define colors for the models
    colors = get_colors()


    #*************#
    #   running   #
    #*************#
    if config['run']:
        print('Running tests...')

        print(test.system.basisObj.names)
        BB = BrainBeatsDataModule()

        # check if the data exists
        def blocks_exist(data_dir, internal):
            for item in os.listdir(data_dir):
                subj_path = os.path.join(data_dir, item)
                if os.path.isdir(subj_path) and item.startswith('B'):
                    for block_name in os.listdir(subj_path):
                        block_path = os.path.join(subj_path, block_name, internal)
                        metab_file = os.path.join(block_path, 'metab.nii.gz')
                        wref_file = os.path.join(block_path, 'wref.nii.gz')
                        if os.path.exists(metab_file) and os.path.exists(wref_file):
                            return True
            return False

        # load the data
        nsa_data_dict = {}
        for limit in nsa_limits:
            internal = f'preproc_{limit}'
            if blocks_exist(config['path2data'], internal):
                data_dict = BB.load_all_blocks(config['path2data'], internal=internal)
            else:
                print(f"Data not found for {internal}. Writing blocks...")
                BB.write_blocks(config['path2data'], config['path2data'], config['process'],
                                config['coil_comb'], limit_nsa=limit - 1)
                if blocks_exist(config['path2data'], internal):
                    data_dict = BB.load_all_blocks(config['path2data'], internal=internal)
                else:
                    print(f"Still no data for {internal}. Converting raw data to NIfTI...")
                    BB.labraw2nifti(config['path2raw'], config['bw'], config['cf'],
                                    config['TE'], config['TR'], config['sample_points'],
                                    config['path2data'])
                    BB.write_blocks(config['path2data'], config['path2data'], config['process'],
                                    config['coil_comb'], limit_nsa=limit - 1)
                    data_dict = BB.load_all_blocks(config['path2data'], internal=internal)
            nsa_data_dict[limit] = data_dict

        all_specs, all_water = [], []
        all_concs_fsl, all_concs_lcm = [], []
        all_crlbs_fsl, all_crlbs_lcm = [], []
        all_fwhm, all_snr = [], []
        times = {}
        for limit, data_dict in nsa_data_dict.items():
            print(f"Loaded {len(data_dict)} subjects with limit {limit}...")

            # filter out subjects
            if config.get('filter_out'):
                to_remove = [key for key in data_dict if any(sub in key for sub in config['filter_out'])]
                for key in to_remove:
                    del data_dict[key]
                    print(f"Filtered out subject {key}.")
            print(f"Remaining subjects with limit {limit}: {len(data_dict)}")

            specs = [block['metab'][:] for val in data_dict.values() for block in val.values()]
            water = [block['wref'][:] for val in data_dict.values() for block in val.values()]

            # flatten first 4 dims (batch, x, y, z)
            specs = np.stack(specs, axis=0)  # stack all spectra
            specs = specs.reshape(-1, *specs.shape[4:])
            specs = np.fft.fft(specs, axis=-1)
            specs = np.stack([specs.real, specs.imag], axis=1)

            water = np.stack(water, axis=0)  # stack all water spectra
            water = water.reshape(-1, *water.shape[4:])

            all_specs.append(specs.copy())
            all_water.append(water.copy())

            try:
                # load the results
                fsl_concs = np.load(config['path2results'] + f'/pseudo_gt/fsl_concs_{limit}.npy')
                lcm_concs = np.load(config['path2results'] + f'/pseudo_gt/lcm_concs_{limit}.npy')

                fsl_crlbs = np.load(config['path2results'] + f'/pseudo_gt/fsl_crlbs_{limit}.npy')
                lcm_crlbs = np.load(config['path2results'] + f'/pseudo_gt/lcm_crlbs_{limit}.npy')

                fwhm = np.load(config['path2results'] + f'/pseudo_gt/fwhm_{limit}.npy')
                snr = np.load(config['path2results'] + f'/pseudo_gt/snr_{limit}.npy')

            except:
                print(f'Running FSL-MRS and LCModel for limit {limit}...')

                # run FSL-MRS
                fsl = test.getModel(model_defs['newton'])
                fsl.save_path = config['path2save'] + '/newton/' + str(limit) + '/'
                start_time = time.time()
                fsl_concs, fsl_crlbs = fsl(specs, water)
                times['newton'] = {limit: time.time() - start_time}

                assert len(fsl.basisFSL.names) == len(test.system.basisObj.names), \
                    f"FSL-MRS basis set names ({len(fsl.basisFSL.names)}) do not match system basis set names ({len(test.system.basisObj.names)})"
                fsl_concs[:, :len(fsl.basisFSL.names)] = np.stack([fsl_concs[:, fsl.basisFSL.names.index(m)]   # sort
                                                                for m in test.system.basisObj.names], axis=1)
                fsl_crlbs[:, :len(fsl.basisFSL.names)] = np.stack([fsl_crlbs[:, fsl.basisFSL.names.index(m)]   # sort
                                                                for m in test.system.basisObj.names], axis=1)

                # run LCModel
                lcm = test.getModel(model_defs['lcmodel'])
                lcm.save_path = config['path2save'] + '/lcmodel/' + str(limit) + '/'
                start_time = time.time()
                lcm_concs, lcm_crlbs = lcm(specs, water)
                times['lcmodel'] = {limit: time.time() - start_time}
                lcm_concs = np.stack([lcm_concs[:, lcm.basisFSL.names.index(m)]   # sort
                                                for m in test.system.basisObj.names], axis=1)
                lcm_crlbs = np.stack([lcm_crlbs[:, lcm.basisFSL.names.index(m)]   # sort
                                                for m in test.system.basisObj.names], axis=1)

                fwhm = [lcm.read_LCModel_coord(f'{lcm.save_path}temp{i}.coord', coord=False, meta=True)[0] 
                        for i in range(len(specs))]
                print(f'FWHM: {np.min(fwhm)} - {np.max(fwhm)} Hz')
                snr = [lcm.read_LCModel_coord(f'{lcm.save_path}temp{i}.coord', coord=False, meta=True)[1] 
                    for i in range(len(specs))]
                print(f'SNR: {np.min(snr)} - {np.max(snr)}')

                # save the results
                if not os.path.exists(config['path2save'] + '/pseudo_gt/'):
                    os.makedirs(config['path2save'] + '/pseudo_gt/')
                np.save(config['path2save'] + f'/pseudo_gt/fsl_concs_{limit}.npy', fsl_concs)
                np.save(config['path2save'] + f'/pseudo_gt/lcm_concs_{limit}.npy', lcm_concs)

                # save the crlbs
                np.save(config['path2save'] + f'/pseudo_gt/fsl_crlbs_{limit}.npy', fsl_crlbs)
                np.save(config['path2save'] + f'/pseudo_gt/lcm_crlbs_{limit}.npy', lcm_crlbs)

                np.save(config['path2save'] + f'/pseudo_gt/fwhm_{limit}.npy', np.array(fwhm))
                np.save(config['path2save'] + f'/pseudo_gt/snr_{limit}.npy', np.array(snr))

            all_concs_fsl.append(fsl_concs.copy())
            all_concs_lcm.append(lcm_concs.copy())
            all_crlbs_fsl.append(fsl_crlbs.copy())
            all_crlbs_lcm.append(lcm_crlbs.copy())
            all_fwhm.append(fwhm.copy())
            all_snr.append(snr.copy())

        # concatenate all data
        specs = np.concatenate(all_specs, axis=0)
        water = np.concatenate(all_water, axis=0)
        fsl_concs = np.concatenate(all_concs_fsl, axis=0)
        lcm_concs = np.concatenate(all_concs_lcm, axis=0)
        fsl_crlbs = np.concatenate(all_crlbs_fsl, axis=0)
        lcm_crlbs = np.concatenate(all_crlbs_lcm, axis=0)
        fwhm = np.concatenate(all_fwhm, axis=0)
        snr = np.concatenate(all_snr, axis=0)
        print(f'Specs shape: {specs.shape}, Water shape: {water.shape}')

        # merge times
        for key in times:
            times[key] = sum(times[key].values())
        
        # create the data loader
        scenarioData = torch.utils.data.DataLoader(specs, batch_size=256, shuffle=False)

        # run OoD tests
        results = {}
        for key, model in models.items():
            if not (isinstance(model, FrameworkLCModel) or isinstance(model, FrameworkFSL)):
                start_time = time.time()
                preds, specs = test.run_noGT(model=model, data=scenarioData)
                times[key] = time.time() - start_time
            for scenario in scenarios:
                if isinstance(model, FrameworkLCModel) or isinstance(model, FrameworkFSL):
                    results[f'{key}_{scenario}'] = {'preds': torch.from_numpy(fsl_concs if key == 'newton' else lcm_concs),
                                                    'specs': specs, 'water': torch.from_numpy(water)}
                else:
                    results[f'{key}_{scenario}'] = {'preds': preds, 'specs': specs, 'water': torch.from_numpy(water)}

        if config['save']:
            os.makedirs(config['path2save'] + '/results/', exist_ok=True)

            for key, model in models.items():
                with h5py.File(os.path.join(config['path2save'], 'results', f'{key}.h5'), 'w') as f:
                    f.create_dataset('preds', data=results[f'{key}_ID']['preds'].cpu().numpy())
                    f.create_dataset('specs', data=results[f'{key}_ID']['specs'].cpu().numpy())
                    f.create_dataset('water', data=results[f'{key}_ID']['water'].cpu().numpy())
                    f.create_dataset('snr', data=10 * np.log10(np.array(snr)))

            # save config
            with open(os.path.join(config['path2save'], 'config.txt'), 'w') as f:
                for key, val in config.items():
                    f.write(f'{key}: {val}\n')

            # save model configs
            if not os.path.exists(config['path2save'] + '/models_configs/'):
                os.makedirs(config['path2save'] + '/models_configs/')
            for key, model in model_defs.items():
                with open(os.path.join(config['path2save'], 'models_configs', f'{key}.json'), 'w') as f:
                    json.dump(model, f, indent=4)

            # save times
            with open(os.path.join(config['path2save'], 'times.txt'), 'w') as f:
                for key, val in times.items():
                    f.write(f'{key}:\n')
                    f.write(f'Time taken: {val / specs.shape[0]} seconds\n')
                    f.write(f'Measured per sample with {specs.shape[0]} samples.\n\n')


    #******************#
    #   load results   #
    #******************#
    results = {}
    for scenario in scenarios:
        for key, model in models.items():

            # load the results from the saved files
            with h5py.File(os.path.join(config['path2results'], 'results', f'{key}.h5'), 'r') as f:
                results[f'{key}_{scenario}'] = {
                    'preds': torch.tensor(f['preds'][...]),
                    'specs': torch.tensor(f['specs'][...]),
                    'water': torch.tensor(f['water'][...]),
                    'snr': torch.tensor(f['snr'][...]),
                }

            # add save-path to model-based models
            if isinstance(model, FrameworkLCModel) or isinstance(model, FrameworkFSL):
                model.save_path = config['path2results'] + '/' + key + '/'

            # replace SNR with FSL-MRS estimate
            if isinstance(model, FrameworkFSL):
                snrs = []
                for i, lim in enumerate(nsa_limits):
                    for j in range(results[f'{key}_{scenario}']['preds'].shape[0] // len(nsa_limits)):
                        res = pickle.load(open(model.save_path + str(lim) + '/' + f'opt{j}.pkl', 'rb'))
                        snrs.append(10 * np.log10(res.SNR.spectrum))
                results[f'{key}_{scenario}']['snr'] = torch.from_numpy(np.array(snrs))

    for scenario in scenarios:
        for key, model in models.items():
            results[f'{key}_{scenario}']['snr'] = results[f'newton_{scenario}']['snr'].clone()


    #***********************#
    #   compute pseudo GT   #
    #***********************#
    print(f'Computing pseudo ground truth...')
    for scenario in scenarios:
        if config['gt_selection'] == 'fsl_64':  # use FSL-MRS as pseudo ground truth with 64 NSA
            truths = results[f'newton_{scenario}']['preds']
            truths = truths[:truths.shape[0] // len(nsa_limits)]
            truths = torch.cat([truths for _ in range(len(nsa_limits))], dim=0)
        elif config['gt_selection'] == 'lcm_64':  # use LCModel as pseudo ground truth with 64 NSA
            truths = results[f'newton_{scenario}']['preds']
            truths = truths[:truths.shape[0] // len(nsa_limits)]
            truths = torch.cat([truths for _ in range(len(nsa_limits))], dim=0)
            truths_lcm = results[f'lcmodel_{scenario}']['preds']
            truths_lcm = truths_lcm[:truths_lcm.shape[0] // len(nsa_limits)]
            truths_lcm = torch.cat([truths_lcm for _ in range(len(nsa_limits))], dim=0)
            truths[:, :len(test.system.basisObj.names)] = truths_lcm
        elif config['gt_selection'] == 'mix_64':
            truths = results[f'newton_{scenario}']['preds']
            truths = truths[:truths.shape[0] // len(nsa_limits)]
            truths = torch.cat([truths for _ in range(len(nsa_limits))], dim=0)
            truths_lcm = results[f'lcmodel_{scenario}']['preds']
            truths_lcm = truths_lcm[:truths_lcm.shape[0] // len(nsa_limits)]
            truths_lcm = torch.cat([truths_lcm for _ in range(len(nsa_limits))], dim=0)
            truths[:, :len(test.system.basisObj.names)] /= 2
            truths[:, :len(test.system.basisObj.names)] += truths_lcm / 2
        elif config['gt_selection'] == 'fsl': # use FSL-MRS as pseudo ground truth
            truths = results[f'newton_{scenario}']['preds']
        elif config['gt_selection'] == 'lcm':  # use LCModel as pseudo ground truth
            truths = results[f'newton_{scenario}']['preds'].clone()   # params of FSL
            truths[:, :len(test.system.basisObj.names)] = results[f'lcmodel_{scenario}']['preds']
        elif config['gt_selection'] == 'mix':
            truths = results[f'newton_{scenario}']['preds'].clone()
            truths[:, :len(test.system.basisObj.names)] /= 2
            truths[:, :len(test.system.basisObj.names)] += results[f'lcmodel_{scenario}']['preds'] / 2
        else:
            raise ValueError(f"Ground truth selection '{config['gt_selection']}' not recognized.")
        
        for key, val in results.items():
            if key.endswith(f'_{scenario}'):
                val['truths'] = truths.clone()


    #*********************#
    #   compute metrics   #
    #*********************#
    for key, val in results.items():

        err_mae = test.system.concsLoss(val['truths'].clone(), val['preds'].clone(), 'mae', exMM=config['exMM'])

        # save additional metrics
        if not os.path.exists(config['path2save'] + '/add_metrics/'):
            os.makedirs(config['path2save'] + '/add_metrics/')

        # # print the errors
        # print(f'{key}) - MAE (exclude MMs: {config["exMM"]}): '
        #       f'{err_mae.mean().item():.4f} '
        #       f'({err_mae.std().item():.4f})')

        with open(config['path2save'] + '/add_metrics/' + f'{key}.txt', 'w') as f:
            f.write(f'{key} - MAE (exclude MMs: {config["exMM"]}):\n '
                    f'{err_mae.mean().item():.4f} '
                    f'(± {err_mae.std().item() / np.sqrt(len(err_mae)):.4f})\n')

    # scale metabolites if required
    if config['scale'] != 'off':
        print(f'Scaling metabolites {"for all models" if config["scale"] == "on" else "for LCModel and FSL-MRS only"}...')
        for scenario in scenarios:
            for key, model in models.items():
                # try to load the scaled predictions if they exist
                try:
                    with h5py.File(os.path.join(config['path2save'], 'results_mosae', f'{key}_{scenario}.h5'), 'r') as f:
                        results[f'{key}_{scenario}']['scale_preds'] = torch.tensor(f['scale_preds'][...])

                # run the scaling if not loaded
                except:
                    if (config['scale'] == 'select' and
                            (isinstance(model, FrameworkLCModel) or isinstance(model, FrameworkFSL))):
                        w = test.system.optimalReference(results[f'{key}_{scenario}']['truths'][:, :test.system.basisObj.n_metabs].clone(),
                                                         results[f'{key}_{scenario}']['preds'][:, :test.system.basisObj.n_metabs].clone(),
                                                         exMM=config['exMM'])
                        results[f'{key}_{scenario}']['scale_preds'] = w * results[f'{key}_{scenario}']['preds'].clone()
                    elif config['scale'] == 'on':
                        w = test.system.optimalReference(results[f'{key}_{scenario}']['truths'][:, :test.system.basisObj.n_metabs].clone(),
                                                         results[f'{key}_{scenario}']['preds'][:, :test.system.basisObj.n_metabs].clone(),
                                                         exMM=config['exMM'])
                        results[f'{key}_{scenario}']['scale_preds'] = w * results[f'{key}_{scenario}']['preds'].clone()

                    # save the scaled predictions
                    if config['save']:
                        if not os.path.exists(config['path2save'] + '/results_mosae/'):
                            os.makedirs(config['path2save'] + '/results_mosae/')
                        with h5py.File(os.path.join(config['path2save'], 'results_mosae', f'{key}_{scenario}.h5'), 'w') as f:
                            f.create_dataset('scale_preds', data=results[f'{key}_{scenario}']['scale_preds'].cpu().numpy())

    else:
        for key, model in models.items():
            if not (isinstance(model, FrameworkLCModel) or isinstance(model, FrameworkFSL)):
                # try to load the scaled concentrations
                try:
                    if config['run']:
                        raise OverwriteError("Forcing to recompute scaled concentrations.")
                    with h5py.File(os.path.join(config['path2results'], 'results_scale', f'{key}.h5'), 'r') as f:
                        for scenario in scenarios:
                            results[f'{key}_{scenario}']['scale_preds'] = torch.tensor(f['scale_preds'][...])
                except:
                    print(f'No water scaled concentrations found for {key}. Computing them...')
                    # compute the scaled concentrations
                    concs = results[f'{key}_ID']['preds'].clone().cpu().numpy()
                    specs = results[f'{key}_ID']['specs'].clone()
                    water = results[f'{key}_ID']['water'].clone().cpu().numpy()
                    scale_preds = model.scaleConcs(specs, concs, water, TE=config['TE'], TR=config['TR'])

                    for scenario in scenarios:
                        results[f'{key}_{scenario}']['scale_preds'] = np.real(scale_preds)
                        
                    # save the scaled concentrations
                    if not os.path.exists(config['path2save'] + '/results_scale/'):
                        os.makedirs(config['path2save'] + '/results_scale/')
                    with h5py.File(os.path.join(config['path2save'], 'results_scale', f'{key}.h5'), 'w') as f:
                        f.create_dataset('scale_preds', data=results[f'{key}_ID']['scale_preds'].cpu().numpy())

            # for LCModel and FSL-MRS concentrations are already scaled
            else:
                for scenario in scenarios:
                    results[f'{key}_{scenario}']['scale_preds'] = results[f'{key}_{scenario}']['preds'].clone()

    # recompute errors if necessary
    if config['error'] != 'msmae' or config['exMM']:
        print('Computing errors...')
        for key, val in results.items():
            if 'ID' in key:
                for name in test.system.basisObj.names:
                    if name in ['Ala', 'GABA', 'Gly', 'PE', 'Ser', 'Tau']: continue   # skip metabolites that are too OOD
                    if config['gt_selection'] in ['fsl', 'fsl_64'] and name in ['Asc', 'Asp', 'Gln']: continue
                    if config['gt_selection'] in ['lcm', 'lcm_64'] and name in ['Asc', 'Glu', 'Cr', 'PCr']: continue

                    if config['dataType'] == 'aumc2_ms':
                        minR, maxR = aumcConcsMS[name]['low_limit'], aumcConcsMS[name]['up_limit']
                    elif config['dataType'] == 'aumc2':
                        minR, maxR = aumcConcs[name]['low_limit'], aumcConcs[name]['up_limit']
                    else:
                        raise ValueError(f"Data type '{config['dataType']}' not recognized for concentration limits.")
                    
                    filter = (val['truths'][:, test.system.basisObj.names.index(name)] >= minR) & \
                                (val['truths'][:, test.system.basisObj.names.index(name)] <= maxR)
                    
                    val['truths'] = val['truths'][filter]
                    val['scale_preds'] = val['scale_preds'][filter]
                    val['preds'] = val['preds'][filter]
                    val['specs'] = val['specs'][filter]
                    val['snr'] = val['snr'][filter]

                print(f'Remaining spectra for scenario {key}_{scenario}: {val["truths"].shape[0]}')

            # compute the error
            results[key]['err'] = test.system.concsLoss(val['truths'].clone(), val['scale_preds'].clone(), config['error'],
                                                        exMM=config['exMM'])
            # print the errors
            if 'ID' in key or 'OOD' in key:
                print(f'{key}) - {config["error"]} (Scaling: {config["scale"]}, exclude MMs: {config["exMM"]}): '
                      f'{results[key]["err"].mean().item():.4f} '
                      f'({results[key]["err"].std().item():.4f})')

            # save additional metrics
            if not os.path.exists(config['path2save'] + '/add_metrics/'):
                os.makedirs(config['path2save'] + '/add_metrics/')

            with open(config['path2save'] + '/add_metrics/' + f'{key}.txt', 'w') as f:
                f.write(f'{key} - {config["error"]} (Scaling: {config["scale"]}, exclude MMs: {config["exMM"]}):\n '
                        f'{results[key]["err"].mean().item():.4f} '
                        f'(± {results[key]["err"].std().item() / np.sqrt(len(val["err"])):.4f})\n')


    #***************************#
    #   plot spectra and fits   #
    #***************************#
    if config['plot_fits']:
        print('Plotting spectra and fits for each scenario...')
        limit = 200   # number of spectra to plot

        # plot mean spectrum and all spectra
        from visualizations.plotFunctions import plot_dataset
        for scenario in ['ID', 'OOD']:
            key = list(models.keys())[0]  # use the first model for plotting
            specs = results[f'{key}_{scenario}']['specs'].cpu().numpy()
            specs = specs[:, 0] + 1j * specs[:, 1]  # convert to complex
            plot_dataset(specs, test.system.basisObj.ppm, (0.5, 4.5), f'{scenario}',
                         save_path=config['path2save'] + f'/specs/')

        # plot individual fits
        from visualizations.plotFunctions import plot_spec_and_fit
        for scenario in ['ID', 'OOD']:
            for key, model in models.items():
                for i, lim in enumerate(nsa_limits):
                    # lcmodel fits are loaded from coord files
                    if isinstance(model, FrameworkLCModel):
                        fits, specs = [], []
                        for j in range(min(results[f'{key}_{scenario}']['preds'].shape[0] // len(nsa_limits), limit)):
                            lcm_fit = model.read_LCModel_fit(model.save_path + str(lim) + '/' + f'temp{j}.coord')
                            fits.append(lcm_fit['completeFit'])
                            specs.append(lcm_fit['data'])
                            ppm = lcm_fit['ppm']
                        fits, specs = np.array(fits), np.array(specs)
                        ppmlim, true = None, None

                    # fsl-mrs fits are loaded from pickled results
                    elif isinstance(model, FrameworkFSL):
                        fits, specs = [], []
                        for j in range(min(results[f'{key}_{scenario}']['preds'].shape[0] // len(nsa_limits), limit)):
                            res = pickle.load(open(model.save_path + str(lim) + '/' + f'opt{j}.pkl', 'rb'))
                            fits.append(np.fft.fft(res.pred))
                            specs.append(np.fft.fft(res.pred + res.residuals))
                        fits = np.array(fits)
                        specs = np.array(specs)
                        ppm = test.system.basisObj.ppm
                        ppmlim = (0.5, 4.0)

                    # nn models
                    else:
                        fits = results[f'{key}_{scenario}']['preds']
                        fits = fits[fits.shape[0] // len(nsa_limits) * i:fits.shape[0] // len(nsa_limits) * (i + 1)]
                        fits = test.system.sigModel.forward(fits).cpu().numpy()
                        specs = results[f'{key}_{scenario}']['specs'].cpu().numpy()
                        specs = specs[specs.shape[0] // len(nsa_limits) * i:specs.shape[0] // len(nsa_limits) * (i + 1)]
                        specs = specs[:, 0] + 1j * specs[:, 1]
                        ppm = test.system.basisObj.ppm
                        ppmlim = (0.5, 4.0)

                    for i in range(min(limit, specs.shape[0])):
                        gt = None  # ground truth is not used in this case
                        plot_spec_and_fit(specs[i], fits[i], gt, ppm, ppmlim, f'fit{i}',
                                          save_path=config['path2save'] + f'/fits/{key}/{scenario}/{lim}/', specOnly=True)


    #**********************#
    #   plot performance   #
    #**********************#
    if config['plot_performance']:
        print('Plotting performance for each scenario...')

        def plotPerformances(param, models, colors, results, scenario, idx, minR, maxR, xlabel, ylabel, bins=20):

            if 'baseline' in xlabel.lower(): xmin, xmax, bins = None, None, 12
            elif 'frequency' in xlabel.lower(): xmin, xmax, bins = None, None, 12
            elif 'voigt' in xlabel.lower(): xmin, xmax = None, None
            elif 'phase' in xlabel.lower(): xmin, xmax, bins = None, None, 16
            elif 'mm' in xlabel.lower(): xmin, xmax, bins = None, None, 12
            else: xmin, xmax = None, None

            plt.figure(figsize=(6.4, 3.2))
            for key, model in models.items():
                err = results[f'{key}_{scenario}']['err'].cpu().numpy()
                if idx >= len(test.system.basisObj.names): err = err.mean(-1)
                else: err = err[:, idx]

                # bin and calculate mean in each bin
                bin_means, bin_edges, _ = binned_statistic(param, err, statistic='mean', bins=bins)

                # calculate bin centers
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                # interpolate over empty bins (NaNs)
                bin_means_series = pd.Series(bin_means)
                bin_means_interp = bin_means_series.interpolate(method='linear').fillna(method='bfill').fillna(
                    method='ffill').values

                if key.lower() in ['newton', 'mh', 'lcmodel']: ls = (0, (1, 1))
                elif 'tt' in key.lower(): ls = (0, (3, 1, 1, 1))
                elif 'own' in key.lower(): ls = (0, (3, 1, 1, 1, 1, 1))
                elif 'cnn' in key.lower(): ls = (0, (5, 1))
                else: ls = '--'

                plt.plot(bin_centers, bin_means_interp, label=key, color=colors[key], linestyle=ls)

            # get current axis and its limits
            ax = plt.gca()
            if xmin is None: xmin, _ = ax.get_xlim()
            if xmax is None: _, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # add dashed line distribution limits and shaded area for out-of-distribution
            if  minR is not None and maxR is not None:
                # define OoD region color and line style
                ood_color = '#ffcccc'  # light red
                ood_line_color = '#d62728'  # strong red
                ood_line_style = (0, (1, 2))  # dotted

                # draw vertical lines at minR and maxR spanning the y-range
                plt.plot([minR, minR], [ymin, ymax], color=ood_line_color, linestyle=ood_line_style, linewidth=1)
                plt.plot([maxR, maxR], [ymin, ymax], color=ood_line_color, linestyle=ood_line_style, linewidth=1)

                # shade the out-of-distribution regions
                plt.fill_betweenx([ymin, ymax], xmin, minR, color=ood_color, alpha=0.5, zorder=0, linewidth=0)
                plt.fill_betweenx([ymin, ymax], maxR, xmax, color=ood_color, alpha=0.5, zorder=0, linewidth=0)

            # plt.legend()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            # set x and y limits
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            if config['path2save'] != '':
                if scenario.lower() == 'ood':
                    name = xlabel.split(' ')[0].lower()  # use the first word of the xlabel as name
                    if not os.path.exists(config['path2save'] + f'/performances/ood/'):
                        os.makedirs(config['path2save'] + f'/performances/ood/')
                    plt.savefig(config['path2save'] + f'/performances/ood/{name}.svg', bbox_inches='tight', dpi=300)
                else:
                    if not os.path.exists(config['path2save'] + f'/performances/'):
                        os.makedirs(config['path2save'] + f'/performances/')
                    plt.savefig(config['path2save'] + f'/performances/{scenario}.svg', bbox_inches='tight', dpi=300)


        for scenario in tqdm(scenarios):
            if scenario.lower() in ['lor', 'gau', 'eps', 'phi']:
                if scenario.lower() == 'lor':
                    idx = len(test.system.basisObj.names)
                    label = 'Linewidth (Lorentzian) [1/s]'
                    minR, maxR = test.system.ps['broadening'][0][0], test.system.ps['broadening'][1][0]
                elif scenario.lower() == 'gau':
                    idx = len(test.system.basisObj.names) + 1
                    label = 'Linewidth (Gaussian) [1/s]'
                    minR, maxR = test.system.ps['broadening'][0][1], test.system.ps['broadening'][1][1]
                elif scenario.lower() == 'eps':
                    idx = len(test.system.basisObj.names) + 2
                    label = 'Frequency [rad/s]'
                    minR, maxR = test.system.ps['shifting'][0], test.system.ps['shifting'][1]
                elif scenario.lower() == 'phi':
                    idx = len(test.system.basisObj.names) + 3
                    label = 'Phase [rad]'
                    minR, maxR = test.system.ps['phi0'][0], test.system.ps['phi0'][1]
                plotPerformances(results[f'{list(models.keys())[0]}_{scenario}']['truths'][:, idx].numpy(),
                                 models, colors, results, scenario, idx, minR, maxR, label,
                                 'MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]')

            elif scenario.lower() == 'lor+gau':
                lor = results[f'{list(models.keys())[0]}_{scenario}']['truths'][:, len(test.system.basisObj.names)].numpy()
                gau = results[f'{list(models.keys())[0]}_{scenario}']['truths'][:, len(test.system.basisObj.names) + 1].numpy()
                idx = len(test.system.basisObj.names) + 20
                minR = test.system.ps['broadening'][0][0] + test.system.ps['broadening'][0][1]
                maxR = test.system.ps['broadening'][1][0] + test.system.ps['broadening'][1][1]
                plotPerformances(lor + gau, models, colors, results, scenario, idx, minR, maxR,
                                 'Voigt Linewidth [1/s]',
                                 'MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]')

            elif scenario.lower() == 'bas':
                idx = len(test.system.basisObj.names) + 4
                baseline = np.abs(test.system.ps['baseline'][0] + test.system.ps['baseline'][1])
                minR, maxR = np.min(baseline), np.max(baseline)
                plotPerformances(results[f'{list(models.keys())[0]}_{scenario}']['truths'][:, idx:].abs().mean(dim=-1).numpy(),
                                 models, colors, results, scenario, idx, minR, maxR,
                                 'Baseline [a.u.]',
                                 'MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]')

            elif scenario.lower() == 'snr':
                idx = len(test.system.basisObj.names) + 20   # dummy index
                snr = results[f'{list(models.keys())[0]}_{scenario}']['snr'].numpy()

                plotPerformances(snr, models, colors, results, scenario, idx, 0, 40,
                                 'SNR [dB]',
                                 'MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]')

            # elif scenario.lower() == 'rw':
            #     # estimate the random walk parameters
            #     noise = results[f'{key}_{scenario}']['noise'].numpy()
            #     noise = noise[:, :, test.system.first:test.system.last]
            #     est_scale = np.abs(np.max(noise, axis=-1) - np.min(noise, axis=-1))

            #     plotPerformances(est_scale[:, 0],  # imaginary part
            #                      models, colors, results, scenario, idx, 0, 0,
            #                      'Random Walk Scale [a.u.]', 'MAE [mM]')

            elif scenario.lower() == 'ood':
                for idx, name in enumerate(test.system.basisObj.names):
                    minR, maxR = test.system.concs[name]['low_limit'], test.system.concs[name]['up_limit']
                    scenario_name = name if name.lower() != 'mm_cmr' else 'MM'

                    plotPerformances(results[f'{list(models.keys())[0]}_{scenario}']['truths'][:, idx].numpy(),
                                     models, colors, results, scenario, idx if name.lower() != 'mm_cmr' else idx + 20,
                                     minR, maxR,
                                     scenario_name + ' Concentration [mM]',
                                     'MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]')

            elif scenario in test.system.basisObj.names:  # metabolites
                idx = test.system.basisObj.names.index(scenario)
                minR, maxR = test.system.concs[scenario]['low_limit'], test.system.concs[scenario]['up_limit']
                scenario_name = scenario if scenario.lower() != 'mm_cmr' else 'MM'

                plotPerformances(results[f'{list(models.keys())[0]}_{scenario}']['truths'][:, idx].numpy(),
                                 models, colors, results, scenario, idx if scenario.lower() != 'mm_cmr' else idx + 20, minR, maxR,
                                 scenario_name + ' Concentration [mM]', 'MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]')


    #*****************#
    #   plot ranges   #
    #*****************#
    if config['plot_ranges']:
        print('Plotting ranges for each scenario...')
        from visualizations.plotFunctionsErr import plotParamRanges

        for snrPlot in [True, False]:
            for scenario in tqdm(scenarios):
                for key, model in models.items():

                    if snrPlot:
                        cval = results[f'{key}_{scenario}']['snr'].numpy()
                        cmin, cmax = np.min(cval), np.max(cval)
                    else:
                        cval = None
                        cmin, cmax = None, None

                    if scenario.lower() in ['lor', 'gau', 'eps', 'phi']:
                        if scenario.lower() == 'lor':
                            idx = len(test.system.basisObj.names)
                            label = 'Linewidth (Lorentzian) [1/s]'
                            minR, maxR = test.system.ps['broadening'][0][0], test.system.ps['broadening'][1][0]
                            bins = 12
                        elif scenario.lower() == 'gau':
                            idx = len(test.system.basisObj.names) + 1
                            label = 'Linewidth (Gaussian) [1/s]'
                            minR, maxR = test.system.ps['broadening'][0][1], test.system.ps['broadening'][1][1]
                            bins = 12
                        elif scenario.lower() == 'eps':
                            idx = len(test.system.basisObj.names) + 2
                            label = 'Frequency [rad/s]'
                            minR, maxR = test.system.ps['shifting'][0], test.system.ps['shifting'][1]
                            bins = 12
                        elif scenario.lower() == 'phi':
                            idx = len(test.system.basisObj.names) + 3
                            label = 'Phase [rad]'
                            minR, maxR = test.system.ps['phi0'][0], test.system.ps['phi0'][1]
                            bins = 12
                        plotParamRanges(results[f'{key}_{scenario}']['truths'][:, idx].numpy(),
                                        results[f'{key}_{scenario}']['err'].mean(dim=-1).numpy(),
                                        c=cval, cmin=cmin, cmax=cmax,
                                        minR=minR, maxR=maxR, xLabel=label, mode='real',
                                        yLabel='MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]', bins=bins,
                                        visR=key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel'])

                        # save x and y values for first couple of points
                        if config['path2save'] != '':
                            limit = min(100, results[f'{key}_{scenario}']['truths'].shape[0])  # limit to first 100 points
                            if not os.path.exists(config['path2save'] + f'/fits/{key}/{scenario}/'):
                                os.makedirs(config['path2save'] + f'/fits/{key}/{scenario}/')

                            np.savetxt(config['path2save'] + f'/fits/{key}/{scenario}/param_ranges.txt',
                                       np.column_stack((np.arange(limit, dtype=int),
                                                        results[f'{key}_{scenario}']['truths'][:limit, idx].numpy(),
                                                        results[f'{key}_{scenario}']['err'].mean(dim=-1)[:limit].numpy())),
                                       header='Index, Parameter Value, MAE [mM]',
                                       fmt='%.4f', delimiter=', ')

                    elif scenario.lower() == 'lor+gau':
                        lor = results[f'{key}_{scenario}']['truths'][:, len(test.system.basisObj.names)].numpy()
                        gau = results[f'{key}_{scenario}']['truths'][:, len(test.system.basisObj.names) + 1].numpy()
                        idx = len(test.system.basisObj.names) + 20
                        minR = test.system.ps['broadening'][0][0] + test.system.ps['broadening'][0][1]
                        maxR = test.system.ps['broadening'][1][0] + test.system.ps['broadening'][1][1]
                        plotParamRanges(lor + gau, results[f'{key}_{scenario}']['err'].mean(dim=-1).numpy(),
                                        c=cval, cmin=cmin, cmax=cmax,
                                        minR=minR, maxR=maxR, xLabel='Voigt Linewidth [1/s]', mode='real',
                                        yLabel='MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]', bins=12,
                                        visR=key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel'])

                        # save x and y values for first couple of points
                        if config['path2save'] != '':
                            limit = min(1000, len(lor + gau))  # limit to first 1000 points
                            if not os.path.exists(config['path2save'] + f'/fits/{key}/{scenario}/'):
                                os.makedirs(config['path2save'] + f'/fits/{key}/{scenario}/')

                            np.savetxt(config['path2save'] + f'/fits/{key}/{scenario}/param_ranges.txt',
                                       np.column_stack((np.arange(limit, dtype=int),
                                                        (lor + gau)[:limit],
                                                        results[f'{key}_{scenario}']['err'].mean(dim=-1)[:limit].numpy())),
                                       header='Index, Voigt Linewidth [1/s], MAE [mM]',
                                       fmt='%.4f', delimiter=', ')

                    elif scenario.lower() == 'bas':
                        idx = len(test.system.basisObj.names) + 4
                        baseline = np.abs(test.system.ps['baseline'][0] + test.system.ps['baseline'][1])
                        minR, maxR = np.min(baseline), np.max(baseline)
                        plotParamRanges(results[f'{key}_{scenario}']['truths'][:, idx:].abs().mean(dim=-1).numpy(),
                                        results[f'{key}_{scenario}']['err'].mean(dim=-1).numpy(),
                                        c=cval, cmin=cmin, cmax=cmax,
                                        minR=minR, maxR=maxR, xLabel='Baseline [a.u.]', mode='real',
                                        yLabel='MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]', bins=12,
                                        visR=key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel'])

                        # save x and y values for first couple of points
                        if config['path2save'] != '':
                            limit = min(1000, results[f'{key}_{scenario}']['truths'][:, idx:].abs().mean(dim=-1).shape[0])  # limit to first 1000 points
                            if not os.path.exists(config['path2save'] + f'/fits/{key}/{scenario}/'):
                                os.makedirs(config['path2save'] + f'/fits/{key}/{scenario}/')

                            np.savetxt(config['path2save'] + f'/fits/{key}/{scenario}/param_ranges.txt',
                                       np.column_stack((np.arange(limit, dtype=int),
                                                        results[f'{key}_{scenario}']['truths'][:limit, idx:].abs().mean(dim=-1).numpy(),
                                                        results[f'{key}_{scenario}']['err'].mean(dim=-1)[:limit].numpy())),
                                       header='Index, Baseline [a.u.], MAE [mM]',
                                       fmt='%.4f', delimiter=', ')

                    elif scenario.lower() == 'snr':
                        snr = results[f'{key}_{scenario}']['snr'].numpy()
                        err = results[f'{key}_{scenario}']['err'].mean(dim=-1).numpy()
                        plotParamRanges(snr, err, minR=None, maxR=None,  # hard estimate from testing range
                                        c=cval, cmin=cmin, cmax=cmax,
                                        xLabel='SNR [dB]', yLabel='MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]', mode='real', bins=12,
                                        visR=key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel'])

                        # save x and y values for first couple of points
                        if config['path2save'] != '':
                            limit = min(1000, len(snr))  # limit to first 1000 points
                            if not os.path.exists(config['path2save'] + f'/fits/{key}/{scenario}/'):
                                os.makedirs(config['path2save'] + f'/fits/{key}/{scenario}/')

                            np.savetxt(config['path2save'] + f'/fits/{key}/{scenario}/snr.txt',
                                       np.column_stack((np.arange(limit, dtype=int), snr[:limit], err[:limit])),
                                       header='Index, SNR [dB], MAE [mM]',
                                       fmt='%.4f', delimiter=', ')

                    elif scenario in test.system.basisObj.names:  # metabolites
                        idx = test.system.basisObj.names.index(scenario)
                        minR, maxR = test.system.concs[scenario]['low_limit'], test.system.concs[scenario]['up_limit']

                        met = results[f'{key}_{scenario}']['truths'][:, idx].numpy()
                        if scenario.lower() == 'mm_cmr': err = results[f'{key}_{scenario}']['err'].mean(dim=-1).numpy()
                        else: err = results[f'{key}_{scenario}']['err'][:, idx].numpy()

                        plotParamRanges(met, err, minR=minR, maxR=maxR, bins=12,
                                        c=cval, cmin=cmin, cmax=cmax,
                                        xLabel= scenario + ' Concentration [mM]' if scenario.lower() != 'mm_cmr' else 'MM Concentration [mM]',
                                        yLabel='MAE [mM]' if scenario.lower() == 'mm_cmr' and config['scale'] == 'off' else 'MOSAE', mode='real',
                                        visR=key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel'])

                        # save x and y values for first couple of points
                        if config['path2save'] != '':
                            limit = min(1000, results[f'{key}_{scenario}']['truths'][:, idx].shape[0])  # limit to first 1000 points
                            if not os.path.exists(config['path2save'] + f'/fits/{key}/{scenario}/'):
                                os.makedirs(config['path2save'] + f'/fits/{key}/{scenario}/')

                            np.savetxt(config['path2save'] + f'/fits/{key}/{scenario}/param_ranges.txt',
                                       np.column_stack((np.arange(limit, dtype=int),
                                                        results[f'{key}_{scenario}']['truths'][:limit, idx].numpy(),
                                                        results[f'{key}_{scenario}']['err'].mean(dim=-1)[:limit].numpy())),
                                       header='Index, Parameter Value, MAE [mM]',
                                       fmt='%.4f', delimiter=', ')

                    elif scenario.lower() == 'ood':
                        for idx, name in enumerate(test.system.basisObj.names):
                            minR, maxR = test.system.concs[name]['low_limit'], test.system.concs[name]['up_limit']
                            scenario_name = name if name.lower() != 'mm_cmr' else 'MM'

                            met = results[f'{key}_{scenario}']['truths'][:, idx].numpy()
                            err = results[f'{key}_{scenario}']['err'][:, idx].numpy()

                            plotParamRanges(met, err, minR=minR, maxR=maxR, bins=12,
                                            c=cval, cmin=cmin, cmax=cmax,
                                            xLabel=scenario_name + ' Concentration [mM]',
                                            yLabel=('MAE [mM]' if name.lower() == 'mm_cmr' else 'Absolute Error') if config['scale'] == 'off' else 'MOSAE [mM]',
                                            mode='real', visR=key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel'])

                            # save x and y values for first couple of points
                            if config['path2save'] != '':
                                if not os.path.exists(config['path2save'] + f'/fits/{key}/{scenario}/'):
                                    os.makedirs(config['path2save'] + f'/fits/{key}/{scenario}/')

                                np.savetxt(config['path2save'] + f'/fits/{key}/{scenario}/{name}_param_ranges.txt',
                                        np.column_stack((np.arange(len(met), dtype=int),
                                                            results[f'{key}_{scenario}']['truths'][:, idx].numpy(),
                                                            results[f'{key}_{scenario}']['err'][:, idx].numpy())),
                                        header='Index, Parameter Value, MAE [mM]',
                                        fmt='%.4f', delimiter=', ')

                                if not os.path.exists(config['path2save'] + f'/ranges{"_snr" if snrPlot else ""}/{key}/{scenario}/'):
                                    os.makedirs(config['path2save'] + f'/ranges{"_snr" if snrPlot else ""}/{key}/{scenario}/')
                                plt.savefig(config['path2save'] + f'/ranges{"_snr" if snrPlot else ""}/{key}/{scenario}/{name}.svg',
                                            bbox_inches='tight', dpi=300)
                        continue

                    else:
                        print(f'Scenario {scenario} not recognized for plotting ranges.')
                        continue

                    if config['path2save']!= '':
                        if not os.path.exists(config['path2save'] + f'/ranges{"_snr" if snrPlot else ""}/{key}/'):
                            os.makedirs(config['path2save'] + f'/ranges{"_snr" if snrPlot else ""}/{key}/')
                        plt.savefig(config['path2save'] + f'/ranges{"_snr" if snrPlot else ""}/{key}/{scenario}.svg',
                                    bbox_inches='tight', dpi=300)


    #****************************#
    #   plot distributions snr   #
    #****************************#
    if config['plot_distributions']:
        print('Plotting distributions for metabolites with SNR cmap...')
        from visualizations.plotFunctionsErr import scatterHist

        for scenario in tqdm(scenarios):
            for key, model in models.items():

                cval = results[f'{key}_{scenario}']['snr'].numpy()
                scen = 'SNR'
                cmin, cmax = int(np.min(cval) + 1), int(np.max(cval) - 1)
                print(np.min(cval) + 1, np.max(cval) - 1)

                if scenario in test.system.basisObj.names + ['ID', 'OOD']:

                        # draw the scatter plot and the histograms
                        for idx, name in enumerate(test.system.basisObj.names + ['NAA+NAAG', 'Cr+PCr', 'Glu+Gln', 'mIns+Gly', 'GPC+PCh']):

                            # create total metabs
                            if '+' in name:
                                mets = name.split('+')
                                gt_concs = sum([results[f'{key}_{scenario}']['truths'][:, test.system.basisObj.names.index(met)].numpy() for met in mets])
                                est_concs = sum([results[f'{key}_{scenario}']['scale_preds'][:, test.system.basisObj.names.index(met)].numpy() for met in mets])
                                errors = sum([results[f'{key}_{scenario}']['err'][:, test.system.basisObj.names.index(met)].numpy() for met in mets])

                            else:
                                if scenario not in ['ID', 'OOD'] and name != scenario: continue

                                gt_concs = results[f'{key}_{scenario}']['truths'][:, idx].numpy()
                                est_concs = results[f'{key}_{scenario}']['scale_preds'][:, idx].numpy()
                                errors = results[f'{key}_{scenario}']['err'][:, idx].numpy()

                            # shuffle
                            perm = np.random.permutation(len(gt_concs))
                            gt_concs, est_concs, errors, cval = gt_concs[perm], est_concs[perm], errors[perm], cval[perm]

                            if config['gt_selection'] in ['lcm', 'lcm_64']:  xlabel_name = 'LCModel Concentration [mM]'
                            elif config['gt_selection'] in ['fsl', 'fsl_64']: xlabel_name = 'FSL-MRS Concentration [mM]'
                            else: xlabel_name = 'Pseudo-True Concentration [mM]'

                            # add out-of-distribution lines
                            if (config['ood_lines'] and key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel']
                                    and scenario.lower() == 'ood'):
                                from simulation.simulationDefs import aumcConcs, aumcConcsMS

                                if '+' in name:
                                    mets = name.split('+')
                                    minR = sum([aumcConcs[met]['low_limit'] for met in mets])
                                    maxR = sum([aumcConcs[met]['up_limit'] for met in mets])
                                else:
                                    minR, maxR = aumcConcs[name]['low_limit'], aumcConcs[name]['up_limit']
                                ax, ax_histx, ax_histy = scatterHist(gt_concs, est_concs,
                                                                     xLabel=xlabel_name,
                                                                     yLabel='Estimated Concentration [mM]',
                                                                     minR=minR, maxR=maxR,
                                                                     c=cval, cmin=cmin, cmax=cmax, bins=100, 
                                                                     name=name, lines=False)

                                if '+' in name:
                                    mets = name.split('+')
                                    minR = sum([aumcConcsMS[met]['low_limit'] for met in mets])
                                    maxR = sum([aumcConcsMS[met]['up_limit'] for met in mets])
                                else:
                                    minR, maxR = aumcConcsMS[name]['low_limit'], aumcConcsMS[name]['up_limit']
                                min_x, max_x = ax.get_xlim()
                                min_y, max_y = ax.get_ylim()

                                # define OoD region color and line style
                                ood_color = '#ffcccc'  # light red
                                ood_line_color = '#d62728'  # strong red
                                ood_line_style = (0, (1, 2))  # dotted

                                # draw OoD lines (dotted red) on the scatter
                                ax.plot([min_x, max_x], [minR, minR], color=ood_line_color,
                                        linestyle=ood_line_style, linewidth=1.5)
                                ax.plot([min_x, max_x], [maxR, maxR], color=ood_line_color,
                                        linestyle=ood_line_style, linewidth=1.5)

                                # shade OoD regions on y-axis (scatter)
                                ax.fill_between([min_x, max_x], min(min_y, minR), minR,
                                                color=ood_color, alpha=0.5, zorder=0)
                                ax.fill_between([min_x, max_x], maxR, max(maxR, max_y),
                                                color=ood_color, alpha=0.5, zorder=0)

                                # shade OoD regions on x-axis (histogram of predictions)
                                maxHist = np.max(ax_histy.get_xlim())
                                ax_histy.plot([0, maxHist], [minR, minR], color=ood_line_color,
                                              linestyle=ood_line_style, linewidth=1.5)
                                ax_histy.plot([0, maxHist], [maxR, maxR], color=ood_line_color,
                                              linestyle=ood_line_style, linewidth=1.5)
                                ax_histy.fill_between([0, maxHist], min(min_y, minR), minR,
                                                      color=ood_color, alpha=0.5, zorder=0)
                                ax_histy.fill_between([0, maxHist], maxR, max(maxR, max_y),
                                                      color=ood_color, alpha=0.5, zorder=0)

                            else:
                                if '+' in name:
                                    mets = name.split('+')
                                    minR = sum([test.system.concs[met]['low_limit'] for met in mets])
                                    maxR = sum([test.system.concs[met]['up_limit'] for met in mets])
                                else:
                                    minR, maxR = test.system.concs[name]['low_limit'], test.system.concs[name]['up_limit']
                                ax, ax_histx, ax_histy = scatterHist(gt_concs, est_concs,
                                                                     xLabel=xlabel_name,
                                                                     yLabel='Estimated Concentration [mM]',
                                                                     minR=minR, maxR=maxR,
                                                                     c=cval, cmin=cmin, cmax=cmax, bins=100, 
                                                                     name=name, lines=False)

                            if config['path2save']!= '':
                                if scenario == name:
                                    if not os.path.exists(config['path2save'] + f'/dists_snr/{key}/{name}/'):
                                        os.makedirs(config['path2save'] + f'/dists_snr/{key}/{name}/')
                                    plt.savefig(config['path2save'] + f'/dists_snr/{key}/{name}/{name}.svg',
                                                bbox_inches='tight', dpi=300)
                                elif scenario.lower() == 'ood':
                                    if not os.path.exists(config['path2save'] + f'/dists_snr/{key}/OOD/'):
                                        os.makedirs(config['path2save'] + f'/dists_snr/{key}/OOD')
                                    plt.savefig(config['path2save'] + f'/dists_snr/{key}/OOD/{scenario}_{name}.svg',
                                                bbox_inches='tight', dpi=300)
                                else:
                                    if not os.path.exists(config['path2save'] + f'/dists_snr/{key}/ID/'):
                                        os.makedirs(config['path2save'] + f'/dists_snr/{key}/ID')
                                    plt.savefig(config['path2save'] + f'/dists_snr/{key}/ID/{scenario}_{name}.svg',
                                                bbox_inches='tight', dpi=300)

    #*********************#
    #   plot all in one   #
    #*********************#
    if config['plot_all_in_one']:
        print('Plotting all metabolites in one scatter plot...')
        from visualizations.plotFunctionsErr import plotAllInOneScatter

        for scenario in tqdm(scenarios):
            for key, model in models.items():
                gt_concs = results[f'{key}_{scenario}']['truths'].numpy()
                est_concs = results[f'{key}_{scenario}']['scale_preds'].numpy()

                cval = results[f'{key}_{scenario}']['snr'].numpy()
                cmin, cmax = int(np.min(cval) + 1), int(np.max(cval) - 1)

                # remove MMs
                mm_idx = test.system.basisObj.names.index('MM_CMR')
                gt_concs = np.delete(gt_concs, mm_idx, axis=1)
                est_concs = np.delete(est_concs, mm_idx, axis=1)
                names = np.delete(test.system.basisObj.names, mm_idx)

                # shuffle
                perm = np.random.permutation(len(gt_concs))
                gt_concs, est_concs, cval = gt_concs[perm], est_concs[perm], cval[perm]

                if config['gt_selection'] in ['lcm', 'lcm_64']: xlabel_name = 'LCModel Concentration [mM]'
                elif config['gt_selection'] in ['fsl', 'fsl_64']: xlabel_name = 'FSL-MRS Concentration [mM]'
                else: xlabel_name = 'Pseudo-True Concentration [mM]'

                plotAllInOneScatter(gt_concs, est_concs, names, c=cval,
                                    cmin=cmin, cmax=cmax, xLabel=xlabel_name,
                                    yLabel='Estimated Concentration [mM]', xplot=5, yplot=4)
                if config['path2save'] != '':
                    if not os.path.exists(config['path2save'] + f'/all_in_one/{key}/'):
                        os.makedirs(config['path2save'] + f'/all_in_one/{key}/')
                    plt.savefig(config['path2save'] + f'/all_in_one/{key}/{scenario}.png',
                                bbox_inches='tight', dpi=300)


    #*******************#
    #   plot heatmaps   #
    #*******************#
    if config['plot_heatmaps']:
        print('Plotting heatmaps for metabolites...')

        import seaborn as sns
        scen_names = [scenario for scenario in scenarios if scenario not in ['Noise']]
        met_names = test.system.basisObj.names
        met_names[met_names.index('MM_CMR')] = 'MMs'

        model_names = {
            'super': 'Supervised',
            'selfsuper': 'Self-Supervised',
            'ttia_super': 'Test-Time Adaptive',
            'ttia_super_iter10': 'Test-Time Adaptive (10 Iter.)',
            'ttia_super_iter100': 'Test-Time Adaptive (100 Iter.)',
            'ttia_super_iter500': 'Test-Time Adaptive (500 Iter.)',
            'ttia_selfsuper': 'Test-Time Adaptive (Self-Sup. Init.)',
            'own_lcm': 'Purely Model-Based',
            'own_lcm_gd': 'Purely Model-Based',
            'cnn_super': 'Supervised (CNN)',
            'cnn_selfsuper': 'Self-Supervised (CNN)',
            'cnn_ttia_super': 'Test-Time Adaptive (CNN)',
            'cnn_ttia_selfsuper': 'Test-Time Adaptive (CNN, Self-Sup. Init.)',
            'ttia_scratch': 'Test-Time Adaptive (From Scratch Init.)',
            'ttoa_super': 'Test-Time Online Adaptive',
            'ttoa_selfsuper': 'Test-Time Online Adaptive (Self-Sup. Init.)',
            'ttda_super': 'Test-Time Domain Adaptive',
            'ttda_selfsuper': 'Test-Time Domain Adaptive (Self-Sup. Init.)',
            'newton': 'FSL-MRS (Newton)',
            'mh': 'FSL-MRS (MH)',
            'lcmodel': 'LCModel',
        }

        print(models.keys())

        mod_names = [model_names[key] for key in models.keys()]

        # for key, model in models.items():

        #     # compute mape
        #     con_matrix = np.stack([results[f'{key}_{scenario}']['truths']
        #                            for scenario in scenarios if scenario not in ['Noise']], axis=0)
        #     pred_matrix = np.stack([results[f'{key}_{scenario}']['scale_preds']
        #                             for scenario in scenarios if scenario not in ['Noise']], axis=0)
        #
        #     con_matrix = con_matrix[..., :pred_matrix.shape[-1]]
        #     pred_matrix = pred_matrix[..., :con_matrix.shape[-1]]
        #
        #     err_matrix_mean = np.abs((con_matrix - pred_matrix) / (con_matrix + 1e-10)) * 100  # MAPE
        #     err_matrix_mean = np.clip(err_matrix_mean, 0, 999)[..., :test.system.basisObj.n_metabs]
        #     err_matrix_mean = np.nanmean(err_matrix_mean, axis=1)
        #
        #     # create a heatmap
        #     plt.figure(figsize=(16, 6))
        #     sns.heatmap(err_matrix_mean, xticklabels=test.system.basisObj.names, yticklabels=scen_names,
        #                 cmap="Reds", annot=True, fmt=".2f")
        #     plt.title(f"Error heatmap (MAPE) for model {key}")
        #     # plt.xlabel("Metabolites")
        #     # plt.ylabel("Scenarios")
        #     plt.tight_layout()
        #
        #     if config['path2save'] != '':
        #         if not os.path.exists(config['path2save'] + f'/heatmaps/{key}/'):
        #             os.makedirs(config['path2save'] + f'/heatmaps/{key}/')
        #         plt.savefig(config['path2save'] + f'/heatmaps/{key}/heatmap.svg', bbox_inches='tight', dpi=300)


        for scenario in scenarios:
            if scenario in ['ID', 'OOD']:

                # compute mape
                con_matrix = np.stack([results[f'{key}_{scenario}']['truths'][..., :test.system.basisObj.n_metabs]
                                       for key in models.keys()], axis=0)
                pred_matrix = np.stack([results[f'{key}_{scenario}']['scale_preds'][..., :test.system.basisObj.n_metabs]
                                        for key in models.keys()], axis=0)

                err_matrix_mean = np.abs((con_matrix - pred_matrix) / (con_matrix + 1e-10)) * 100
                err_matrix_mean = np.clip(err_matrix_mean, 0, 999)[..., :test.system.basisObj.n_metabs]
                err_matrix_mean = np.nanmean(err_matrix_mean, axis=1)

                # create a heatmap
                plt.figure(figsize=(len(test.system.basisObj.names) / 2, len(mod_names) / 2))
                sns.heatmap(err_matrix_mean,
                            # xticklabels=test.system.basisObj.names, yticklabels=mod_names,
                            xticklabels=False, yticklabels=False,
                            cmap="Purples", annot=True, fmt=".1f", cbar=False)
                # plt.title(f"Error heatmap (MAPE) for scenario {scenario}")
                # plt.xlabel("Metabolites")
                # plt.ylabel("Models")
                plt.tight_layout()

                if config['path2save'] != '':
                    if not os.path.exists(config['path2save'] + f'/heatmaps/{scenario}/'):
                        os.makedirs(config['path2save'] + f'/heatmaps/{scenario}/')
                    plt.savefig(config['path2save'] + f'/heatmaps/{scenario}/heatmap.svg', bbox_inches='tight', dpi=300)


                # create heatmap for certain metbas
                # mets = ['Cr', 'Glu', 'GSH', 'mIns', 'Lac', 'NAA', 'PCh', 'PCr']
                mets = ['Cr', 'GABA', 'Gln', 'Glu', 'GPC', 'GSH', 'mIns', 'Lac', 'NAAG', 'NAA', 'PCh', 'PCr', 'Scyllo']
                met_idx = [test.system.basisObj.names.index(met) for met in mets]

                plt.figure(figsize=(len(mets) / 2, len(mod_names) / 2))
                sns.heatmap(err_matrix_mean[:, met_idx],
                            # xticklabels=mets, yticklabels=mod_names,
                            xticklabels=False, yticklabels=False,
                            cmap="Purples", annot=True, fmt=".1f", cbar=False)
                # plt.title(f"Error heatmap (MAPE) for scenario {scenario}")
                # plt.xlabel("Metabolites")
                # plt.ylabel("Models")
                plt.tight_layout()

                if config['path2save'] != '':
                    if not os.path.exists(config['path2save'] + f'/heatmaps/{scenario}/'):
                        os.makedirs(config['path2save'] + f'/heatmaps/{scenario}/')
                    plt.savefig(config['path2save'] + f'/heatmaps/{scenario}/heatmap_few_mets.svg', bbox_inches='tight', dpi=300)


                # create total metabs
                mets = ['NAA+NAAG', 'Cr+PCr', 'Glu+Gln', 'mIns+Gly', 'GPC+PCh']
                met_idx = [[test.system.basisObj.names.index(met.split('+')[0]),
                            test.system.basisObj.names.index(met.split('+')[1])] for met in mets]

                con_matrix_sum = np.stack([np.stack([results[f'{key}_{scenario}']['truths'][:, idx1] +
                                                  results[f'{key}_{scenario}']['truths'][:, idx2]
                                                  for idx1, idx2 in met_idx], axis=-1)
                                           for key in models.keys()], axis=0)
                pred_matrix_sum = np.stack([np.stack([results[f'{key}_{scenario}']['scale_preds'][:, idx1] +
                                                   results[f'{key}_{scenario}']['scale_preds'][:, idx2]
                                                    for idx1, idx2 in met_idx], axis=-1)
                                            for key in models.keys()], axis=0)

                # # add more
                # add_mets = ['GSH', 'GABA']
                # mets.extend(add_mets)
                #
                # con_matrix_sum = np.concatenate((con_matrix_sum,
                #                                 np.stack([np.stack([results[f'{key}_{scenario}']['truths'][:, test.system.basisObj.names.index(met)]
                #                                           for met in add_mets], axis=-1)
                #                                             for key in models.keys()], axis=0)), axis=-1)
                # pred_matrix_sum = np.concatenate((pred_matrix_sum,
                #                                  np.stack([np.stack([results[f'{key}_{scenario}']['scale_preds'][:, test.system.basisObj.names.index(met)]
                #                                           for met in add_mets], axis=-1)
                #                                             for key in models.keys()], axis=0)), axis=-1)

                err_matrix_mean_sum = np.abs((con_matrix_sum - pred_matrix_sum) / (con_matrix_sum + 1e-10)) * 100
                err_matrix_mean_sum = np.clip(err_matrix_mean_sum, 0, 999)
                err_matrix_mean_sum = np.nanmean(err_matrix_mean_sum, axis=1)

                plt.figure(figsize=(len(mets) / 2, len(mod_names) / 2))
                sns.heatmap(err_matrix_mean_sum, xticklabels=False, yticklabels=False, cmap="Purples", annot=True,
                            fmt=".1f", cbar=False, vmin=0, vmax=20)
                # plt.title(f"{scenario} (MAPE)")
                plt.axis("off")  # removes the axes completely
                plt.tight_layout()

                if config['path2save'] != '':
                    if not os.path.exists(config['path2save'] + f'/heatmaps/{scenario}/'):
                        os.makedirs(config['path2save'] + f'/heatmaps/{scenario}/')
                    plt.savefig(config['path2save'] + f'/heatmaps/{scenario}/heatmap_sum_mets.svg', bbox_inches='tight', dpi=300)

                # compute mae
                con_matrix = np.stack([results[f'{key}_{scenario}']['truths'][..., :test.system.basisObj.n_metabs]
                                       for key in models.keys()], axis=0)
                pred_matrix = np.stack(
                    [results[f'{key}_{scenario}']['scale_preds'][..., :test.system.basisObj.n_metabs]
                     for key in models.keys()], axis=0)
                err_matrix_mean = np.abs((con_matrix - pred_matrix))
                err_matrix_mean = np.nanmean(err_matrix_mean, axis=1)

                # create a heatmap
                plt.figure(figsize=(len(test.system.basisObj.names) / 2, len(mod_names) / 2))
                sns.heatmap(err_matrix_mean,
                            # xticklabels=test.system.basisObj.names, yticklabels=mod_names,
                            xticklabels=False, yticklabels=False,
                            cmap="Purples", annot=True, fmt=".2f", cbar=False, vmin=0, vmax=1)
                # plt.title(f"Error heatmap (MAPE) for scenario {scenario}")
                # plt.xlabel("Metabolites")
                # plt.ylabel("Models")
                plt.tight_layout()

                if config['path2save'] != '':
                    if not os.path.exists(config['path2save'] + f'/heatmaps_mae/{scenario}/'):
                        os.makedirs(config['path2save'] + f'/heatmaps_mae/{scenario}/')
                    plt.savefig(config['path2save'] + f'/heatmaps_mae/{scenario}/heatmap.svg', bbox_inches='tight',
                                dpi=300)

                # create heatmap for certain metbas
                # mets = ['Cr', 'Glu', 'GSH', 'mIns', 'Lac', 'NAA', 'PCh', 'PCr']
                # mets = ['Cr', 'GABA', 'Gln', 'Glu', 'GPC', 'GSH', 'mIns', 'Lac', 'NAAG', 'NAA', 'PCh', 'PCr', 'Scyllo']
                mets = ['Cr', 'Gln', 'Glu', 'Gly', 'GPC', 'mIns', 'NAAG', 'NAA', 'PCh', 'PCr']

                met_idx = [test.system.basisObj.names.index(met) for met in mets]

                plt.figure(figsize=(len(mets) / 2, len(mod_names) / 2))
                sns.heatmap(err_matrix_mean[:, met_idx],
                            # xticklabels=mets, yticklabels=mod_names,
                            xticklabels=False, yticklabels=False,
                            cmap="Purples", annot=True, fmt=".2f", cbar=False, vmin=0, vmax=1)
                # plt.title(f"Error heatmap (MAPE) for scenario {scenario}")
                # plt.xlabel("Metabolites")
                # plt.ylabel("Models")
                plt.tight_layout()

                if config['path2save'] != '':
                    if not os.path.exists(config['path2save'] + f'/heatmaps_mae/{scenario}/'):
                        os.makedirs(config['path2save'] + f'/heatmaps_mae/{scenario}/')
                    plt.savefig(config['path2save'] + f'/heatmaps_mae/{scenario}/heatmap_few_mets.svg',
                                bbox_inches='tight', dpi=300)

                # create total metabs
                mets = ['NAA+NAAG', 'Cr+PCr', 'Glu+Gln', 'mIns+Gly', 'GPC+PCh']
                met_idx = [[test.system.basisObj.names.index(met.split('+')[0]),
                            test.system.basisObj.names.index(met.split('+')[1])] for met in mets]

                con_matrix_sum = np.stack([np.stack([results[f'{key}_{scenario}']['truths'][:, idx1] +
                                                     results[f'{key}_{scenario}']['truths'][:, idx2]
                                                     for idx1, idx2 in met_idx], axis=-1)
                                           for key in models.keys()], axis=0)
                pred_matrix_sum = np.stack([np.stack([results[f'{key}_{scenario}']['scale_preds'][:, idx1] +
                                                      results[f'{key}_{scenario}']['scale_preds'][:, idx2]
                                                      for idx1, idx2 in met_idx], axis=-1)
                                            for key in models.keys()], axis=0)

                # # add more
                # add_mets = ['GSH', 'GABA']
                # mets.extend(add_mets)
                #
                # con_matrix_sum = np.concatenate((con_matrix_sum,
                #                                 np.stack([np.stack([results[f'{key}_{scenario}']['truths'][:, test.system.basisObj.names.index(met)]
                #                                           for met in add_mets], axis=-1)
                #                                             for key in models.keys()], axis=0)), axis=-1)
                # pred_matrix_sum = np.concatenate((pred_matrix_sum,
                #                                  np.stack([np.stack([results[f'{key}_{scenario}']['scale_preds'][:, test.system.basisObj.names.index(met)]
                #                                           for met in add_mets], axis=-1)
                #                                             for key in models.keys()], axis=0)), axis=-1)

                err_matrix_mean_sum = np.abs((con_matrix_sum - pred_matrix_sum))
                err_matrix_mean_sum = np.nanmean(err_matrix_mean_sum, axis=1)

                plt.figure(figsize=(len(mets) / 2, len(mod_names) / 2))
                sns.heatmap(err_matrix_mean_sum, xticklabels=False, yticklabels=False, cmap="Purples", annot=True,
                            fmt=".2f", cbar=False, vmin=0, vmax=1)
                # plt.title(f"{scenario} (MAPE)")
                plt.axis("off")  # removes the axes completely
                plt.tight_layout()

                if config['path2save'] != '':
                    if not os.path.exists(config['path2save'] + f'/heatmaps_mae/{scenario}/'):
                        os.makedirs(config['path2save'] + f'/heatmaps_mae/{scenario}/')
                    plt.savefig(config['path2save'] + f'/heatmaps_mae/{scenario}/heatmap_sum_mets.svg',
                                bbox_inches='tight', dpi=300)
