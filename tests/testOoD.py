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
from test import Test


#*************#
#   testing   #
#*************#
if __name__ == '__main__':

    # initialize the configuration
    config = {
        # path to a trained model
        'checkpoint_path': '',

        # path to basis set
        'path2basis': '../Data/BasisSets/7T_sLASER_OIT_TE34.basis',

        # path to data
        # 'path2exec': '~/lcmodel-6.3-1N/bin/lcmodel',  # LCModel executable path
        # 'control': '../Data/Other/BrainBeats.control',
        'control': '../Data/Other/Synth.control',

        # model
        'model': 'lcm',  # 'nn', 'ls', 'lcm', ...
        'specification': [   # model or list of models to test or 'all' for all models
            'super',
            'selfsuper',
            'ttia_super',
            'ttoa_super',
            'ttda_super',
            'own_lcm_gd',
            'newton',
            'lcmodel',

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
        'dataType': 'aumc2_ms',  # 'clean', 'std', 'std_rw', 'std_rw_p', 'custom', ...
        'test_size': 10000,  # number of test samples

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

        # for gd model (potentially overwritten in the modelDefs.py)
        'adaptMode': 'per_spec_adapt',  # 'model_only', 'domain_adapt', 'stream_adapt', 'per_spec_adapt'
        'innerEpochs': 50,  # number of inner epochs
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
        'save_path': '',  # path to save the results (empty for no saving)

        # mode
        'load_model': True,  # load model from path2trained
        'skip_train': True,  # skip the training procedure

        # scenario settings
        # 'scenarios': ['ID'],
        # 'scenarios': ['ID', 'OOD'],
        # 'scenarios': ['NAA', 'Cr', 'Glu', 'GSH', 'GABA', 'Gly'],
        # 'scenarios': ['ID', 'Lor+Gau', 'Eps', 'Phi', 'Bas', 'SNR', 'RW', 'MM_CMR', 'OOD', 'Noise'],  # scenarios to test
        'scenarios': ['ID', 'Lor+Gau', 'Eps', 'Phi', 'Bas', 'SNR', 'RW', 'MM_CMR', 'OOD', 'Noise', 'NAA', 'Cr', 'Glu', 'GSH', 'GABA', 'Gly'],

        # visual settings
        'load': True,  # load the data (will generate it if False)
        'run': False,  # run the inference (will try to load results if False)
        'save': True,   # save the results of the run
        'path2results': '../Test/paper/sim/10k/',  # path to the saved results
        'path2save': '../Test/paper/sim/10k_mosae/',  # path to save the results
        'error': 'mae',  # 'mae', 'mse', 'mape', ...
        'exMM': True,  # exclude MM from the results, i.e. zero them
        'scale': 'on',  # 'on', 'off', 'select' - scale the concentrations optimally ('select' for LCModel and FSL-MRS)
        'ood_lines': True,  # add dashed lines for OoD ranges in the dists plots

        'plot_fits': False,  # plot the spectra and fits for each scenario
        'plot_performance': True,  # plot the performance of models in one
        'plot_ranges': True,  # plot the ranges for each scenario
        'plot_distributions': True,  # plot the distributions of metabolites
        'plot_all_in_one': True,  # plot all scatter of dists in one figure
        'plot_histograms': True,  # plot the histograms (for noise scenario)
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

        # aumc2
        'super.mlp.aumc2': './wandb/run-20250803_162127-wyoo5p40/files/checkpoints/epoch=0-step=7537664.ckpt',
        'selfsuper.mlp.aumc2': './wandb/run-20250803_162153-48ke5d6u/files/checkpoints/epoch=0-step=3572224.ckpt',

        # aumc2_ms
        'super.cnn.aumc2_ms': './wandb/run-20250803_162112-64zynxew/files/checkpoints/epoch=0-step=2617600.ckpt',
        'selfsuper.cnn.aumc2_ms': './wandb/run-20250803_162057-v53fzp9m/files/checkpoints/epoch=0-step=1530624.ckpt',

        # aumc2
        'super.cnn.aumc2': './wandb/run-20250803_162250-mkasyalx/files/checkpoints/epoch=0-step=8883712.ckpt',
        'selfsuper.cnn.aumc2': './wandb/run-20250803_162322-fixfk343/files/checkpoints/epoch=0-step=2685184.ckpt',
    }

    # set the checkpoint path
    if config['checkpoint_path'] == '':
        if not isinstance(config['specification'], list):
            ckpt_key = config['specification'] + '.' + config['arch'] + '.' + config['dataType']
            if ckpt_key in checkpoint_paths:
                config['checkpoint_path'] = checkpoint_paths[ckpt_key]

    # main setup (in distribution)
    test = Test(config)

    print(test.system.basisObj.names)

    # define OoD scenarios
    scenarios = config['scenarios']

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

    # define colors for the models
    colors = get_colors()


    #******************************#
    #   get OoD concs and params   #
    #******************************#
    def getOoDParams(scenario, test, concs=None, params=None):
        if concs is None: concs = test.system.concs.copy()
        if params is None: params = test.system.ps.copy()
        concsOoD, paramsOoD = concs.copy(), params.copy()

        if scenario in ['ID']:  # in distribution
            pass

        elif scenario == 'Lor':  # Lorentzian broadening
            paramsOoD['broadening'] = [params['broadening'][0],
                                       (params['broadening'][1][0] * 2, params['broadening'][1][1])]
        elif scenario == 'Gau':  # Gaussian broadening
            paramsOoD['broadening'] = [params['broadening'][0],
                                       (params['broadening'][1][0], params['broadening'][1][1] * 2)]
        elif scenario == 'Lor+Gau':  # Lorentzian and Gaussian broadening
            paramsOoD['broadening'] = [params['broadening'][0],
                                       (params['broadening'][1][0] * 2, params['broadening'][1][1] * 2)]
        elif scenario == 'Eps':  # frequency shifting
            paramsOoD['shifting'] = [params['shifting'][0] * 4, params['shifting'][1] * 4]

        elif scenario == 'Phi':  # phase shifting
            paramsOoD['phi0'] = [-2.0, 2.0]

        elif scenario == 'Bas':  # baseline
            paramsOoD['baseline'] = [[elem * 4 for elem in params['baseline'][0]],
                                     [elem * 4 for elem in params['baseline'][1]]]
        elif scenario == 'SNR':  # SNR
            # paramsOoD['noise_std'] = [params['noise_std'][0] / 20, params['noise_std'][1] * 20]
            paramsOoD['noise_db'] = [-18, 64]  # SNR in dB
            del paramsOoD['noise_mean'], paramsOoD['noise_std']  # remove noise_mean and noise_std

        elif scenario == 'RW':  # random walk
            paramsOoD['scale'] = [0, 100000]
            paramsOoD['smooth'] = [1, 100000]
            paramsOoD['limits'] = [[-1000000, 0], [0, 1000000]]

        elif scenario in test.system.basisObj.names:
            # concsOoD[scenario] = {'name': scenario, 'low_limit': 0,
            #                       'up_limit': concs[scenario]['up_limit'] * 2}
            quarter = (concs[scenario]['up_limit'] - concs[scenario]['low_limit']) / 2
            concsOoD[scenario] = {
                'name': scenario,
                'low_limit': max(concs[scenario]['low_limit'] - quarter, 0),
                'up_limit': concs[scenario]['up_limit'] + quarter,
            }
        elif scenario == 'OOD':  # out-of-distribution
            for key in concs.keys():
                quarter = (concs[key]['up_limit'] - concs[key]['low_limit']) / 2
                concsOoD[key] = {
                    'name': key,
                    'low_limit': max(concs[key]['low_limit'] - quarter, 0),
                    'up_limit': concs[key]['up_limit'] + quarter,
                }
        elif scenario == 'OOD2':  # out-of-distribution
            for key in concs.keys():
                half = (concs[key]['up_limit'] - concs[key]['low_limit'])
                concsOoD[key] = {
                    'name': key,
                    'low_limit': max(concs[key]['low_limit'] - half, 0),
                    'up_limit': concs[key]['up_limit'] + half,
                }
        elif scenario == 'Noise':  # noise
            for key in concs.keys():
                concsOoD[key] = {'name': key, 'low_limit': 0, 'up_limit': 0}
                paramsOoD['baseline'] = [[0 for _ in params['baseline'][0]],
                                         [0 for _ in params['baseline'][1]]]
        else:
            raise ValueError(f'Unknown scenario: {scenario}')
        return concsOoD, paramsOoD


    #*************#
    #   running   #
    #*************#
    if config['run']:
        print('Running tests...')

        # memory usage
        import psutil, os, gc
        def print_memory(msg=""):
            process = psutil.Process(os.getpid())
            print(f"{msg} | RAM used: {process.memory_info().rss / 1024 ** 3:.2f} GB")

        # run OoD tests
        results = {}
        for scenario in tqdm(scenarios):
            concs, params = test.system.concs.copy(), test.system.ps.copy()
            concsOoD, paramsOoD = getOoDParams(scenario, test, concs, params)

            if config['load']:
                scenarioData = torch.load(config['path2results'] + '/data/' + scenario + '.pth', weights_only=False)
            else:
                scenarioData = test.getDataLoader(config=test.config, concs=concsOoD, params=paramsOoD).test_dataloader()
                if not os.path.exists(config['path2save'] + '/data/'): os.makedirs(config['path2save'] + '/data/')
                torch.save(scenarioData, config['path2save'] + '/data/' + scenario + '.pth')

            # only test_size samples
            scenarioData = torch.utils.data.DataLoader(
                torch.utils.data.Subset(scenarioData.dataset, range(config['test_size'])),
                batch_size=scenarioData.batch_size,
                shuffle=False,
                num_workers=scenarioData.num_workers
            )

            # run the models
            for key, model in tqdm(models.items(), leave=False):

                if isinstance(model, FrameworkLCModel) or isinstance(model, FrameworkFSL):
                    model.save_path = config['path2save'] + '/' + key + '/' + scenario + '/'

                start_time = time.time()
                truths, preds, specs, specs_sep = test.run(model=model, data=scenarioData)
                end_time = time.time()

                results[f'{key}_{scenario}'] = {'truths': truths,
                                                'preds': preds.to(truths.device),
                                                'specs': specs,
                                                'noise': specs_sep[..., -1].clone(),
                                                'concs': concs,
                                                'params': params,
                                                'concsOoD': concsOoD,
                                                'paramsOoD': paramsOoD,
                                                'timing': (end_time - start_time) / len(scenarioData.dataset)}

            print_memory(f"After scenario '{scenario}'")

        # compute error
        print('Computing errors...')
        for key, val in tqdm(results.items()):
            results[key]['err'] = test.system.concsLoss(val['truths'], val['preds'], 'msmae')

        if config['save']:
            os.makedirs(config['path2save'] + '/results/' , exist_ok=True)

            for scenario in scenarios:
                for key, model in models.items():
                    with h5py.File(os.path.join(config['path2save'], 'results', f'{key}_{scenario}.h5'), 'w') as f:
                        f.create_dataset('preds', data=results[f'{key}_{scenario}']['preds'].detach().cpu().numpy())
                        f.create_dataset('err', data=results[f'{key}_{scenario}']['err'].detach().cpu().numpy())
                        f.create_dataset('truths', data=results[f'{key}_{scenario}']['truths'].cpu().numpy())
                        f.create_dataset('specs', data=results[f'{key}_{scenario}']['specs'].cpu().numpy())
                        f.create_dataset('noise', data=results[f'{key}_{scenario}']['noise'].cpu().numpy())

                        # save JSON-serializable Python objects as byte strings
                        f.create_dataset('concs',
                                         data=np.bytes_(json.dumps(results[f'{key}_{scenario}']['concs'])))
                        f.create_dataset('params',
                                         data=np.bytes_(json.dumps(results[f'{key}_{scenario}']['params'])))
                        f.create_dataset('concsOoD',
                                         data=np.bytes_(json.dumps(results[f'{key}_{scenario}']['concsOoD'])))
                        f.create_dataset('paramsOoD',
                                         data=np.bytes_(json.dumps(results[f'{key}_{scenario}']['paramsOoD'])))

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

            # save MSMAE results and timings (on txt file in folder)
            for key, val in results.items():
                if not os.path.exists(os.path.join(config['path2save'], 'metrics')):
                    os.makedirs(os.path.join(config['path2save'], 'metrics'))

                with open(os.path.join(config['path2save'], 'metrics', f'{key}.txt'), 'w') as f:
                    f.write(f'{key} - msmae: '
                            f'{val["err"].mean().item():.4f} '
                            f'(± {val["err"].std().item() / np.sqrt(len(val["err"])):.4f})\n')
                    f.write(f'Time taken: {val["timing"]} seconds\n')
                    f.write(f'Measured per sample with {len(val["err"])} samples.\n')


    #******************#
    #   load results   #
    #******************#
    results = {}
    for scenario in scenarios:
        for key, model in models.items():
            with h5py.File(os.path.join(config['path2results'], 'results', f'{key}_{scenario}.h5'), 'r') as f:
                results[f'{key}_{scenario}'] = {
                    'truths': torch.tensor(f['truths'][...])[:config['test_size']],
                    'preds': torch.tensor(f['preds'][...])[:config['test_size']],
                    'specs': torch.tensor(f['specs'][...])[:config['test_size']],
                    'noise': torch.tensor(f['noise'][...])[:config['test_size']],
                    'err': torch.tensor(f['err'][...])[:config['test_size']],
                    'concs': json.loads(f['concs'][()].decode('utf-8')),
                    'params': json.loads(f['params'][()].decode('utf-8')),
                    'concsOoD': json.loads(f['concsOoD'][()].decode('utf-8')),
                    'paramsOoD': json.loads(f['paramsOoD'][()].decode('utf-8')),
                }

            if isinstance(model, FrameworkLCModel) or isinstance(model, FrameworkFSL):
                model.save_path = config['path2results'] + '/' + key + '/' + scenario + '/'


    #*********************#
    #   compute metrics   #
    #*********************#
    for key, val in tqdm(results.items()):
        err_mae = test.system.concsLoss(val['truths'].clone(), val['preds'].clone(), 'mae', exMM=config['exMM'])

        # save additional metrics
        if not os.path.exists(config['path2save'] + '/add_metrics/'):
            os.makedirs(config['path2save'] + '/add_metrics/')

        # print the errors
        print(f'{key}) - MAE (exclude MMs: {config["exMM"]}): '
              f'{err_mae.mean().item():.4f} '
              f'({err_mae.std().item():.4f})')

        with open(config['path2save'] + '/add_metrics/' + f'{key}.txt', 'w') as f:
            f.write(f'{key} - MAE (exclude MMs: {config["exMM"]}):\n '
                    f'{err_mae.mean().item():.4f} '
                    f'(± {err_mae.std().item() / np.sqrt(len(err_mae)):.4f})\n')

    # scale metabolites if required
    if config['scale'] != 'off':
        print(f'Scaling metabolites {"for all models" if config["scale"] == "on" else "for LCModel and FSL-MRS only"}...')
        for scenario in scenarios:
            if scenario == 'Noise':   # do not scale noise
                for key, model in models.items():
                    results[f'{key}_{scenario}']['scale_preds'] = results[f'{key}_{scenario}']['preds'].clone()
            else:
                for key, model in models.items():
                    # try to load the scaled predictions if they exist
                    try:
                        with h5py.File(os.path.join(config['path2results'], 'results', f'{key}_{scenario}.h5'), 'r') as f:
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
                            with h5py.File(os.path.join(config['path2results'], 'results', f'{key}_{scenario}.h5'), 'a') as f:
                                f.create_dataset('scale_preds', data=results[f'{key}_{scenario}']['scale_preds'].cpu().numpy())
    else:
        print('No scaling of metabolites applied.')
        for scenario in scenarios:
            for key, model in models.items():
                results[f'{key}_{scenario}']['scale_preds'] = results[f'{key}_{scenario}']['preds'].clone()

    # recompute errors if necessary
    if config['error'] != 'msmae' or config['exMM']:
        print('Computing errors...')
        for key, val in tqdm(results.items()):
            results[key]['err'] = test.system.concsLoss(val['truths'].clone(), val['scale_preds'].clone(), config['error'],
                                                        exMM=config['exMM'])
            # print the errors
            print(f'{key}) - {config["error"]} (Scaling: {config["scale"]}, exclude MMs: {config["exMM"]}): '
                  f'{results[key]["err"].mean().item():.4f} '
                  f'(±  {results[key]["err"].std().item() / np.sqrt(len(val["err"])):.4f})')

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
        # scenarios2plot = ['ID']
        scenarios2plot = ['ID', 'Lor', 'Gau', 'Eps', 'Phi', 'Bas', 'SNR', 'RW', 'MM_CMR', 'OOD', 'Lor+Gau', 'Noise']
        scenarios2plot += test.system.basisObj.names  # add basis set names to the scenarios to plot

        # plot mean spectrum and all spectra
        from visualizations.plotFunctions import plot_dataset
        for scenario in tqdm(scenarios):
            key = list(models.keys())[0]  # use the first model for plotting
            specs = results[f'{key}_{scenario}']['specs'].cpu().numpy()
            specs = specs[:, 0] + 1j * specs[:, 1]  # convert to complex
            plot_dataset(specs, test.system.basisObj.ppm, (0.5, 4.5), f'{scenario}',
                         save_path=config['path2save'] + f'/specs/')

        # plot individual fits
        from visualizations.plotFunctions import plot_spec_and_fit
        for scenario in tqdm(scenarios):
            if scenario not in scenarios2plot: continue
            for key, model in models.items():

                # lcmodel fits are loaded from coord files
                if isinstance(model, FrameworkLCModel):
                    fits, specs = [], []
                    model.save_path = config['path2results'] + '/' + key + '/' + scenario + '/'
                    for i in range(min(results[f'{key}_{scenario}']['preds'].shape[0], limit)):
                        lcm_fit = model.read_LCModel_fit(model.save_path + f'temp{i}.coord')
                        fits.append(lcm_fit['completeFit'])
                        specs.append(lcm_fit['data'])
                        ppm = lcm_fit['ppm']
                    fits, specs = np.array(fits), np.array(specs)
                    ppmlim, true = None, None

                # fsl-mrs fits are loaded from pickled results
                elif isinstance(model, FrameworkFSL):
                    import pickle
                    fits, specs = [], []
                    model.save_path = config['path2results'] + '/' + key + '/' + scenario + '/'
                    for i in range(min(results[f'{key}_{scenario}']['preds'].shape[0], limit)):
                        res = pickle.load(open(model.save_path + f'opt{i}.pkl', 'rb'))
                        fits.append(np.fft.fft(res.pred))
                        specs.append(np.fft.fft(res.pred + res.residuals))
                    fits = np.array(fits)
                    specs = np.array(specs)
                    ppm = test.system.basisObj.ppm
                    ppmlim = (0.5, 4.0)

                # other models (e.g. nn, ls)
                else:
                    fits = test.system.sigModel.forward(results[f'{key}_{scenario}']['preds']).cpu().numpy()
                    specs = results[f'{key}_{scenario}']['specs'].cpu().numpy()
                    specs = specs[:, 0] + 1j * specs[:, 1]  # convert to complex
                    noise = results[f'{key}_{scenario}']['noise'].cpu().numpy()
                    noise = noise[:, 0] + 1j * noise[:, 1]  # convert to complex
                    true = specs - noise  # ground truth
                    ppm = test.system.basisObj.ppm
                    ppmlim = (0.5, 4.0)

                for i in range(min(limit, specs.shape[0])):
                    # if true is not None: gt = true[i]
                    # else: gt = None
                    gt = None  # ground truth is not used in this case
                    plot_spec_and_fit(specs[i], fits[i], gt, ppm, ppmlim, f'fit{i}',
                                      save_path=config['path2save'] + f'/fits/{key}/{scenario}/', specOnly=True)


    #**********************#
    #   plot performance   #
    #**********************#
    if config['plot_performance']:
        print('Plotting performance for each scenario...')

        def plotPerformances(param, models, colors, results, scenario, idx, minR, maxR, xlabel, ylabel, bins=20):

            if 'baseline' in xlabel.lower(): xmin, xmax, bins = 600, 2600, 12
            elif 'frequency' in xlabel.lower(): xmin, xmax, bins = None, None, 12
            elif 'voigt' in xlabel.lower(): xmin, xmax = 4, None
            elif 'phase' in xlabel.lower(): xmin, xmax, bins = None, None, 16
            elif 'random' in xlabel.lower(): xmin, xmax, bins = 1e3, 5.5e5, 30
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
            # plt.xlim(xmin, xmax)
            plt.xlim(max(xmin, bin_edges[0]), min(xmax, bin_edges[-1]))
            plt.ylim(ymin, ymax)

            if config['path2save'] != '':
                if scenario.lower() == 'ood':
                    name = xlabel.split(' ')[0]
                    if not os.path.exists(config['path2save'] + f'/performances/{scenario}/'):
                        os.makedirs(config['path2save'] + f'/performances/{scenario}/')
                    plt.savefig(config['path2save'] + f'/performances/{scenario}/{name}.svg', bbox_inches='tight', dpi=300)
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
                                 'Baseline [a.u.]', 'MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]')

            elif scenario.lower() == 'snr':
                idx = len(test.system.basisObj.names) + 20   # dummy index
                # compute SNR
                signal = (results[f'{list(models.keys())[0]}_{scenario}']['specs'] - results[f'{list(models.keys())[0]}_{scenario}']['noise']).numpy()
                signal = signal[:, 0] + 1j * signal[:, 1]
                signal = signal[:, test.system.first:test.system.last]
                signal = np.sum(np.abs(signal) ** 2, axis=-1)

                noise = results[f'{list(models.keys())[0]}_{scenario}']['noise'].numpy()
                noise = noise[:, 0] + 1j * noise[:, 1]
                noise = noise[:, test.system.first:test.system.last]
                noise = np.sum(np.abs(noise) ** 2, axis=-1)

                snr = 10 * np.log10(signal / noise)
                print(np.min(snr), np.max(snr))

                plotPerformances(snr, models, colors, results, scenario, idx, 0, 40,
                                 'SNR [dB]', 'MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]')

            elif scenario.lower() == 'rw':
                # estimate the random walk parameters
                noise = results[f'{list(models.keys())[0]}_{scenario}']['noise'].numpy()
                noise = noise[:, :, test.system.first:test.system.last]
                est_scale = np.abs(np.max(noise, axis=-1) - np.min(noise, axis=-1))

                plotPerformances(est_scale[:, 0],  # imaginary part
                                 models, colors, results, scenario, idx, 0, 0,
                                 'Random Walk Scale [a.u.]',
                                 'MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]')

            elif scenario.lower() == 'ood':
                for idx, name in enumerate(test.system.basisObj.names):
                    minR, maxR = test.system.concs[name]['low_limit'], test.system.concs[name]['up_limit']
                    plotPerformances(results[f'{list(models.keys())[0]}_{scenario}']['truths'][:, idx].numpy(),
                                     models, colors, results, scenario, idx, minR, maxR,
                                     name + ' Concentration [mM]',
                                     'MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]')

            elif scenario in test.system.basisObj.names:  # metabolites
                idx = test.system.basisObj.names.index(scenario)
                minR, maxR = test.system.concs[scenario]['low_limit'], test.system.concs[scenario]['up_limit']
                scenario_name = scenario if scenario.lower() != 'mm_cmr' else 'MM'


                plotPerformances(results[f'{list(models.keys())[0]}_{scenario}']['truths'][:, idx].numpy(),
                                 models, colors, results, scenario, idx if scenario.lower() != 'mm_cmr' else idx + 20, minR, maxR,
                                 scenario_name + ' Concentration [mM]',
                                 'MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]')


    #*****************#
    #   plot ranges   #
    #*****************#
    if config['plot_ranges']:
        print('Plotting ranges for each scenario...')
        from visualizations.plotFunctionsErr import plotParamRanges

        limit = 200  # only for txt

        for snrPlot in [True, False]:
            for scenario in tqdm(scenarios):
                for key, model in models.items():

                    if snrPlot:
                        # compute SNR
                        signal = (results[f'{key}_{scenario}']['specs'] - results[f'{key}_{scenario}']['noise']).numpy()
                        signal = signal[:, 0] + 1j * signal[:, 1]
                        signal = signal[:, test.system.first:test.system.last]
                        signal = np.sum(np.abs(signal) ** 2, axis=-1)

                        noise = results[f'{key}_{scenario}']['noise'].numpy()
                        noise = noise[:, 0] + 1j * noise[:, 1]
                        noise = noise[:, test.system.first:test.system.last]
                        noise = np.sum(np.abs(noise) ** 2, axis=-1)
                        snr = 10 * np.log10(signal / noise)
                        cval = snr
                        cmin, cmax = -20, 60
                    else:
                        cval = None
                        cmin, cmax = None, None

                    if scenario.lower() in ['lor', 'gau', 'eps', 'phi']:
                        if scenario.lower() == 'lor':
                            idx = len(test.system.basisObj.names)
                            label = 'Linewidth (Lorentzian) [1/s]'
                            minR, maxR = test.system.ps['broadening'][0][0], test.system.ps['broadening'][1][0]
                            bins = 20
                        elif scenario.lower() == 'gau':
                            idx = len(test.system.basisObj.names) + 1
                            label = 'Linewidth (Gaussian) [1/s]'
                            minR, maxR = test.system.ps['broadening'][0][1], test.system.ps['broadening'][1][1]
                            bins = 20
                        elif scenario.lower() == 'eps':
                            idx = len(test.system.basisObj.names) + 2
                            label = 'Frequency [rad/s]'
                            minR, maxR = test.system.ps['shifting'][0], test.system.ps['shifting'][1]
                            bins = 20
                        elif scenario.lower() == 'phi':
                            idx = len(test.system.basisObj.names) + 3
                            label = 'Phase [rad]'
                            minR, maxR = test.system.ps['phi0'][0], test.system.ps['phi0'][1]
                            bins = 20
                        plotParamRanges(results[f'{key}_{scenario}']['truths'][:, idx].numpy(),
                                        results[f'{key}_{scenario}']['err'].mean(dim=-1).numpy(),
                                        c=cval, cmin=cmin, cmax=cmax,
                                        minR=minR, maxR=maxR, xLabel=label,
                                        yLabel='MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]',
                                        bins=bins, visR=key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel'])

                        # save x and y values for first couple of points
                        if config['path2save'] != '':
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
                                        minR=minR, maxR=maxR, xLabel='Voigt Linewidth [1/s]',
                                        yLabel='MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]',
                                        bins=20, visR=key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel'])

                        # save x and y values for first couple of points
                        if config['path2save'] != '':
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
                                        minR=minR, maxR=maxR, xLabel='Baseline [a.u.]',
                                        yLabel='MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]',
                                        bins=20, visR=key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel'])

                        # save x and y values for first couple of points
                        if config['path2save'] != '':
                            if not os.path.exists(config['path2save'] + f'/fits/{key}/{scenario}/'):
                                os.makedirs(config['path2save'] + f'/fits/{key}/{scenario}/')

                            np.savetxt(config['path2save'] + f'/fits/{key}/{scenario}/param_ranges.txt',
                                       np.column_stack((np.arange(limit, dtype=int),
                                                        results[f'{key}_{scenario}']['truths'][:limit, idx:].abs().mean(dim=-1).numpy(),
                                                        results[f'{key}_{scenario}']['err'].mean(dim=-1)[:limit].numpy())),
                                       header='Index, Baseline [a.u.], MAE [mM]',
                                       fmt='%.4f', delimiter=', ')

                    elif scenario.lower() == 'snr':
                        # compute SNR
                        signal = (results[f'{key}_{scenario}']['specs'] - results[f'{key}_{scenario}']['noise']).numpy()
                        signal = signal[:, 0] + 1j * signal[:, 1]
                        signal = signal[:, test.system.first:test.system.last]
                        signal = np.sum(np.abs(signal) ** 2, axis=-1)

                        noise = results[f'{key}_{scenario}']['noise'].numpy()
                        noise = noise[:, 0] + 1j * noise[:, 1]
                        noise = noise[:, test.system.first:test.system.last]
                        noise = np.sum(np.abs(noise) ** 2, axis=-1)

                        snr = 10 * np.log10(signal / noise)

                        err = results[f'{key}_{scenario}']['err'].mean(dim=-1).numpy()
                        plotParamRanges(snr, err, minR=0, maxR=40,  # hard estimate from testing range
                                        c=cval, cmin=cmin, cmax=cmax,
                                        xLabel='SNR [dB]', yLabel='MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]',
                                        bins=20, visR=key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel'])

                        # save x and y values for first couple of points
                        if config['path2save'] != '':
                            if not os.path.exists(config['path2save'] + f'/fits/{key}/{scenario}/'):
                                os.makedirs(config['path2save'] + f'/fits/{key}/{scenario}/')

                            np.savetxt(config['path2save'] + f'/fits/{key}/{scenario}/snr.txt',
                                       np.column_stack((np.arange(limit, dtype=int), snr[:limit], err[:limit])),
                                       header='Index, SNR [dB], MAE [mM]',
                                       fmt='%.4f', delimiter=', ')

                    elif scenario.lower() == 'rw':
                        # estimate the random walk parameters
                        noise = results[f'{key}_{scenario}']['noise'].numpy()
                        noise = noise[:, :, test.system.first:test.system.last]
                        est_scale = np.abs(np.max(noise, axis=-1) - np.min(noise, axis=-1))

                        plotParamRanges(est_scale[:, 0],   # imaginary part
                                        results[f'{key}_{scenario}']['err'].mean(dim=-1).numpy(),
                                        c=cval, cmin=cmin, cmax=cmax,
                                        minR=0, maxR=0, xLabel='Random Walk Scale [a.u.]',
                                        yLabel='MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]',
                                        bins=200, visR=key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel'])

                        # save x and y values for first couple of points
                        if config['path2save'] != '':
                            if not os.path.exists(config['path2save'] + f'/fits/{key}/{scenario}/'):
                                os.makedirs(config['path2save'] + f'/fits/{key}/{scenario}/')

                            np.savetxt(config['path2save'] + f'/fits/{key}/{scenario}/rw.txt',
                                       np.column_stack((np.arange(limit, dtype=int),
                                                        est_scale[:limit, 0],
                                                        results[f'{key}_{scenario}']['err'].mean(dim=-1)[:limit].numpy())),
                                       header='Index, Random Walk Scale [a.u.], MAE [mM]',
                                       fmt='%.4f', delimiter=', ')

                    elif scenario in test.system.basisObj.names:  # metabolites
                        idx = test.system.basisObj.names.index(scenario)
                        minR, maxR = test.system.concs[scenario]['low_limit'], test.system.concs[scenario]['up_limit']

                        met = results[f'{key}_{scenario}']['truths'][:, idx].numpy()
                        if scenario.lower() == 'mm_cmr': err = results[f'{key}_{scenario}']['err'].mean(dim=-1).numpy()
                        else: err = results[f'{key}_{scenario}']['err'][:, idx].numpy()

                        plotParamRanges(met, err, minR=minR, maxR=maxR, bins=20,
                                        c=cval, cmin=cmin, cmax=cmax,
                                        xLabel= scenario + ' Concentration [mM]' if scenario.lower() != 'mm_cmr' else 'MM Concentration [mM]',
                                        yLabel='MAE [mM]' if scenario.lower() == 'mm_cmr' and config['scale'] == 'off' else 'MOSAE [mM]',
                                        visR=key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel'])

                        # save x and y values for first couple of points
                        if config['path2save'] != '':
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
                            met = results[f'{key}_{scenario}']['truths'][:, idx].numpy()
                            err = results[f'{key}_{scenario}']['err'][:, idx].numpy()

                            minR, maxR = test.system.concs[name]['low_limit'], test.system.concs[name]['up_limit']
                            plotParamRanges(met, err, minR=minR, maxR=maxR, bins=20,
                                            c=cval, cmin=cmin, cmax=cmax,
                                            xLabel=name + ' Concentration [mM]',
                                            yLabel='MAE [mM]' if config['scale'] == 'off' else 'MOSAE [mM]',
                                            visR=key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel'])

                            # save x and y values for first couple of points
                            if config['path2save'] != '':
                                if not os.path.exists(config['path2save'] + f'/fits/{key}/{scenario}/'):
                                    os.makedirs(config['path2save'] + f'/fits/{key}/{scenario}/')

                                np.savetxt(config['path2save'] + f'/fits/{key}/{scenario}/{name}_param_ranges.txt',
                                           np.column_stack((np.arange(limit, dtype=int),
                                                            results[f'{key}_{scenario}']['truths'][:limit, idx].numpy(),
                                                            results[f'{key}_{scenario}']['err'][:limit, idx].numpy())),
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


    #************************#
    #   plot distributions   #
    #************************#
    if config['plot_distributions']:
        print('Plotting distributions for metabolites...')
        from visualizations.plotFunctionsErr import scatterHist

        for scenario in tqdm(scenarios):
            for key, model in models.items():
                if scenario in test.system.basisObj.names + ['ID', 'OOD']:

                        # draw the scatter plot and the histograms
                        for idx, name in enumerate(test.system.basisObj.names):

                            if scenario not in ['ID', 'OOD'] and name != scenario: continue

                            gt_concs = results[f'{key}_{scenario}']['truths'][:, idx].numpy()
                            est_concs = results[f'{key}_{scenario}']['scale_preds'][:, idx].numpy()
                            errors = results[f'{key}_{scenario}']['err'][:, idx].numpy()

                            # add out-of-distribution lines
                            if (config['ood_lines'] and key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel']
                                    and scenario.lower() == 'ood'):
                                from simulation.simulationDefs import aumcConcs, aumcConcsMS

                                minR, maxR = aumcConcs[name]['low_limit'], aumcConcs[name]['up_limit']
                                ax, ax_histx, ax_histy = scatterHist(gt_concs, est_concs,
                                                                     xLabel='True Concentration [mM]',
                                                                     yLabel='Estimated Concentration [mM]',
                                                                     minR=minR, maxR=maxR, bins=100, name=name)

                                minR, maxR = aumcConcsMS[name]['low_limit'], aumcConcsMS[name]['up_limit']

                                # define OoD region color and line style
                                ood_color = '#ffcccc'  # light red
                                ood_line_color = '#d62728'  # strong red
                                ood_line_style = (0, (1, 2))  # dotted

                                # draw OoD lines (dotted red) on the scatter
                                ax.plot([np.min(gt_concs), np.max(gt_concs)], [minR, minR], color=ood_line_color,
                                        linestyle=ood_line_style, linewidth=1.5)
                                ax.plot([np.min(gt_concs), np.max(gt_concs)], [maxR, maxR], color=ood_line_color,
                                        linestyle=ood_line_style, linewidth=1.5)

                                # shade OoD regions on y-axis (scatter)
                                ax.fill_between([np.min(gt_concs), np.max(gt_concs)], np.min(gt_concs), minR,
                                                color=ood_color, alpha=0.5, zorder=0)
                                ax.fill_between([np.min(gt_concs), np.max(gt_concs)], maxR, np.max(gt_concs),
                                                color=ood_color, alpha=0.5, zorder=0)

                                # shade OoD regions on x-axis (histogram of predictions)
                                maxHist = np.max(ax_histy.get_xlim())
                                ax_histy.plot([minR, maxHist], [minR, minR], color=ood_line_color,
                                              linestyle=ood_line_style, linewidth=1.5)
                                ax_histy.plot([minR, maxHist], [maxR, maxR], color=ood_line_color,
                                              linestyle=ood_line_style, linewidth=1.5)
                                ax_histy.fill_between([minR, maxHist], np.min(gt_concs), minR,
                                                      color=ood_color, alpha=0.5, zorder=0)
                                ax_histy.fill_between([minR, maxHist], maxR, np.max(gt_concs),
                                                      color=ood_color, alpha=0.5, zorder=0)
                            else:
                                minR, maxR = test.system.concs[name]['low_limit'], test.system.concs[name]['up_limit']
                                ax, ax_histx, ax_histy = scatterHist(gt_concs, est_concs,
                                                                     xLabel='True Concentration [mM]',
                                                                     yLabel='Estimated Concentration [mM]',
                                                                     minR=minR, maxR=maxR,
                                                                     c=errors, cmin=0, cmax=5, bins=100, name=name)

                            if config['path2save']!= '':
                                if scenario == name:
                                    if not os.path.exists(config['path2save'] + f'/dists/{key}/{name}/'):
                                        os.makedirs(config['path2save'] + f'/dists/{key}/{name}/')
                                    plt.savefig(config['path2save'] + f'/dists/{key}/{name}/{name}.svg',
                                                bbox_inches='tight', dpi=300)
                                elif scenario == 'OOD':
                                    if not os.path.exists(config['path2save'] + f'/dists/{key}/OOD/'):
                                        os.makedirs(config['path2save'] + f'/dists/{key}/OOD')
                                    plt.savefig(config['path2save'] + f'/dists/{key}/OOD/{name}.svg',
                                                bbox_inches='tight', dpi=300)
                                else:
                                    if not os.path.exists(config['path2save'] + f'/dists/{key}/ID/'):
                                        os.makedirs(config['path2save'] + f'/dists/{key}/ID')
                                    plt.savefig(config['path2save'] + f'/dists/{key}/ID/{scenario}_{name}.svg',
                                                bbox_inches='tight', dpi=300)

                else:
                    
                    if scenario.lower() == 'bas':
                        cval = results[f'{key}_{scenario}']['truths'][:, idx:].abs().mean(dim=-1).numpy()
                        scen = 'Baseline'
                        cmin, cmax = None, None

                    elif scenario.lower() == 'snr':
                        # compute SNR
                        signal = (results[f'{key}_{scenario}']['specs'] - results[f'{key}_{scenario}']['noise']).numpy()
                        signal = signal[:, 0] + 1j * signal[:, 1]
                        signal = np.fft.fft(signal)[:, test.system.first:test.system.last]
                        signal = np.sum(np.abs(signal) ** 2, axis=-1)

                        noise = results[f'{key}_{scenario}']['noise'].numpy()
                        noise = noise[:, 0] + 1j * noise[:, 1]
                        noise = np.fft.fft(noise)[:, test.system.first:test.system.last]
                        noise = np.sum(np.abs(noise) ** 2, axis=-1)

                        cval = 10 * np.log10(signal / noise)
                        scen = 'SNR'
                        cmin, cmax = None, None

                    elif scenario.lower() == 'rw':
                        # estimate the random walk parameters
                        noise = results[f'{key}_{scenario}']['noise'].numpy()
                        noise = noise[:, 0, test.system.first:test.system.last]
                        cval = np.abs(np.max(noise, axis=-1) - np.min(noise, axis=-1))
                        scen = 'Random_Walk'
                        cmin, cmax = np.min(cval), np.max(cval) / 10

                    elif scenario.lower() == 'noise':
                        cval = np.var(results[f'{key}_{scenario}']['noise'].numpy(), axis=-1)[:, 0]
                        scen = 'Noise Variance'
                        cmin, cmax = None, None

                    else:
                        if scenario.lower() == 'lor+gau':
                            lor = results[f'{key}_{scenario}']['truths'][:, len(test.system.basisObj.names)].numpy()
                            gau = results[f'{key}_{scenario}']['truths'][:, len(test.system.basisObj.names) + 1].numpy()
                            cval = lor + gau
                            scen = 'Voigt'
                            cmin, cmax = None, None
                        else:
                            if scenario.lower() == 'lor':
                                idx = len(test.system.basisObj.names)
                                scen = 'Lorentzian'
                            elif scenario.lower() == 'gau':
                                idx = len(test.system.basisObj.names) + 1
                                scen = 'Gaussian'
                            elif scenario.lower() == 'eps':
                                idx = len(test.system.basisObj.names) + 2
                                scen = 'Frequency'
                            elif scenario.lower() == 'phi':
                                idx = len(test.system.basisObj.names) + 3
                                scen = 'Phase'
                            cval =  results[f'{key}_{scenario}']['truths'][:, idx].numpy()
                            cmin, cmax = None, None

                    for idx, name in enumerate(test.system.basisObj.names):
                        gt_concs = results[f'{key}_{scenario}']['truths'][:, idx].numpy()
                        est_concs = results[f'{key}_{scenario}']['scale_preds'][:, idx].numpy()
                        minR, maxR = test.system.concs[name]['low_limit'], test.system.concs[name]['up_limit']

                        ax, ax_histx, ax_histy = scatterHist(gt_concs, est_concs,
                                                             xLabel='True Concentration [mM]',
                                                             yLabel='Estimated Concentration [mM]',
                                                             c=cval, cmin=cmin, cmax=cmax, bins=100,
                                                             minR=minR, maxR=maxR,
                                                             name=name, lines=scenario.lower() != 'noise')

                        if config['path2save']!= '':
                            if not os.path.exists(config['path2save'] + f'/dists/{key}/{scen}/'):
                                os.makedirs(config['path2save'] + f'/dists/{key}/{scen}/')
                            plt.savefig(config['path2save'] + f'/dists/{key}/{scen}/{scenario}_{name}.svg',
                                        bbox_inches='tight', dpi=300)


    #****************************#
    #   plot distributions snr   #
    #****************************#
    if config['plot_distributions']:
        print('Plotting distributions for metabolites with SNR cmap...')
        from visualizations.plotFunctionsErr import scatterHist

        for scenario in tqdm(scenarios):
            for key, model in models.items():
                if scenario in test.system.basisObj.names + ['ID', 'OOD']:

                        # draw the scatter plot and the histograms
                        for idx, name in enumerate(test.system.basisObj.names):

                            if scenario not in ['ID', 'OOD'] and name != scenario: continue

                            gt_concs = results[f'{key}_{scenario}']['truths'][:, idx].numpy()
                            est_concs = results[f'{key}_{scenario}']['scale_preds'][:, idx].numpy()
                            errors = results[f'{key}_{scenario}']['err'][:, idx].numpy()

                            # compute SNR
                            signal = (results[f'{key}_{scenario}']['specs'] - results[f'{key}_{scenario}']['noise']).numpy()
                            signal = signal[:, 0] + 1j * signal[:, 1]
                            signal = np.fft.fft(signal)[:, test.system.first:test.system.last]
                            signal = np.sum(np.abs(signal) ** 2, axis=-1)

                            noise = results[f'{key}_{scenario}']['noise'].numpy()
                            noise = noise[:, 0] + 1j * noise[:, 1]
                            noise = np.fft.fft(noise)[:, test.system.first:test.system.last]
                            noise = np.sum(np.abs(noise) ** 2, axis=-1)

                            cval = 10 * np.log10(signal / noise)
                            scen = 'SNR'
                            cmin, cmax = 0, 40
                                # print(np.min(cval), np.max(cval))

                            # add out-of-distribution lines
                            if (config['ood_lines'] and key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel']
                                    and scenario.lower() == 'ood'):
                                minR, maxR = aumcConcs[name]['low_limit'], aumcConcs[name]['up_limit']
                                ax, ax_histx, ax_histy = scatterHist(gt_concs, est_concs,
                                                                     xLabel='True Concentration [mM]',
                                                                     yLabel='Estimated Concentration [mM]',
                                                                     minR=minR, maxR=maxR,
                                                                     c=cval, cmin=cmin, cmax=cmax, bins=100, name=name)

                                minR, maxR = aumcConcsMS[name]['low_limit'], aumcConcsMS[name]['up_limit']

                                # define OoD region color and line style
                                ood_color = '#ffcccc'  # light red
                                ood_line_color = '#d62728'  # strong red
                                ood_line_style = (0, (1, 2))  # dotted

                                min_y, max_y = ax.get_ylim()

                                # draw OoD lines (dotted red) on the scatter
                                ax.plot([np.min(gt_concs), np.max(gt_concs)], [minR, minR], color=ood_line_color,
                                        linestyle=ood_line_style, linewidth=1.5)
                                ax.plot([np.min(gt_concs), np.max(gt_concs)], [maxR, maxR], color=ood_line_color,
                                        linestyle=ood_line_style, linewidth=1.5)

                                # shade OoD regions on y-axis (scatter)
                                ax.fill_between([np.min(gt_concs), np.max(gt_concs)], min_y, minR,
                                                color=ood_color, alpha=0.5, zorder=0)
                                ax.fill_between([np.min(gt_concs), np.max(gt_concs)], maxR, max_y,
                                                color=ood_color, alpha=0.5, zorder=0)

                                # shade OoD regions on x-axis (histogram of predictions)
                                maxHist = np.max(ax_histy.get_xlim())
                                ax_histy.plot([minR, maxHist], [minR, minR], color=ood_line_color,
                                              linestyle=ood_line_style, linewidth=1.5)
                                ax_histy.plot([minR, maxHist], [maxR, maxR], color=ood_line_color,
                                              linestyle=ood_line_style, linewidth=1.5)
                                ax_histy.fill_between([minR, maxHist], min_y, minR,
                                                      color=ood_color, alpha=0.5, zorder=0)
                                ax_histy.fill_between([minR, maxHist], maxR, max_y,
                                                      color=ood_color, alpha=0.5, zorder=0)
                            else:
                                if scenario.lower() == 'ood':
                                    minR, maxR = aumcConcs[name]['low_limit'], aumcConcs[name]['up_limit']
                                else:
                                    minR, maxR = test.system.concs[name]['low_limit'], test.system.concs[name]['up_limit']
                                ax, ax_histx, ax_histy = scatterHist(gt_concs, est_concs,
                                                                     xLabel='True Concentration [mM]',
                                                                     yLabel='Estimated Concentration [mM]',
                                                                     minR=minR, maxR=maxR,
                                                                     c=cval, cmin=cmin, cmax=cmax, bins=100, name=name)

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


    if config['plot_all_in_one']:
        print('Plotting all metabolites in one scatter plot...')
        from visualizations.plotFunctionsErr import plotAllInOneScatter

        for scenario in tqdm(scenarios):
            for key, model in models.items():
                gt_concs = results[f'{key}_{scenario}']['truths'].numpy()
                est_concs = results[f'{key}_{scenario}']['scale_preds'].numpy()

                # compute SNR
                signal = (results[f'{key}_{scenario}']['specs'] - results[f'{key}_{scenario}']['noise']).numpy()
                signal = signal[:, 0] + 1j * signal[:, 1]
                signal = np.fft.fft(signal)[:, test.system.first:test.system.last]
                signal = np.sum(np.abs(signal) ** 2, axis=-1)

                noise = results[f'{key}_{scenario}']['noise'].numpy()
                noise = noise[:, 0] + 1j * noise[:, 1]
                noise = np.fft.fft(noise)[:, test.system.first:test.system.last]
                noise = np.sum(np.abs(noise) ** 2, axis=-1)

                cval = 10 * np.log10(signal / noise)
                cmin, cmax = 0, 40

                # remove MMs
                mm_idx = test.system.basisObj.names.index('MM_CMR')
                gt_concs = np.delete(gt_concs, mm_idx, axis=1)
                est_concs = np.delete(est_concs, mm_idx, axis=1)
                names = np.delete(test.system.basisObj.names, mm_idx)

                plotAllInOneScatter(gt_concs, est_concs, names, c=cval,
                                    cmin=cmin, cmax=cmax, xLabel='True Concentration [mM]',
                                    yLabel='Estimated Concentration [mM]', xplot=5, yplot=4)
                if config['path2save'] != '':
                    if not os.path.exists(config['path2save'] + f'/all_in_one/{key}/'):
                        os.makedirs(config['path2save'] + f'/all_in_one/{key}/')
                    plt.savefig(config['path2save'] + f'/all_in_one/{key}/{scenario}.png',
                                bbox_inches='tight', dpi=300)


    if config['plot_histograms']:
        print('Plotting histograms for metabolites...')

        import matplotlib.cm as cm
        import matplotlib.colors as colors

        def plotHistograms(x, y, c=None, minR=None, maxR=None, name=None):

            # ignore nan values
            mask = ~np.isnan(x) & ~np.isnan(y)
            x, y = x[mask], y[mask]
            if c is not None: c = c[mask]

            # filter out large outliers for non-MM metabolites
            if 'MM' not in name:
                idx = np.where(y < 20)[0]
                x, y = x[idx], y[idx]
                if c is not None: c = c[idx]

            # compute full plotting range
            low = -0.1 # min(np.min(y), np.min(x), minR) - 0.1 * max(np.max(y), np.max(x), maxR)
            high = 1.7 # max(np.max(y), np.max(x), maxR) * 1.1
            bins = np.linspace(low, high, 100)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            hist, _ = np.histogram(y, bins=bins)

            fig, ax = plt.subplots(figsize=(3, 1.75))

            # color histogram by local noise variance
            if c is not None:
                norm = colors.Normalize(vmin=np.percentile(c, 5), vmax=np.percentile(c, 95))
                sm = cm.ScalarMappable(norm=norm, cmap='plasma')
                for i in range(len(bins) - 1):
                    bin_mask = (y >= bins[i]) & (y < bins[i + 1])
                    if np.any(bin_mask):
                        avg_c = np.mean(c[bin_mask])
                        bar_color = sm.to_rgba(avg_c)
                    else:
                        bar_color = 'lightgray'
                    ax.bar(bin_centers[i], hist[i], width=bins[1] - bins[0], color=bar_color, edgecolor='black')
            else:
                ax.hist(y, bins=bins, color='blue', edgecolor='black')

            # ground truth line
            ax.axvline(0, color='green', linestyle='--', label='Ground Truth')

            # # training range lines
            # if minR is not None and maxR is not None:
            #     ax.axvline(minR, color='gray', linestyle='-', label='Training Range')
            #     ax.axvline(maxR, color='gray', linestyle='-')

            # ax.set_title(f'{name} Estimate Histogram')
            ax.set_xlabel('Estimated Concentration [mM]')
            # ax.set_ylabel('Frequency')
            # ax.legend()
            ax.set_xlim([low, high])

            # no y-axis ticks
            ax.tick_params(axis='y', length=0, width=0)  # or turn it off
            ax.set_yticklabels([])

            # format x-axis ticks
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))

            plt.tight_layout()
            return ax


        for scenario in tqdm(scenarios):
            for key, model in models.items():
                if scenario.lower() == 'noise':
                    cval = np.std(results[f'{key}_{scenario}']['noise'].numpy(), axis=-1)[:, 0]
                    for idx, name in enumerate(test.system.basisObj.names):
                        gt_concs = results[f'{key}_{scenario}']['truths'][:, idx].numpy()
                        est_concs = results[f'{key}_{scenario}']['scale_preds'][:, idx].numpy()

                        minR, maxR = test.system.concs[name]['low_limit'], test.system.concs[name]['up_limit']

                        ax = plotHistograms(gt_concs, est_concs, c=cval, minR=minR, maxR=maxR, name=name)

                        if config['ood_lines'] and key not in ['own_lcm', 'own_lcm_gd', 'newton', 'mh', 'lcmodel']:
                            # define OoD region color and line style
                            ood_color = '#ffcccc'  # light red
                            ood_line_color = '#d62728'  # strong red
                            ood_line_style = (0, (1, 2))  # dotted

                            ax.axvline(minR, color=ood_line_color, linestyle=ood_line_style, linewidth=1.5)
                            ax.axvline(maxR, color=ood_line_color, linestyle=ood_line_style, linewidth=1.5)

                            # get y-limits of the current axes (height of the histogram)
                            ymin, ymax = ax.get_ylim()

                            # shade left OoD region (x < minR)
                            ax.fill_betweenx([ymin, ymax], x1=ax.get_xlim()[0], x2=minR,
                                             color=ood_color, alpha=0.5, zorder=0)

                            # shade right OoD region (x > maxR)
                            ax.fill_betweenx([ymin, ymax], x1=maxR, x2=ax.get_xlim()[1],
                                             color=ood_color, alpha=0.5, zorder=0)


                        if config['path2save'] != '':
                            if not os.path.exists(config['path2save'] + f'/histograms/{key}/{scenario}/'):
                                os.makedirs(config['path2save'] + f'/histograms/{key}/{scenario}/')
                            plt.savefig(config['path2save'] + f'/histograms/{key}/{scenario}/{name}.svg',
                                        bbox_inches='tight', dpi=300)


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
        mod_names = [model_names[key] for key in models.keys()]

        for key, model in models.items():

            # compute mape
            con_matrix = np.stack([results[f'{key}_{scenario}']['truths']
                                   for scenario in scenarios if scenario not in ['Noise']], axis=0)
            pred_matrix = np.stack([results[f'{key}_{scenario}']['scale_preds']
                                    for scenario in scenarios if scenario not in ['Noise']], axis=0)
            con_matrix = con_matrix[..., :pred_matrix.shape[-1]]
            pred_matrix = pred_matrix[..., :con_matrix.shape[-1]]

            err_matrix_mean = np.abs((con_matrix - pred_matrix) / (con_matrix + 1e-10)) * 100  # MAPE
            err_matrix_mean = np.clip(err_matrix_mean, 0, 999)[..., :test.system.basisObj.n_metabs]
            err_matrix_mean = np.nanmean(err_matrix_mean, axis=1)

            # create a heatmap
            plt.figure(figsize=(16, 6))
            sns.heatmap(err_matrix_mean, xticklabels=test.system.basisObj.names, yticklabels=scen_names,
                        cmap="Reds", annot=True, fmt=".1f")
            plt.title(f"Error heatmap (MAPE) for model {key}")
            # plt.xlabel("Metabolites")
            # plt.ylabel("Scenarios")
            plt.tight_layout()

            if config['path2save'] != '':
                if not os.path.exists(config['path2save'] + f'/heatmaps/{key}/'):
                    os.makedirs(config['path2save'] + f'/heatmaps/{key}/')
                plt.savefig(config['path2save'] + f'/heatmaps/{key}/heatmap.svg', bbox_inches='tight', dpi=300)


        for scenario in scenarios:
            if scenario.lower() == 'noise':
                continue

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
                        cmap="Purples", annot=True, fmt=".1f", cbar=False, vmin=0, vmax=100)
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
                        cmap="Purples", annot=True, fmt=".1f", cbar=False, vmin=0, vmax=100)
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
                        fmt=".1f", cbar=False, vmin=0, vmax=10)
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
            pred_matrix = np.stack([results[f'{key}_{scenario}']['scale_preds'][..., :test.system.basisObj.n_metabs]
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
                plt.savefig(config['path2save'] + f'/heatmaps_mae/{scenario}/heatmap.svg', bbox_inches='tight', dpi=300)


            # create heatmap for certain metbas
            # mets = ['Cr', 'Glu', 'GSH', 'mIns', 'Lac', 'NAA', 'PCh', 'PCr']
            mets = ['Cr', 'GABA', 'Gln', 'Glu', 'GPC', 'GSH', 'mIns', 'Lac', 'NAAG', 'NAA', 'PCh', 'PCr', 'Scyllo']
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
                plt.savefig(config['path2save'] + f'/heatmaps_mae/{scenario}/heatmap_few_mets.svg', bbox_inches='tight', dpi=300)


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
                plt.savefig(config['path2save'] + f'/heatmaps_mae/{scenario}/heatmap_sum_mets.svg', bbox_inches='tight', dpi=300)
