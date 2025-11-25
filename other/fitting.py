####################################################################################################
#                                            fitting.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 25/08/24                                                                                #
#                                                                                                  #
# Purpose: Model based fitting of MRS data.                                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutup; shutup.please()   # shut up warnings

from fsl_mrs.utils.mrs_io import read_FID

from shutil import copy

from tqdm import tqdm

# own
from frameworks.frameworkFSL import FrameworkFSL
from frameworks.frameworkLCM import FrameworkLCModel
from other.inVivoDataModule import BrainBeatsDataModule


#****************#
#   initialize   #
#****************#
config = {
        # 'path2basis': '../Data/BasisSets/FOCI_slaser_basis_MMshift_Insshift/',
        'path2basis': '../Data/BasisSets/FOCI_slaser_basis_MMshift_Insshift.basis',
        # 'path2basis': '../Data/BasisSets/7T_sLASER_OIT_TE34.basis',

        'path2data': '../../raw/PARREC_physlog_raw/',   # path to the data
        # 'path2frac': '../../LCModel_output/',   # path to the tissue fractions

        'path2exec': '~/lcmodel-6.3-1N/bin/lcmodel',  # LCModel executable path
        'path2cntrl': '../Data/Other/BrainBeats.control',   # LCModel control file

        'save_path': '../Data/BrainBeatsCCA/',

        'process': 'custom',   # 'custom', 'fsl-mrs'
        'coil_comb': 'adaptive',   # 'adaptive', 'simple', 'fls-mrs'

        'method': 'LCModel',   # 'LCModel', 'MH', 'Newton'
        'ppmlim': (0.5, 4.0),

        'sample_points': 1024,
        'cf': 298029903/1E6,
        'bw': 3000,
        'TE': 0.036,
        'TR': 5.0,

        'plot_dir': '../Imgs/BrainBeatsCCA/',

        'filter_out': ['1004', '1070', '1073'],   # filter out scans

        # stages
        'mat2nifti': False,
        'sdat2nifti': False,
        'labraw2nifti': True,
        '2blocks': True,
        'fitting': True,
    }



#*************#
#   running   #
#*************#
if __name__ == '__main__':

    # initialize fitting model
    if config['method'].lower() == 'lcmodel':
        lcm = FrameworkLCModel(config['path2basis'], ppmlim=config['ppmlim'], 
                               control=config['path2cntrl'],
                               sample_points=config['sample_points'], 
                               path2exec=config['path2exec'])
    elif config['method'].lower() == 'newton' or config['method'].lower() == 'mh':
        lcm = FrameworkFSL(config['path2basis'], method=config['method'], 
                           ppmlim=config['ppmlim'], bandwidth=config['bw'],
                           TE=config['TE'], TR=config['TR'], 
                           sample_points=config['sample_points'])
    else: 
        raise ValueError('Method not recognized')

    # initialize data module
    BB = BrainBeatsDataModule()

    # data conversion
    if config['mat2nifti'] or config['sdat2nifti'] or config['labraw2nifti']:
        if config['mat2nifti']: func = BB.mat2nifti
        elif config['sdat2nifti']: func = BB.sdat2nifti
        elif config['labraw2nifti']: func = BB.labraw2nifti
        else: raise ValueError('Conversion method not recognized')

        print('Converting the data to nifti...')
        func(config['path2data'], config['bw'], config['cf'],
             config['TE'], config['TR'], config['sample_points'],
             config['save_path'])
        print('Done.')
        config['path2data'] = config['save_path']

    # create blocks
    if config['2blocks']:   # data processing
        print('Creating blocks...')
        BB.write_blocks(config['path2data'], config['save_path'], config['process'], config['coil_comb'])
        print('Done.')

    # quantification with fitting method
    if config['fitting']:
        print('Fitting...')
        scan = {1: 'PreS', 2: 'PostS', 3: 'PostS', 4: 'PreS', 5: 'PostS2', 6: 'PostS2'}
        for item in tqdm(os.listdir(config['save_path'])):
            # loop through blocks 1-6
            for i in range(1, 7):
                path = f'{config["save_path"]}/{item}/block{i}'
                if not os.path.exists(path): continue   # check if folder exisits

                # load data
                try:
                    x = read_FID(f'{path}/preproc/metab.nii.gz')[:, 0, 0]
                    x_ref = read_FID(f'{path}/preproc/wref.nii.gz')[:, 0, 0]
                except:
                    print(f"Issues with reading file at {path}! Skipping...")
                    continue

                # load tissues
                if 'path2frac' in config:
                    try:
                        fracFile = open(f'{config["path2frac"]}/scan_{item}/voxel_fractions_{scan[i]}.txt').readlines()
                        frac = [{fracFile[0].split(',')[t]: float(f) for t, f in enumerate(fracFile[1].split(',')) if t < 3 }]
                    except:
                        print(f'Issues with tissue fractions: {config["path2frac"]}/scan_{item}/voxel_fractions_{scan[i]}.txt')
                        continue
                else: frac = None

                # to frequency domain (stack real and imaginary part) <- this is my convention
                x = np.fft.fft(x, axis=-1)
                x = np.stack((x.real, x.imag), axis=1)

                # quantify
                lcm.save_path = f'{path}/{config["method"].lower()}/'
                concs, crlbs = lcm(x, x_ref, frac)
        print('Done.')

    print('Load results and visualize...')
    if config['method'].lower() == 'newton' or config['method'].lower() == 'mh':

        # get signal model paramter ranges
        res1, res2, res3, res4, res5, res6 = [], [], [], [], [], []
        all_res = [res1, res2, res3, res4, res5, res6]

        conc1, conc2, conc3, conc4, conc5, conc6 = [], [], [], [], [], []
        all_concs = [conc1, conc2, conc3, conc4, conc5, conc6]

        for item in tqdm(os.listdir(config['save_path'])):
            # filter out scans
            for filter_out in config['filter_out']:
                if filter_out in item: break
            else:
                # loop through blocks 1-6
                for i in range(6):
                    path = f'{config["save_path"]}/{item}/block{i+1}/'
                    try:
                        with open(f'{path}/{config["method"]}/opt0.pkl', 'rb') as f:
                            opt = pickle.load(f)
                            names = opt.params_names
                            metab_names = names[:opt.numMetabs]
                        all_res[i].append(opt.params)
                        all_concs[i].append(opt.getConc(scaling='molarity'))
                    except:
                        print(f"Issues with reading file at {path}! Skipping...")
                        continue

                    # save all specs and fits for quality check in one folder
                    if not os.path.exists(f'{config["plot_dir"]}/QualityCheck/{config["method"]}/'):
                        os.makedirs((f'{config["plot_dir"]}/QualityCheck/{config["method"]}/'))

                    copy(f'{path}/{config["method"]}/fit0.png',
                         f'{config["plot_dir"]}/QualityCheck/{config["method"]}/{item}_block{i+1}.png')
        
        # visualize ranges
        for k, name in enumerate(names):
            plt.figure()
            for l, res in enumerate(all_res):
                res = np.array(res)
                plt.scatter(np.arange(1, res.shape[0] + 1), res[:, k], label=f'block {l}')
            plt.legend()
            plt.xlabel('Samples')
            plt.ylabel(name)

            if 'plot_dir' in config:
                if not os.path.exists(f'{config["plot_dir"]}/ParamRanges/{config["method"]}/'): 
                    os.makedirs(f'{config["plot_dir"]}/ParamRanges/{config["method"]}/')
                plt.savefig(f'{config["plot_dir"]}/ParamRanges/{config["method"]}/{name}_range.png', 
                            bbox_inches='tight', dpi=300, transparent=True)

        # visualize concentrations
        for k, name in enumerate(metab_names):
            plt.figure()
            for l, conc in enumerate(all_concs):
                conc = np.array(conc)
                plt.scatter(np.arange(1, conc.shape[0] + 1), conc[:, k], label=f'block {l}')
            plt.legend()
            plt.xlabel('Samples')
            plt.ylabel(name)

            if 'plot_dir' in config:
                if not os.path.exists(f'{config["plot_dir"]}/Concentrations/{config["method"]}/'): 
                    os.makedirs(f'{config["plot_dir"]}/Concentrations/{config["method"]}/')
                plt.savefig(f'{config["plot_dir"]}/Concentrations/{config["method"]}/{name}_concs.png', 
                            bbox_inches='tight', dpi=300, transparent=True)
        
        if 'plot_dir' not in config: plt.show()
    

    elif config['method'].lower() == 'lcmodel':
        # save all specs and fits for quality check in one folder
        def plot_LCModel_fit(fit):
            plt.figure()
            plt.plot(fit['ppm'], fit['data'], 'k', label='Data', linewidth=1)
            plt.plot(fit['ppm'], fit['completeFit'], 'r', label='Fit', alpha=0.6, linewidth=2)
            plt.plot(fit['ppm'], fit['data'] - fit['completeFit'] + 1.1 * np.max(fit['data']),
                     'k', label='Residual', alpha=0.8, linewidth=1)
            plt.xlabel('Chemical Shift [ppm]')
            plt.gca().invert_xaxis()

        for item in tqdm(os.listdir(config['save_path'])):
            # filter out scans
            for filter_out in config['filter_out']:
                if filter_out in item: break
            else:
                # loop through blocks 1-6
                for i in range(6):
                    path = f'{config["save_path"]}/{item}/block{i+1}/{config["method"]}/'
                    try:
                        fit = lcm.read_LCModel_fit(path + 'temp0.coord')
                        plot_LCModel_fit(fit)
                    except:
                        print(f"Issues with reading file at {path}! Skipping...")
                        continue

                    # save all specs and fits for quality check in one folder
                    if 'plot_dir' in config:
                        if not os.path.exists(f'{config["plot_dir"]}/QualityCheck/{config["method"]}/'):
                            os.makedirs((f'{config["plot_dir"]}/QualityCheck/{config["method"]}/'))
                        plt.savefig(f'{config["plot_dir"]}/QualityCheck/{config["method"]}/'
                                    f'{item}_block{i+1}.png', dpi=300, bbox_inches='tight')

        # load concs and crlbs and visualize ranges
        res1, res2, res3, res4, res5, res6 = [], [], [], [], [], []
        all_res = [res1, res2, res3, res4, res5, res6]
        for item in tqdm(os.listdir(config['save_path'])):
            # filter out scans
            for filter_out in config['filter_out']:
                if filter_out in item: break
            else:
                # loop through block
                for i in range(6):
                    path = f'{config["save_path"]}/{item}/block{i+1}/{config["method"]}/'
                    # try:
                    try:
                        metabs, concs, crlbs, tcr, fwhm, snr, shift, phase = lcm.read_LCModel_coord(path + 'temp0.coord')
                        all_res[i].append(concs)
                    except:
                        print(f"Issues with reading file at {path}! Skipping...")
                        continue

        # visualize ranges
        for k, name in enumerate(metabs):
            plt.figure()
            for l, res in enumerate(all_res):
                res = np.array(res)
                plt.scatter(np.arange(1, res.shape[0] + 1), res[:, k], label=f'block {l}')
            plt.legend()
            plt.xlabel('Samples')
            plt.ylabel(name)

            if 'plot_dir' in config:
                if not os.path.exists(f'{config["plot_dir"]}/Concentrations/{config["method"]}/'):
                    os.makedirs(f'{config["plot_dir"]}/Concentrations/{config["method"]}/')
                plt.savefig(f'{config["plot_dir"]}/Concentrations/{config["method"]}/{name}_range.png',
                            bbox_inches='tight', dpi=300, transparent=True)


    
    