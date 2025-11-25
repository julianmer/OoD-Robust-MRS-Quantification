####################################################################################################
#                                           modelDefs.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 22/07/25                                                                                #
#                                                                                                  #
# Purpose: Definition of model configurations for the framework.                                   #
#                                                                                                  #
####################################################################################################


##*******************************************#
#   fet dictionary of model configurations   #
#********************************************#
def get_models(config, checkpoint_paths):
    """
    Returns a dictionary of model configurations based on the provided base
    configuration and checkpoint paths.

    @param config -- Base configuration dictionary containing common parameters.
    @param checkpoint_paths -- Dictionary mapping model specifications to their checkpoint paths.

    @return models -- Dictionary containing model configurations for different specifications.
    """

    # def supervised model
    config_super = config.copy()
    config_super['specification'] = 'super'
    config_super['model'] = 'nn'
    config_super['arch'] = 'mlp'
    config_super['checkpoint_path'] = checkpoint_paths[config_super['specification'] + '.' +
                                                       config_super['arch'] + '.' +
                                                       config_super['dataType']]

    # def self-supervised model
    config_self = config.copy()
    config_self['specification'] = 'selfsuper'
    config_self['model'] = 'nn'
    config_self['arch'] = 'mlp'
    config_self['checkpoint_path'] = checkpoint_paths[config_self['specification'] + '.' +
                                                      config_self['arch'] + '.' +
                                                      config_self['dataType']]

    # def test-time instance adaptive model
    config_ttia_super = config.copy()
    config_ttia_super['specification'] = 'ttia_super'
    config_ttia_super['model'] = 'gd'
    config_ttia_super['arch'] = 'mlp'
    config_ttia_super['adaptMode'] = 'per_spec_adapt'  # 'model_only', 'domain_adapt', 'stream_adapt', 'per_spec_adapt'
    config_ttia_super['innerEpochs'] = 50  # number of inner epochs
    config_ttia_super['innerBatch'] = 1  # batch size for the inner loop
    config_ttia_super['innerLr'] = 1e-4  # learning rate for the inner loop
    config_ttia_super['innerLoss'] = 'mse_specs'  # loss function for the inner loop
    config_ttia_super['initMode'] = 'nn'  # 'nn', 'rand', 'fsl', ...
    config_ttia_super['checkpoint_path'] = checkpoint_paths['super' + '.' +
                                                            config_ttia_super['arch'] + '.' +
                                                            config_ttia_super['dataType']]

    # def test-time instance adaptive model (iter 10)
    config_ttia_super_iter10 = config_ttia_super.copy()
    config_ttia_super_iter10['specification'] = 'ttia_super_iter10'
    config_ttia_super_iter10['innerEpochs'] = 10  # number of inner epochs

    # def test-time instance adaptive model (iter 100)
    config_ttia_super_iter100 = config_ttia_super.copy()
    config_ttia_super_iter100['specification'] = 'ttia_super_iter100'
    config_ttia_super_iter100['innerEpochs'] = 100  # number of inner epochs

    # def test-time instance adaptive model (iter 500)
    config_ttia_super_iter500 = config_ttia_super.copy()
    config_ttia_super_iter500['specification'] = 'ttia_super_iter500'
    config_ttia_super_iter500['innerEpochs'] = 500  # number of inner epochs

    # def test-time instance adaptive model (self-supervised)
    config_ttia_self = config_ttia_super.copy()
    config_ttia_self['specification'] = 'ttia_selfsuper'
    config_ttia_self['checkpoint_path'] = checkpoint_paths['selfsuper' + '.' +
                                                           config_ttia_self['arch'] + '.' +
                                                           config_ttia_self['dataType']]

    # def supervised model (with CNN architecture)
    config_cnn_super = config_super.copy()
    config_cnn_super['arch'] = 'cnn'
    config_cnn_super['checkpoint_path'] = checkpoint_paths[config_cnn_super['specification'] + '.' +
                                                           config_cnn_super['arch'] + '.' +
                                                           config_cnn_super['dataType']]

    # def self-supervised model (with CNN architecture)
    config_cnn_self = config_self.copy()
    config_cnn_self['arch'] = 'cnn'
    config_cnn_self['checkpoint_path'] = checkpoint_paths[config_cnn_self['specification'] + '.' +
                                                          config_cnn_self['arch'] + '.' +
                                                          config_cnn_self['dataType']]

    # def test-time instance adaptive model (CNN architecture)
    config_cnn_ttia_super = config_ttia_super.copy()
    config_cnn_ttia_super['arch'] = 'cnn'
    config_cnn_ttia_super['checkpoint_path'] = checkpoint_paths['super' + '.' +
                                                                config_cnn_ttia_super['arch'] + '.' +
                                                                config_cnn_ttia_super['dataType']]

    # def test-time instance adaptive model (CNN architecture)
    config_cnn_ttia_self = config_ttia_self.copy()
    config_cnn_ttia_self['arch'] = 'cnn'
    config_cnn_ttia_self['checkpoint_path'] = checkpoint_paths['selfsuper' + '.' +
                                                               config_cnn_ttia_self['arch'] + '.' +
                                                               config_cnn_ttia_self['dataType']]

    # def own LCM model
    config_own_lcm = config.copy()
    config_own_lcm['specification'] = 'own_lcm'
    config_own_lcm['model'] = 'ls'
    config_own_lcm['initMethod'] = 'nn'

    # def own LCM model (gradient descent)
    config_own_lcm_gd = config.copy()
    config_own_lcm_gd['specification'] = 'own_lcm_gd'
    config_own_lcm_gd['model'] = 'gd'
    config_own_lcm_gd['load_model'] = False
    config_own_lcm_gd['adaptMode'] = 'model_only'  # 'model_only', 'domain_adapt', 'stream_adapt', 'per_spec_adapt'
    config_own_lcm_gd['innerEpochs'] = 1000  # number of inner epochs
    config_own_lcm_gd['innerBatch'] = 1  # batch size for the inner loop
    config_own_lcm_gd['innerLr'] = 1e-1  # learning rate for the inner loop
    config_own_lcm_gd['innerLoss'] = 'mse_specs'  # loss function for the inner loop
    config_own_lcm_gd['initMode'] = 'nn'  # 'nn', 'rand', 'fsl', ...

    # def test-time instance adaptive model (trained from scratch)
    config_ttia_scratch = config.copy()
    config_ttia_scratch['specification'] = 'ttia_scratch'
    config_ttia_scratch['model'] = 'gd'
    config_ttia_scratch['arch'] = 'mlp'
    config_ttia_scratch['load_model'] = False
    config_ttia_scratch['adaptMode'] = 'per_spec_adapt'  # 'model_only', 'domain_adapt', 'stream_adapt', 'per_spec_adapt'
    config_ttia_scratch['innerEpochs'] = 100000  # number of inner epochs
    config_ttia_scratch['innerBatch'] = 16  # batch size for the inner loop
    config_ttia_scratch['innerLr'] = 1e-4  # learning rate for the inner loop
    config_ttia_scratch['innerLoss'] = 'mse_specs'  # loss function for the inner loop
    config_ttia_scratch['initMode'] = 'nn'  # 'nn', 'rand', 'fsl', ...

    # def test-time online adaptive model
    config_ttoa_super = config.copy()
    config_ttoa_super['specification'] = 'ttoa_super'
    config_ttoa_super['model'] = 'gd'
    config_ttoa_super['arch'] = 'mlp'
    config_ttoa_super['adaptMode'] = 'stream_adapt'  # 'model_only', 'domain_adapt', 'stream_adapt', 'per_spec_adapt'
    config_ttoa_super['innerEpochs'] = 1  # number of inner epochs
    config_ttoa_super['innerBatch'] = 16  # batch size for the inner loop
    config_ttoa_super['innerLr'] = 1e-4  # learning rate for the inner loop
    config_ttoa_super['innerLoss'] = 'mse_specs'  # loss function for the inner loop
    config_ttoa_super['initMode'] = 'nn'
    config_ttoa_super['checkpoint_path'] = checkpoint_paths['super' + '.' +
                                                            config_ttoa_super['arch'] + '.' +
                                                            config_ttoa_super['dataType']]

    # def test-time online adaptive model (self-supervised)
    config_ttoa_self = config_ttoa_super.copy()
    config_ttoa_self['specification'] = 'ttoa_selfsuper'
    config_ttoa_self['checkpoint_path'] = checkpoint_paths['selfsuper' + '.' +
                                                           config_ttoa_self['arch'] + '.' +
                                                           config_ttoa_self['dataType']]

    # def test-time domain adaptive model
    config_ttda_super = config.copy()
    config_ttda_super['specification'] = 'ttda_super'
    config_ttda_super['model'] = 'gd'
    config_ttda_super['arch'] = 'mlp'
    config_ttda_super['adaptMode'] = 'domain_adapt'
    config_ttda_super['innerEpochs'] = 1000  # number of inner epochs
    config_ttda_super['innerBatch'] = 16  # batch size
    config_ttda_super['innerLr'] = 1e-4  # learning rate for the inner loop
    config_ttda_super['innerLoss'] = 'mse_specs'
    config_ttda_super['initMode'] = 'nn'
    config_ttda_super['checkpoint_path'] = checkpoint_paths['super' + '.' +
                                                            config_ttda_super['arch'] + '.' +
                                                            config_ttda_super['dataType']]

    # def test-time domain adaptive model (self-supervised)
    config_ttda_self = config_ttda_super.copy()
    config_ttda_self['specification'] = 'ttda_selfsuper'
    config_ttda_self['checkpoint_path'] = checkpoint_paths['selfsuper' + '.' +
                                                           config_ttda_self['arch'] + '.' +
                                                           config_ttda_self['dataType']]

    # def Newton model
    config_newton = config.copy()
    config_newton['specification'] = 'fsl_newton'
    config_newton['model'] = 'lcm'
    config_newton['method'] = 'Newton'
    config_newton['save_path'] = config['path2save'] + '/newton/'

    # def MH model
    config_mh = config.copy()
    config_mh['specification'] = 'fsl_mh'
    config_mh['model'] = 'lcm'
    config_mh['method'] = 'MH'
    config_mh['save_path'] = config['path2save'] + '/mh/'

    # def LCModel model
    config_lcm = config.copy()
    config_lcm['specification'] = 'lcmodel'
    config_lcm['model'] = 'lcm'
    config_lcm['method'] = 'LCModel'
    config_lcm['save_path'] = config['path2save'] + '/lcmodel/'

    # gather all models in a dictionary
    models = {
        'super': config_super,
        'selfsuper': config_self,
        'ttia_super': config_ttia_super,
        'ttia_super_iter10': config_ttia_super_iter10,
        'ttia_super_iter100': config_ttia_super_iter100,
        'ttia_super_iter500': config_ttia_super_iter500,
        'ttia_selfsuper': config_ttia_self,
        'cnn_super': config_cnn_super,
        'cnn_selfsuper': config_cnn_self,
        'cnn_ttia_super': config_cnn_ttia_super,
        'cnn_ttia_selfsuper': config_cnn_ttia_self,
        'own_lcm': config_own_lcm,
        'own_lcm_gd': config_own_lcm_gd,
        'ttia_scratch': config_ttia_scratch,
        'ttoa_super': config_ttoa_super,
        'ttoa_selfsuper': config_ttoa_self,
        'ttda_super': config_ttda_super,
        'ttda_selfsuper': config_ttda_self,
        'newton': config_newton,
        'mh': config_mh,
        'lcmodel': config_lcm
    }
    return models


#******************************#
#   define colors for models   #
#******************************#
def get_colors():
    return {
        'super': '#F46D43',  # orange
        'selfsuper': '#D73027',  # vivid red

        'ttia_super': '#1A9850',  # strong green
        'ttia_super_iter10': '#00441B',  # darkest forest green
        'ttia_super_iter100': '#66BD63', # medium green
        'ttia_super_iter500': '#A6DBA0', # lighter green

        # 'ttoa_super': '#7570B3',   # strong purple
        'ttoa_super': '#E78AC3',  # pink
        # 'ttoa_super': '#C51B7D',  # strong fuchsia
        'ttda_super': '#4575B4',  # strong blue

        'own_lcm': '#FDCB02',  # yellow
        'own_lcm_gd': '#FDCB02',  # yellow

        'newton': '#666666',  # medium gray
        'mh': '#666666',  # medium gray
        'lcmodel': '#A6761D',  # brown-gold

        'ttia_scratch': '#1B9E77',  # teal
        'ttia_selfsuper': '#8DA0CB',  # soft blue
        'ttoa_selfsuper': '#FC8D62',  # muted orange
        'ttda_selfsuper': '#66C2A5',  # aqua

        # 'cnn_super': '#AE017E',  # deep purple-pink
        # 'cnn_selfsuper': '#F768A1',  # bright pink
        # 'cnn_ttia_super': '#FBB4B9',  # soft pink
        # 'cnn_ttia_selfsuper': '#FEEBE2'  # very light pastel pink

        'cnn_super': '#762A83',        # deep violet
        'cnn_selfsuper': '#9970AB',    # medium purple
        'cnn_ttia_super': '#C2A5CF',   # lavender
        'cnn_ttia_selfsuper': '#E7D4E8' # very light violet
    }
