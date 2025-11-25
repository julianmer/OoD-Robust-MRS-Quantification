####################################################################################################
#                                          frameworkGD.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/25                                                                                #
#                                                                                                  #
# Purpose: Implementation of a self-supervised fitting method in PyTorch for linear combination    #
#          modeling in magnetic resonance spectroscopy.                                            #
#                                                                                                  #
####################################################################################################



#*************#
#   imports   #
#*************#
import copy
import os
import torch
import shutup; shutup.please()   # shut up warnings

from joblib import Parallel, delayed

# own
from frameworks.frameworkNN import FrameworkNN
from utils.processing import processSpectra



#**************************************************************************************************#
#                                           FrameworkGD                                            #
#**************************************************************************************************#
#                                                                                                  #
# The framework allowing to define, train, and test data-driven models.                            #
#                                                                                                  #
#**************************************************************************************************#
class FrameworkGD(FrameworkNN):
    def __init__(self, path2basis, basisFmt='', specType='synth', dataType='none', ppmlim=None,
                 norm=True, forwardNorm=True, val_size=1000, batch=16, lr=1e-3,
                 reg_l1=0.0, reg_l2=0.0, loss='mse', arch='mlp', adaptMode='model_only',
                 innerEpochs=1e2, innerBatch=1, innerLr=1e-2, innerOptimizer='adam',
                 innerLoss='mse', innerState='train', bnState='eval', initMode='rand',
                 multiprocessing=True, **kwargs):
        FrameworkNN.__init__(self, path2basis, basisFmt, specType, dataType, ppmlim, norm,
                             forwardNorm, val_size, batch, lr, reg_l1, reg_l2, loss, arch,
                             **kwargs)
        self.adaptMode = adaptMode
        self.innerEpochs = innerEpochs
        self.innerBatch = innerBatch
        self.innerLr = innerLr
        self.innerOptimizer = self.getOptimizer(innerOptimizer)
        self.innerLoss = innerLoss
        self.innerState = innerState
        self.bnState = bnState
        self.initMode = initMode
        self.multiprocessing = multiprocessing

        if kwargs.get('load_model', False) and 'checkpoint_path' in kwargs:
            self.net = FrameworkNN.load_from_checkpoint(
                path2basis=path2basis, basisFmt=basisFmt, specType=specType,
                dataType=dataType, ppmlim=ppmlim, val_size=val_size, batch=batch, lr=lr,
                reg_l1=reg_l1, reg_l2=reg_l2, loss=loss, arch=arch, **kwargs
            ).net


    #****************************#
    #   select inner optimizer   #
    #****************************#
    def getOptimizer(self, innerOptimizer='adam'):
        if innerOptimizer == 'adam':
            return torch.optim.Adam
        elif innerOptimizer == 'adagrad':
            return torch.optim.Adagrad
        elif innerOptimizer == 'adamax':
            return torch.optim.Adamax
        elif innerOptimizer == 'adamw':
            return torch.optim.AdamW
        elif innerOptimizer == 'rmsprop':
            return torch.optim.RMSprop
        elif innerOptimizer == 'sgd':
            return torch.optim.SGD
        else:
            raise ValueError(f"Unknown inner optimizer: {innerOptimizer}")


    #*********************#
    #   input to output   #
    #*********************#
    def forward(self, x, x_ref=None, frac=None, x0=None):
        assert x_ref is None, 'Water referencing not supported!'
        assert frac is None, 'Tissue correction not supported!'
        assert x0 is None, 'Initial parameters not supported!'

        if self.adaptMode == 'model_only':
            return self.model_only_gd(x, x0)

        elif self.adaptMode == 'domain_adapt':
            return self.domain_adapt(x)

        elif self.adaptMode == 'stream_adapt':
            return self.stream_adapt(x)

        elif self.adaptMode == 'per_spec_adapt':
            return self.per_spec_adapt(x)

        else:
            raise ValueError(f"Unknown adapt_mode: {self.adaptMode}")


    #***************************#
    #   initialize parameters   #
    #***************************#
    def initalizeTheta(self, x, mode='nn'):
        if mode == 'nn':
            xs = x[:, :, self.first:self.last:self.skip]
            return self.net(xs.float())
        else:
            x = x[:, 0] + 1j * x[:, 1]
            theta = self.sigModel.initParam(x, mode=mode, basisFSL=self.basisObj.basisFSL)
            theta[:, :len(self.basisFSL.names)] = torch.stack([theta[:, self.basisFSL.names.index(m)]  # sort
                                                               for m in self.basisObj.names], dim=1)
            return theta


    #***************************#
    #   purely model-based GD   #
    #***************************#
    def model_only_gd(self, x, x0=None):
        # no neural network involved, only the model parameters are optimized via gradient descent
        # with an activation function used for constraints such as positivity for concentrations...

        class ActivatedTheta(torch.nn.Module):
            def __init__(self, theta_init, n_metabs, activation='custom'):
                super().__init__()
                self.theta = torch.nn.Parameter(theta_init)
                self.n_metabs = n_metabs
                self.act_fn = self.get_activation_fn(activation)
                self.inv_act_fn = self.get_inverse_activation_fn(activation)

            def get_activation_fn(self, activation):
                if activation == 'default':
                    def activation_fn(theta):
                        thetac = torch.nn.functional.relu(theta[:, :self.n_metabs])
                        thetas = torch.nn.functional.relu(theta[:, self.n_metabs:self.n_metabs + 2]) + 1
                        thetao = theta[:, self.n_metabs + 2:-7]
                        thetap1 = torch.nn.functional.hardtanh(theta[:, -7:-6]) * 1e-4
                        thetar = theta[:, -6:]
                        return torch.cat((thetac, thetas, thetao, thetap1, thetar), dim=-1)
                    return activation_fn
                elif activation == 'custom':
                    def activation_fn(theta):
                        thetac = torch.nn.functional.softplus(theta[:, :self.n_metabs])
                        thetas = torch.nn.functional.softplus(theta[:, self.n_metabs:self.n_metabs + 2]) + 1
                        thetao = theta[:, self.n_metabs + 2:-7]
                        thetap1 = torch.nn.functional.tanh(theta[:, -7:-6]) * 1e-4
                        thetar = theta[:, -6:]
                        return torch.cat((thetac, thetas, thetao, thetap1, thetar), dim=-1)
                    return activation_fn
                else:
                    raise ValueError(f"Unknown activation function: {activation}")

            def get_inverse_activation_fn(self, activation):
                if activation == 'default':
                    def inverse(activated_theta):
                        thetac = activated_theta[:, :self.n_metabs]  # Assume > 0
                        thetas = activated_theta[:, self.n_metabs:self.n_metabs + 2] - 1
                        thetao = activated_theta[:, self.n_metabs + 2:-7]
                        thetap1 = torch.clamp(activated_theta[:, -7:-6] / 1e-4, -1, 1)
                        thetar = activated_theta[:, -6:]
                        return torch.cat((thetac, thetas, thetao, thetap1, thetar), dim=-1)
                    return inverse
                if activation == 'custom':
                    def inverse(activated_theta):
                        def softplus_inv(y):
                            return torch.log(torch.expm1(y))
                        thetac = softplus_inv(activated_theta[:, :self.n_metabs])
                        thetas = softplus_inv(activated_theta[:, self.n_metabs:self.n_metabs + 2] - 1)
                        thetao = activated_theta[:, self.n_metabs + 2:-7]
                        z = torch.clamp(activated_theta[:, -7:-6] / 1e-4, -0.999, 0.999)
                        thetap1 = 0.5 * torch.log((1 + z) / (1 - z))
                        thetar = activated_theta[:, -6:]
                        return torch.cat((thetac, thetas, thetao, thetap1, thetar), dim=-1)
                    return inverse
                else:
                    raise ValueError(f"Unknown activation function: {activation}")

            def forward(self):
                return self.act_fn(self.theta)

        def adapt_theta(x_i, theta_i):
            x_i = x_i.clone()
            device = x_i.device
            model = ActivatedTheta(theta_i.detach(), self.basisObj.n_metabs).to(device)
            optimizer = self.innerOptimizer(model.parameters(), lr=self.innerLr)

            for _ in range(int(self.innerEpochs)):
                optimizer.zero_grad()
                with torch.enable_grad():
                    theta_act = model()
                    loss = self.loss(x_i, None, None, theta_act, type=self.innerLoss).mean()
                    loss.backward()
                optimizer.step()

            with torch.no_grad():
                return model().detach()

        # normalize the input if required
        if self.norm:
            xs = x[:, :, self.first:self.last:self.skip]
            norm = torch.sqrt(torch.sum(xs[:, 0, :] ** 2 + xs[:, 1, :] ** 2, dim=1, keepdim=True))
            norm = torch.clamp(norm, min=1e-8)  # avoid division by zero
            x = x / norm.unsqueeze(-1)  # normalize the spectra

        # initialize the parameters
        if x0 is not None:
            theta = x0.clone().to(x.device)
        else:
            theta = self.initalizeTheta(x, mode=self.initMode).clone().to(x.device)

        # roughly scale the initial parameters
        act_mod = ActivatedTheta(theta.detach(), self.basisObj.n_metabs)
        theta_act = act_mod.act_fn(theta)
        s = self.sigModel.forward(theta_act)
        s = processSpectra(s, self.basis)
        x_norm = torch.sqrt(torch.sum(x[:, 0, :] ** 2 + x[:, 1, :] ** 2, dim=1, keepdim=True))
        s_norm = torch.sqrt(torch.sum(s[:, 0, :] ** 2 + s[:, 1, :] ** 2, dim=1, keepdim=True))
        delta = torch.clamp(x_norm / s_norm, min=1e-8)  # avoid division by zero
        theta[:, :self.basisObj.n_metabs] = act_mod.inv_act_fn(theta_act * delta)[:, :self.basisObj.n_metabs]
        theta[:, -6:] *= act_mod.inv_act_fn(theta_act * delta)[:, -6:]  # scale the baseline (attention: assumes fix 2nd order baseline)

        # prepare batches
        x_batches = [x[i:i + self.innerBatch].to(x.device)
                     for i in range(0, x.shape[0], self.innerBatch)]
        theta_batches = [theta[i:i + self.innerBatch]
                         for i in range(0, theta.shape[0], self.innerBatch)]

        # parallel execution
        if self.multiprocessing:
            n_cpus = os.cpu_count() or 1
            optimized_thetas = Parallel(n_jobs=min(len(x_batches), n_cpus))(
                delayed(adapt_theta)(x_i, t_i) for x_i, t_i in zip(x_batches, theta_batches)
            )
        else:
            optimized_thetas = [adapt_theta(x_i, t_i) for x_i, t_i in zip(x_batches, theta_batches)]
        theta = torch.cat(optimized_thetas, dim=0)

        # scale the concentrations with the norm
        if self.norm and self.forwardNorm:
            theta[:, :self.basisObj.n_metabs] *= norm
            theta[:, -6:] *= norm  # scale the baseline (attention: assumes fix 2nd order baseline)

        return theta


    #****************************#
    #   batch-wise fine-tuning   #
    #****************************#
    def domain_adapt(self, x):
        # train the model for all given spectra sequentially, so for each batch the model
        # is updated until all spectra are seen, then next epoch, predictions are made at end

        # select the spectral range of interest
        xs = x[:, :, self.first:self.last:self.skip]

        # normalize the input if required
        if self.norm:
            norm = torch.sqrt(torch.sum(xs[:, 0, :] ** 2 + xs[:, 1, :] ** 2, dim=1, keepdim=True))
            norm = torch.clamp(norm, min=1e-8)  # avoid division by zero
            xs = xs / norm.unsqueeze(-1)  # normalize the spectra

        with torch.enable_grad():   # makes sure gradients are enabled!
            model = copy.deepcopy(self.net)
            model.train() if self.innerState == 'train' else model.eval()

            # set batch norm to evaluation mode if applicable
            if hasattr(model, 'set_bn_mode'): model.set_bn_mode(self.bnState)

            optimizer = self.innerOptimizer(model.parameters(), lr=self.innerLr)
            for _ in range(int(self.innerEpochs)):
                for i in range(0, x.shape[0], self.innerBatch):  # loop per inner batch
                    theta = model(xs[i:i + self.innerBatch].float())

                    # scale the concentrations with the norm
                    if self.norm and self.forwardNorm:
                        theta[:, :self.basisObj.n_metabs] *= norm[i:i + self.innerBatch]
                        # scale the baseline (attention: assumes fix 2nd order baseline)
                        theta[:, -6:] *= norm[i:i + self.innerBatch]

                    loss = self.loss(x[i:i + self.innerBatch], None, None, theta, type=self.innerLoss).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # final forward after adaptation
        theta = model(xs.float())

        # scale the concentrations with the norm
        if self.norm and self.forwardNorm:
            theta[:, :self.basisObj.n_metabs] *= norm
            theta[:, -6:] *= norm

        return theta


    #************************#
    #   stream fine-tuning   #
    #************************#
    def stream_adapt(self, x):
        # train the model on batch, make predictions, then update the model with the next batch
        assert self.innerEpochs == 1, 'Inner epochs must be 1 for stream adaptation!'

        # select the spectral range of interest
        xs = x[:, :, self.first:self.last:self.skip]

        # normalize the input if required
        if self.norm:
            norm = torch.sqrt(torch.sum(xs[:, 0, :] ** 2 + xs[:, 1, :] ** 2, dim=1, keepdim=True))
            norm = torch.clamp(norm, min=1e-8)  # avoid division by zero
            xs = xs / norm.unsqueeze(-1)  # normalize the spectra

        with torch.enable_grad():   # makes sure gradients are enabled!
            model = copy.deepcopy(self.net)
            model.train() if self.innerState == 'train' else model.eval()

            # set batch norm to evaluation mode if applicable
            if hasattr(model, 'set_bn_mode'): model.set_bn_mode(self.bnState)

            optimizer = self.innerOptimizer(model.parameters(), lr=self.innerLr)
            thetas = []
            for _ in range(int(self.innerEpochs)):
                for i in range(0, x.shape[0], self.innerBatch):  # loop per inner batch
                    theta = model(xs[i:i + self.innerBatch].float())

                    # scale the concentrations with the norm
                    if self.norm and self.forwardNorm:
                        theta[:, :self.basisObj.n_metabs] *= norm[i:i + self.innerBatch]
                        # scale the baseline (attention: assumes fix 2nd order baseline)
                        theta[:, -6:] *= norm[i:i + self.innerBatch]
                    thetas.append(theta)

                    loss = self.loss(x[i:i + self.innerBatch], None, None, theta, type=self.innerLoss).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        thetas = torch.cat(thetas, dim=0)
        return thetas


    #*****************************#
    #   sample-wise fine-tuning   #
    #*****************************#
    def per_spec_adapt(self, x):
        # train the model per inner batch, so for each batch the model is updated,
        # then the model is reinitialized for the next inner batch

        def adapt_single(x_i):
            x_i = x_i.clone()
            model = copy.deepcopy(self.net)
            model.train() if self.innerState == 'train' else model.eval()

            # set batch norm to evaluation mode if applicable
            if hasattr(model, 'set_bn_mode'): model.set_bn_mode(self.bnState)

            optimizer = self.innerOptimizer(model.parameters(), lr=self.innerLr)
            with torch.enable_grad():   # makes sure gradients are enabled!
                for _ in range(self.innerEpochs):
                    pred = model(x_i[:, :, self.first:self.last].float())
                    loss = self.loss(x_i, None, None, pred, type=self.innerLoss).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # final forward after adaptation
            return model(x_i[:, :, self.first:self.last].float())

        # normalize the input if required
        if self.norm:
            xs = x[:, :, self.first:self.last:self.skip]
            norm = torch.sqrt(torch.sum(xs[:, 0, :] ** 2 + xs[:, 1, :] ** 2, dim=1, keepdim=True))
            norm = torch.clamp(norm, min=1e-8)  # avoid division by zero
            x = x / norm.unsqueeze(-1)  # normalize the spectra

        # create batches
        x_batches = [x[i:i + self.innerBatch] for i in range(0, x.shape[0], self.innerBatch)]

        # parallel execution across CPU cores (adjust n_jobs as needed)
        if self.multiprocessing:
            n_cpus = os.cpu_count() or 1
            optimized_thetas = Parallel(n_jobs=min(len(x_batches), n_cpus))(
                delayed(adapt_single)(x_i) for x_i in x_batches
            )
        else:
            optimized_thetas = [adapt_single(x_i) for x_i in x_batches]
        theta = torch.cat(optimized_thetas, dim=0)

        # scale the concentrations with the norm
        if self.norm and self.forwardNorm:
            theta[:, :self.basisObj.n_metabs] *= norm
            theta[:, -6:] *= norm  # scale the baseline (attention: assumes fix 2nd order baseline)

        return theta