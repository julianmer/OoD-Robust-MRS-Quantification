# Strategies to Minimize Out-of-Distribution Effects in Data-Driven MRS Quantification

## Abstract

**Purpose:** This study systematically compared data-driven and model-based strategies for metabolite quantification in magnetic resonance spectroscopy (MRS), focusing on resilience to out-of-distribution (OoD) effects and the balance between accuracy, robustness, and generalizability.

**Methods:** A neural network architecture designed for MRS quantification was trained using three distinct strategies: supervised regression, self-supervised learning, and test-time adaptation. These were compared against model-based fitting tools. Experiments combined large-scale simulated data, designed to probe metabolite concentration extrapolation and signal variability, with 1H single-voxel 7T in-vivo human brain spectra.

**Results:** In simulations, supervised learning achieved high accuracy for spectra similar to those in the training distribution, but showed marked degradation when extrapolated beyond the training distribution. Test-time adaptation proved more resilient to OoD effects, while self-supervised learning achieved intermediate performance. In-vivo experiments showed larger variance across the methods (data-driven and model-based) due to domain shift. Across all strategies, overlapping metabolites and baseline variability remained persistent challenges.

**Conclusion:** While strong performance can be achieved by data-driven methods for MRS metabolite quantification, their reliability is contingent on careful consideration of the training distribution and potential OoD effects. When such conditions in the target distribution cannot be anticipated, test-time adaptation strategies ensure consistency between the quantification, the data, and the model, enabling reliable data-driven MRS pipelines.

## Overview

This repository consists of the following Python scripts:
* The `train.py` implements the pipeline to train (and test) the deep learning approaches.
* The `sweep.py` defines ranges to sweep for optimal hyperparamters using Weights & Biases.
* The `frameworks/` folder holds the frameworks for model-based and data-driven methods.
  * The `framework.py` defines the framework class to inherit from.
  * The `frameworkFSL.py` consists of a wrapper for FLS-MRS for easy use.
  * The `frameworkGD.py`implements gradient descent-based fitting, including purely model-based and test-time adaptation.
  * The `frameworkLCM.py` is a wrapper for LCModel.
  * The `frameworkNN.py` holds the base framework class for deep learning-based methods.
  * The `lcmodel/` dictionary holds the LCModel binaries and executables.
* The `loading/` folder holds the scripts to automate the loading of MRS data formats.
  * The `dicom.py` defines functions for the DICOM loader (Philips).
  * The `lcmodel.py` contains loaders for the LCModel formats.
  * The `loadBasis.py` holds the loader for numerous basis set file formats.
  * The `loadConc.py` enables loading of concentration files provided by fitting software.
  * The `loadData.py` defines the loader for the MRS data.
  * The `loadLABRAW.py` allows to load the Philips lab raw format.
  * The `loadMRSI.py` contains functions for MRSI data.
  * The `philips.py` holds functions to load Philips data.
* The `models/` folder holds the models for the deep learning approaches.
  * The `nnModels.py` defines the neural network models.
* The `other/` folder holds miscellaneous scripts.
  * The `brainbeats.py` defines functions to load and process the BrainBeats dataset.
  * The `fitting.py` performs fitting and reporting of in-vivo spectra.
* The `simulation/` folder holds the scripts to simulate MRS spectra.
  * The `basis.py` has the basis set class to hold the spectra.
  * The `dataModules.py` are creating datasets by loading in-vivo data or simulating ad-hoc during training.
  * The `sigModels.py` defines signal models to simulate MRS spectra.
  * The `simulation.py` draws simulation parameters from distibutions to allow simulation with the signal model.
  * The `simulationDefs.py` holds predefined simulation parameters ranges.
* The `tests/` folder holds the test scripts.
  * The `modelDefs.py` defines model configurations for testing.
  * The `test.py` defines the test class to inherit from. 
  * The `testInVivoOoD.py` contains the tests for the in-vivo in- and out-of-distribution tests.
  * The `testOoD.py` contains the tests for the in- and out-of-distribution tests.
* The `utils/` folder holds helpful functionalities.
  * The `auxiliary.py` defines some helpful functions.
  * The `components.py` consists of functions to create signal components.
  * The `gpu_config.py` is used for the GPU configuration.
  * The `processing.py` defines functions for processing MRS data.
  * The `structures.py` implements helpful structures.
* The `visualisation/` folder holds scripts for visualisation of results.
  * The `plotFunctions.py` defines functions to plotting spectra and results.
  * The `plotFunctionsErr.py` holds functions to plotting benchmarking results.

## Requirements

| Module            | Version |
|:------------------|:-------:|
| fsl_mrs           | 2.1.20  |
| h5py              | 3.14.0  |
| matplotlib        | 3.10.7  |
| nifti_mrs         |  1.3.3  |
| numpy             |  2.3.5  |
| pandas            |  2.3.3  |
| psutil            |  7.0.0  |
| pydicom           |  3.0.1  |
| pytorch_lightning |  2.5.5  |
| scikit_learn |  1.7.2  |
| scipy             | 1.16.3  |
| seaborn	          | 0.13.2  |
| shutup            |  0.3.0  |
| spec2nii          |  0.8.6  |
| torch             |  2.7.1  |
| torchmetrics      |  1.7.4  |
| tqdm              | 4.67.1  |
| wandb             | 0.23.0  |
