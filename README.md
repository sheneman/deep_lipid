Deep Lipid
==========

Deep learning semantic segmentation of lipid droplets from quantitative phase images (QPI)
---------------------------------------------------------------------------------------------------------------

This software is associated with a manuscript currently under peer review.

A bioRxiv preprint of this paper is available:
Deep learning classification of lipid droplets in quantitative phase images
L. Sheneman, G. Stephanopoulos, A. E. Vasdekis
bioRxiv 2020.06.01.128447; doi: https://doi.org/10.1101/2020.06.01.128447


The related data:  https://doi.org/10.7923/3d0d-yb04


Acknowledgments
--------------
We gratefully acknowledge financial support U.S. Department of Energy, 
Office of Biological & Environmental Research (DE-SC0019249).

Overview:
---------
This library of Python code is used for performing semantic segmentation 
of images using 6 different machine learning methods. Five of the methods are implemented 
entirely within the scikit-learn framework.  The Convolutional Neural Network (CNN)
method requires Keras with a TensorFlow backend and generally uses a different set of 
scripts in order to perform the complete training and evaluation.

Training Scripts
----------------
### __train.py__
The main tool for performing scikit-learn based machine learning.  This tool
will optionally perform k-fold cross validation of all of the specified
models against the training set.  It will also perform a full train-test
split model training run using the specified training and test sets.  It
will use the results of that operation to compute ROC Curves and the
corresponding AUC metrics for all model types.

Usage:
python train.py [ --help | --verbose | --config=\<YAML config file\> ]

### __unet.py__
The primary tool for training a UNet CNN model given a directory of training
images and a corresonding directory of masks (i.e. "true" binary labels).
This tool combines a stream of unsigned 8-bit, pre-padded (256x256) training
images and their corresponding pre-padded binary masks and trains a Unet CNN.

32-bit float raw images must be initially preprocessed:
   - Scaled to normalized 8-bit unsigned representation
   - Padded to be 256x256 (see pad.py)

The binary masks must also be padded to a consistent 256x256 size.

Most of the training parameters (Epochs, Batch size, etc.) are hardcoded.

The output is a trained model and the model training history.

Usage:
  python unet.py

### __unet_kfold.py__

Perform a "manual" k-fold cross validation with the UNet CNN model.  This
assumes that a directory hierarchy has been configured and populated (see
unet_kfold_prepare.py).

This program will save the trained models for each fold in the appropriate
place in the directory hierarchy.  Those trained models can then be loaded
by unet_classify.py to compute the predicted output (binary semantic
segmentation maps) against the hold-out (test) set within each fold and then
scored against the "true" binary labels using various # model
effectiveness metrics (score.py).

Usage:
  python unet_kfold.py



