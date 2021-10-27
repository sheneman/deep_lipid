Deep Lipid
==========

Deep learning semantic segmentation of lipid droplets from quantitative phase images (QPI)
---------------------------------------------------------------------------------------------------------------

This software is associated with a published PLOS ONE journal article:

Deep learning classification of lipid droplets in quantitative phase images
L. Sheneman, G. Stephanopoulos, A. E. Vasdekis;   https://doi.org/10.1371/journal.pone.0249196


The related data:  https://doi.org/10.7923/3d0d-yb04


# Acknowledgments

We gratefully acknowledge financial support U.S. Department of Energy, 
Office of Biological & Environmental Research (DE-SC0019249).

# Overview:

This library of Python code is used for performing semantic segmentation 
of images using 6 different machine learning methods. Five of the methods are implemented 
entirely within the scikit-learn framework.  The Convolutional Neural Network (CNN)
method requires Keras with a TensorFlow backend and generally uses a different set of 
scripts in order to perform the complete training and evaluation.

# Training Scripts

### **train.py**
The main tool for performing scikit-learn based machine learning.  This tool
will optionally perform k-fold cross validation of all of the specified
models against the training set.  It will also perform a full train-test
split model training run using the specified training and test sets.  It
will use the results of that operation to compute ROC Curves and the
corresponding AUC metrics for all model types.

Usage:
> python train.py [ --help | --verbose | --config=\<YAML config file\> ]

### **unet.py**
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
> python unet.py

### **unet_kfold.py**

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
> python unet_kfold.py


# Classification Scripts

### **classify.py**

Loads a saved and trained scikit-learn machine learning model from
disk and uses that model to classify all raw images within the
specified directory.

Usage:
> python classify.py [ --help | --verbose | --config=\<YAML config file\> ]


### **unet_classify.py**

This tool will load a trained and saved Keras/TensorFlow model from disk and
stream a set of raw unsigned 8-bit, pre-padded (256x256), images through the
UNET classifier in order to generate a set of binary semantic segmentation
maps for each image.  It will save the resulting binary output images to disk
in the specified output folder.   Paths and filenames are hardcoded below
(for now).

Initial 32-bit float raw images must be initially preprocessed:
  - Scaled to normalized 8-bit unsigned representation
  - Padded to be 256x256 (see pad.py)

Usage:
> python unet_classify.py

# Scoring and Evaluation Scripts

### **score.py**
A tool for scoring the binary segmentation maps created by our trained
machine learning classifiers during the classification step against the
binary "true" labels.   This will score an entire directory of classified
image output and will report many metrics in a spreadsheet format with some
summary statistics.

Usage:
> python score.py [ --help | --verbose | --config=\<YAML config file\> ]


### **roc.py**

*NOTE: This tool is currently deprecated and unused*

A tool for generating Receiver Operator Characteristic (ROC) curves and
Area Under Curve (AUC) metrics for trained models that have already been
stored on disk.

Usage:
> python roc.py [ --help | --verbose | --config=\<YAML config file\> ]


### **rocplt.py***

Simple brutish helper script to Generate a ROC curve plot given a CSV of the
correct format.  Used to make the ROC CURVE figure in related manuscript. Many 
assumptions and hardcoded parameters within script.

Usage:
> python rocplt.py


### **unet_roc.py**

A simple convenience tool for generating ROC curves and AUC metrics from
the probabalistic (i.e. not thresholded binary) output from the UNet CNN.

You must specify the path to the unpadded (original dimensions) "true"
binary images and the unpadded probabilistic output from the UNet CNN.  From
that, this tool will use scikit-learn functions to compute the ROC Curve and
the AUC metric.

Usage:
> python unet_roc.py [ --help | --verbose | --config=\<YAML config file\> ]


# Auxilliary Tools

### **pad.py**

The UNET CNN model requires all input images to be a consistent 256x256
dimension.  This tool preprocesses images of smaller dimensions to conform
to that requirement.

pad.py takes a  set of grayscale input images and pad them evenly on top,
bottom, and sides with negative pixels (value = 0) to form a 256x256 image
with the original image properly centered.

The related unpad.py tool performs the exact opposite operation in order to
result in a file with its original dimensions.

Usage:
> python pad.py [ --help | --verbose | --config=\<YAML config file\> ]


### **unpad.py**

The counterpart to pad.py, this tool crops all of the padded images in the
specified input directory back to their original, unpadded dimensions.

You must specify three parameters in the YAML config:

 - rawdir: the path to a directory containing the images in their original size
 - padded: the input directory of padded images
 - unpadded: the output directory in which to place the cropped images

unpadding the output from the UNet CNN back to its original size is a
critical postprocessing step prior to scoring (e.g. score.py)

Usage:
> python unpad.py [ --help | --verbose | --config=\<YAML config file\> ]


### **diff.py**

A simple tool for helping visualize the differences between classified
binary segmentation maps from our models compared to the binary "true"
labels.  Takes all of binary images in the input directory, compares
them (pixel-by-pixel) to the corresponding binary "true" labels.
Determines the true/false positives/negatives and outputs a color-coded
RGB image showing the differences.

Usage:
> python diff.py [ --help | --verbose | --config=\<YAML config file\> ]


### **split.py**

*NOTE: This tool is currently deprecated and unused*

A simple tool for helping to create a train/validate/test splits based on
source directories of files and specified split ratios.

Usage:
> python split.py [ --help | --verbose | --config=\<YAML config file\> ]


### **treeviz.py**

A silly tool for taking a single trained decision tree from a random forest
model and visualizing it using the GraphViz library.

Usage:
> python treeviz.py [ --help | --verbose | --config=\<YAML config file\> ]


### **unet_kfold_prepare.py** 

This is an auxillary tool to prepare a directory hierarchy populated with
raw and binary training images for use in doing a "manual" k-fold cross
validation of Keras/TensorFlow models.

Usage:
> python unet_kfold_prepare.py

