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
1. train.py - 
