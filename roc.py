###############################################################################
# roc.py
#
# sheneman@uidaho.edu
#
# NOTE: ***This tool is currently deprecated and unused***
#
# A tool for generating Receiver Operator Characteristic (ROC) curves and 
# Area Under Curve (AUC) metrics for trained models that have already been 
# stored on disk.
#
# Usage:
# python roc.py [ --help | --verbose | --config=<YAML config file> ]
#
###############################################################################

from PIL import Image, ImageFilter
import os
import pandas as pd
import getopt
import yaml
import pickle
from os import listdir
from os.path import isfile, join
import numpy
import scipy
import cv2
import pprint
from xgboost import XGBClassifier
#from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage.filters import gaussian_filter
from skimage import feature
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectFromModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import glob
import sys
from time import sleep, time
from datetime import datetime
from random import seed
from random import random

import preprocess



# set our random seed based on current time
now = int(time())

# SHENEMAN - UNCOMMENT SEED CALL BELOW WHEN DONE TESTING
seed(now)
numpy.random.seed(now)
numpy.set_printoptions(threshold=numpy.inf)



###############################################################################
#
# HANDLE Command line arguments
#
#
def usage():
	print("python roc.py [ --help | --verbose | --config=<YAML config filename> ] ")

try:
	opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["help", "config="])
except getopt.GetoptError as err:
	print(err)  # will print something like "option -a not recognized"
	usage()
	sys.exit(2)

configfile = None
verbose = False

for o, a in opts:
	if o == "-v":
		verbose = True
	elif o in ("-h", "--help"):
		usage()
		sys.exit()
	elif o in ("-c", "--config"):
		configfile = a
	else:
		assert False, "unhandled option"

if(configfile == None):
	configfile="roc.yaml"

###############################################################################

###############################################################################
#
# Format and Example config YAML file:
#
# FORMAT:
# -------
#   rawdir:   <path to raw images>
#   bindir:   <path to bin images>
#   model:    <path and filename for pickled model to use>
#   testlist: <input path for test images>
#
# EXAMPLE:
# --------
#   rawdir:             "../images/raw"
#   bindir:             "../images/binary"
#   model:		"./models/rf.model"
#   testlist:  		"./testlist.txt"
#
###############################################################################

cf = open(configfile, "r")
config = yaml.load(cf, Loader=yaml.FullLoader)
print("YAML CONFIG:")
for c in config:
	print("    [%s]:\"%s\"" %(c,config[c]))
print("\n")
cf.close()



###############################################################################
##  FUNCTION DEFINITIONS
###############################################################################


###############################################################################
#
# function:  build_dataset()
#
# Load all of the images from the specified files, extract all features.  This builds "dataset"
# which is a list of feature arrays.  Every element in the list is 
# a list of preprocessed images

def build_dataset(filenames, nfeatures):

	# allocate an array for images
	num_images = len(filenames)
	raw_images = numpy.empty(num_images, dtype=object)
	bin_images = numpy.empty(num_images, dtype=object)

	# Read raw and binary images into these preallocated numpy arrays of objects
	index = 0
	pixel_cnt = 0
	for f in filenames:
		print("%d: [%s]" %(index,f))
		rawpath = config["rawdir"] + "/" + f
		binpath = config["bindir"] + "/" + f

		raw_image = Image.open(rawpath)
		raw_images[index] = numpy.array(raw_image)
		raw_image.close()

		bin_image = Image.open(binpath)
		bin_images[index] = numpy.array(bin_image)
		bin_images[index] = cv2.normalize(bin_images[index],None,0,1,cv2.NORM_MINMAX,cv2.CV_8U)   # SHENEMAN
		bin_image.close()
	
		pixel_cnt += raw_images[index].size
		index += 1

	print("Number of Pixels in %d images: %d" %(index,pixel_cnt))


	# Now that we know the number of pixels, we can allocate raw and bin arrays
	raw = numpy.empty((pixel_cnt,nfeatures),dtype=numpy.uint8)
	bin = numpy.empty(pixel_cnt,dtype=numpy.uint8)


	#
	# Process raw images
	#
	pixel_cnt = 0
	pixel_index = 0
	for raw_cv2 in raw_images:

		pixels = preprocess.image_preprocess(f, nfeatures, raw_cv2)
		raw[pixel_index:pixel_index+pixels.shape[0],:] = pixels

		pixel_index+=pixels.shape[0]
		pixel_cnt += raw_cv2.size


	#
	# Process binary images
	#
	pixel_index = 0
	for bin_cv2 in bin_images:

		pixels = bin_cv2.flatten(order='F')
		bin[pixel_index:pixel_index+len(pixels)] = pixels

		pixel_index += len(pixels)
	

	return(raw, bin, pixel_cnt)


#################################################################################################




#
#  MAIN CODE HERE
#


#
# get feature labels
#
flabels   = preprocess.feature_labels()
nfeatures = preprocess.feature_count(flabels)
print("Number of Feature Labels: %d" %(len(flabels)))
print(flabels)


test_filenames       = [line.rstrip('\n') for line in open(config["testlist"])]

print("Loading Test Image Data...")
(X_test, Y_test, test_pixel_cnt) = build_dataset(test_filenames, nfeatures)


print("Test Pixels: %d" %test_pixel_cnt)
print("Feature Vector Length: %d" %nfeatures)

# print out training and test sizes
print("X_test: %d, Y_test: %d" %(len(X_test), len(Y_test)))

classifier = pickle.load(open(config["model"],'rb'))


Y_pred = classifier.predict(X_test)
yproba = classifier.predict_proba(X_test)[::,1]
fpr,tpr,_ = roc_curve(Y_test, yproba)
auc = roc_auc_score(Y_test, yproba)
print('FPR:', fpr)
print('TPR:', tpr)
print('AUC:', auc)


exit(0)

