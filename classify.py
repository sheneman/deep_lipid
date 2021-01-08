###############################################################################
# classify.py
#
# sheneman@uidaho.edu
# 
# Loads a saved and trained scikit-learn machine learning model from 
# disk and uses that model to classify all raw images within the 
# specified directory.  
#
# Usage: 
# python classify.py [ --help | --verbose | --config=<YAML config file> ]
#
###############################################################################

import os
import sys
import getopt
import yaml
import pickle
import numpy
import cv2
from os import listdir
from os.path import isfile, join
from PIL import Image

import preprocess

numpy.set_printoptions(threshold=sys.maxsize)


###############################################################################
#
# HANDLE Command line arguments
#
#
def usage():
	print("python classify.py [ --help | --verbose | --config=<YAML config filename> ] ")

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
	configfile="classify.yaml"

###############################################################################

###############################################################################
#
# Format and Example config YAML file:
#
# FORMAT:
# -------
#   inputlist:	<path to a text file that names which files to classify>
#   rawdir: 	<path to raw images>
#   model:	<path to the machine learning model to use for classification>
#   outputdir:	<path to output folder>
#
# EXAMPLE:
# --------
#   inputlist:	"./input_filenames.txt"
#   rawdir:	"../images/raw"
#   model:	"./models/rf.model"
#   outputdir:	"./output_directory"
#
###############################################################################

cf = open(configfile, "r")
config = yaml.load(cf, Loader=yaml.FullLoader)
print("YAML CONFIG:")
for c in config:
	print("    [%s]:\"%s\"" %(c,config[c]))
print("\n")
cf.close()



#
# get feature labels
#
flabels   = preprocess.feature_labels()
nfeatures = preprocess.feature_count(flabels)
print("Number of Feature Labels: %d" %nfeatures)


# get the list of raw files to classify
filenames = [line.rstrip('\n') for line in open(config["inputlist"])]


# Load the Trained Classifier Model
print("Loading Trained Classifier Model = %s" %config["model"])
classifier = pickle.load(open(config["model"],'rb'))
classifier.verbose = False
print("Trained Classifier Model Loaded!")



##############################################################################
# Load all of the raw images from the specified input directory and perform
# image segmentation (pixel classification) using the loaded machine learning
# model.
#
# For each image, extract all features for every pixel in the raw image and
# pass the 2D feature array to the model for classification.

for filename in filenames:

	output  = config["outputdir"] + "/" + filename
	rawpath = config["rawdir"] + "/" + filename

	print("Loading: %s" %rawpath)
	raw_img = Image.open(rawpath)

	numcols,numrows = raw_img.size
	raw_cv2 = numpy.array(raw_img)
	raw_img.close()

	# extract feature array for loaded image
	data = preprocess.image_preprocess(filename, nfeatures, raw_cv2)

	print("Image Size: numcols=%d x numrows=%d" %(numcols,numrows))
	print("Num Features: %d" %nfeatures)

	# classify our input pixels and their feature vector
	Y_pred = classifier.predict(data)  
	
	predicted_array = numpy.reshape(Y_pred,(numrows,numcols),order='F')
	predicted_array = cv2.normalize(predicted_array,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	predicted_image = Image.fromarray(predicted_array)
	
	predicted_image.save(output, "TIFF")
	predicted_image.close()

exit(0)
