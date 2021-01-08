###############################################################################
# train.py
#
# sheneman@uidaho.edu
#
# The main tool for performing scikit-learn based machine learning.  This tool
# will optionally perform k-fold cross validation of all of the specified 
# models against the training set.  It will also perform a full train-test
# split model training run using the specified training and test sets.  It 
# will use the results of that operation to compute ROC Curves and the 
# corresponding AUC metrics for all model types.
#
# Usage:
# python train.py [ --help | --verbose | --config=<YAML config file> ]
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

# our local image preprocessing and feature extraction routines
import preprocess



# set our random seed based on current time
now = int(time())

seed(now)
numpy.random.seed(now)
numpy.set_printoptions(threshold=numpy.inf)

# Available scoring metrics
print("Available Scoring Metrics:\n")
print(sorted(metrics.SCORERS.keys()))

# Scoring metrics to use for k-fold cross validation
cv_scoring = ['accuracy', 'balanced_accuracy', 'precision', 'jaccard', 'f1', 'recall', 'roc_auc']


###############################################################################
#
# HANDLE Command line arguments
#
#
def usage():
	print("python train.py [ --help | --verbose | --config=<YAML config filename> ] ")

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
	configfile="train.yaml"

###############################################################################

###############################################################################
#
# Format and Example config YAML file:
#
# FORMAT:
# -------
#   rf_model: path and filename for pickled model output for Random Forest
#   mlp_model: path and filename for pickled model output for MLP classifier
#   xgb_model: path and filename for pickled model output for XGBoost classifier
#   svm_model: path and filename for pickled model output for SVM classifier
#   lda_model: path and filename for pickled model output for LDA classifier
#   threads: integer specifying the number of parallel threads to use
#   rawdir: <path to input raw images>
#   bindir: <path to input bin images>
#   trainlist: <input path for list of training images>
#   testlist: <input path for list of test images>
#   importance: <output path for RF feature importance telemetry>
#   k_fold: <integer> - zero means don't do k-fold cross validation, otherwise k = k_fold
#
# EXAMPLE:
# --------
#   rf_model:		"./models/random_forest.model"
#   mlp_model:		"./models/neural_network.model"
#   xgb_model:		"./models/XGBoost.model"
#   svm_model:		"./models/SVM.model"
#   lda_model:		"./models/LDA.model"
#   threads:		25
#   rawdir:             "../images/raw"
#   bindir:             "../images/binary"
#   trainlist:		"./trainlist.txt"
#   testlist:  		"./testlist.txt"
#   importance:         "./runs/feature_importance.csv"
#   k_fold:		0
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
# The main function that efficiently builds up our long 2D array of pixels 
# and their related extracted features.  This will be passed to the model fit 
# functions to train the machine learning classifier models.
#
# This function loads every raw and binary image in filenames into memory,
# computes the total number of pixels in these images, pre-allocates the output
# arrays in one step.
#
# Then for every raw image, this function calls preprocess.image_preprocess()
# to extract the k (e.g. k=80) features per pixel.  The result is flattened and 
# inserted into the pre-allocated output arrays.
#
# A similar but simpler operation is done to construct the corresponding long 
# binary label array that will be required for model fitting.
#
# This function returns 3 things:  the long 2D feature-extracted data, 
# the corresponding 1D binary label data array, and the total pixel count.
# 
#

def build_dataset(filenames, nfeatures):

	# allocate an array for images
	num_images = len(filenames)
	raw_images = numpy.empty(num_images, dtype=object)
	bin_images = numpy.empty(num_images, dtype=object)

	# Read raw and binary images into these preallocated numpy arrays of objects
	#  while also counting the total accumulated number of pixels across all input 
	#  images so we can allocated our output arrays in one operation for efficiency
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

# Check to see if we are doing K-Fold Cross Validation
k_fold = config["k_fold"]
if(k_fold > 0):
	print("K-Fold Cross Validation Enabled.  K = %d" %k_fold)
else:
	print("No K-Fold Cross Validation...")


train_filenames      = [line.rstrip('\n') for line in open(config["trainlist"])]
test_filenames       = [line.rstrip('\n') for line in open(config["testlist"])]

print("Loading Training Image Data...")
(X_train, Y_train, train_pixel_cnt)                = build_dataset(train_filenames, nfeatures)

print("Loading Test Image Data...")
(X_test, Y_test, test_pixel_cnt) = build_dataset(test_filenames, nfeatures)


# for debugging, this spews out all of the preprocessed images for the first image into a folder
#preprocess.output_preprocessed(train_raw[0], "debug")

print("Train Pixels: %d" %train_pixel_cnt)
print("Test Pixels: %d" %test_pixel_cnt)
print("Feature Vector Length: %d" %nfeatures)


# print out training and test sizes
print("X_train: %d, Y_train: %d, " %(len(X_train), len(Y_train)))
print("X_test: %d, Y_test: %d" %(len(X_test), len(Y_test)))


# Instantiate our k-fold splits
if(k_fold>0):
	kf = KFold(n_splits = k_fold, shuffle=False)
	kf.get_n_splits(X_train)
	print(kf)

# Instantiate the classifiers
n_estimators = config["threads"]
n_jobs = int(n_estimators/2+1)

svm_classifier = svm.SVC(kernel='poly', degree=3, gamma='scale', verbose=True, max_iter=1000, cache_size=5000, random_state=now, probability=True)
xgb_classifier = XGBClassifier(n_estimators=100, verbosity=2, nthread=config["threads"], max_depth=4, subsample=0.5)
rf_classifier = RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=config["threads"], random_state=now)
mlp_classifier = MLPClassifier(hidden_layer_sizes=(50,25), max_iter=1000, n_iter_no_change=50, activation = 'relu',solver='adam',random_state=now,verbose=True)
lda_classifier = LDA(solver='svd')


# Do the compute-intensive k-fold cross-validation 
if(k_fold > 0):

	print("K-Fold Cross Validating for Support Vector Machine (SVM) Model...\n")
	svm_cv_results = cross_validate(svm_classifier, X_train, Y_train, cv=kf, scoring=cv_scoring);
	print("Support Vector Machine (SVM) Cross Validation Results:")	
	print(svm_cv_results)
	print("\n\n\n")

	print("K-Fold Cross Validating for XGBoost Model...\n")
	xgb_cv_results = cross_validate(xgb_classifier, X_train, Y_train, cv=kf, scoring=cv_scoring);
	print("XGBoost Cross Validation Results:")	
	print(xgb_cv_results)
	print("\n\n\n")

	print("K-Fold Cross Validating for Random Forest Model...\n")
	rf_cv_results  = cross_validate(rf_classifier, X_train, Y_train, cv=kf, scoring=cv_scoring);
	print("Random Forest (RF) Cross Validation Results:")	
	print(rf_cv_results)
	print("\n\n\n")

	print("K-Fold Cross Validating for MLP Neural Network Model...\n")
	mlp_cv_results = cross_validate(mlp_classifier, X_train, Y_train, cv=kf, scoring=cv_scoring);
	print("Neural Network (MLP) Cross Validation Results:")	
	print(mlp_cv_results)
	print("\n\n\n")

	print("K-Fold Cross Validating for Linear Discrimant Analysis (LDA) Model...\n")
	lda_cv_results = cross_validate(lda_classifier, X_train, Y_train, cv=kf, scoring=cv_scoring);
	print("Linear Discriminate Analysis (LDA) Cross Validation Results:")	
	print(lda_cv_results)
	print("\n\n\n")


#########################################################################
#
# Now perform a secondary training based on our train-test split to 
# train the full training set to create savable and reusable models and
# compute the Receiver Operating Characteristic (ROC) curve plots
# and Area Under the Curve (AUC) metrics for each trained model.
#

# train and pickle SVM classifier model
print("Training the SVM Classifier...")
svm_classifier.fit(X_train, Y_train)
print("DUMPING SVM Model...")
pickle.dump(svm_classifier, open(config["svm_model"],'wb'))

# train and pickle XGBoost Classifier
print("Training the XGBoost Classifier...")
xgb_classifier.fit(X_train, Y_train)
print("DUMPING XGBoost Classifier...")
pickle.dump(xgb_classifier, open(config["xgb_model"],'wb'))

# train and pickle Random Forest model
print("Training the Random Forest...")
rf_classifier.fit(X_train, Y_train)
print("DUMPING Random Forest Model...")
pickle.dump(rf_classifier,  open(config["rf_model"],'wb'))

# train and pickle MLP classifier model
print("Training the MLP Neural Network...")
mlp_classifier.fit(X_train, Y_train)
print("DUMPING MLP Model...")
pickle.dump(mlp_classifier, open(config["mlp_model"],'wb'))

# train and pickle LDA Classifier
print("Training the Linear Discriminant Analysis (LDA) Classifier...")
lda_classifier.fit(X_train, Y_train)
print("DUMPING LDA Classifier...")
pickle.dump(lda_classifier, open(config["lda_model"],'wb'))


# Output the most important random forest features to a telemetry file
feature_file = open(config["importance"], "w")
for feature in sorted(zip(flabels, rf_classifier.feature_importances_), key=lambda x: x[1], reverse=True):
	feature_file.write("%s,%f\n" %feature)
feature_file.close()


# Create Results Table Here for ROC Curves
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# SVM
Y_pred = svm_classifier.predict(X_test)
yproba = svm_classifier.predict_proba(X_test)[::,1]
fpr,tpr,_ = roc_curve(Y_test, yproba)
auc = roc_auc_score(Y_test, yproba)
result_table = result_table.append({'classifiers':"SVM", 'fpr':fpr, 'tpr':tpr, 'auc':auc}, ignore_index = True)
print('SVM Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('SVM Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('SVM Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('SVM FPR:', fpr)
print('SVM TPR:', tpr)
print('SVM AUC:', auc)
print("\n")


# XGBoost
Y_pred = xgb_classifier.predict(X_test)
yproba = xgb_classifier.predict_proba(X_test)[::,1]
fpr,tpr,_ = roc_curve(Y_test, yproba)
auc = roc_auc_score(Y_test, yproba)
result_table = result_table.append({'classifiers':"XGB", 'fpr':fpr, 'tpr':tpr, 'auc':auc}, ignore_index = True)
print('XGB Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('XGB Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('XGB Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('XGB FPR:', fpr)
print('XGB TPR:', tpr)
print('XGB AUC:', auc)
print("\n")



# Random Forest (RF)
Y_pred = rf_classifier.predict(X_test)
yproba = rf_classifier.predict_proba(X_test)[::,1]
fpr,tpr,_ = roc_curve(Y_test, yproba)
auc = roc_auc_score(Y_test, yproba)
result_table = result_table.append({'classifiers':"RF", 'fpr':fpr, 'tpr':tpr, 'auc':auc}, ignore_index = True)
print('RF Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('RF Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('RF Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('RF FPR:', fpr)
print('RF TPR:', tpr)
print('RF AUC:', auc)
print("\n")



# Multilayer Perceptron (MLP)
Y_pred = mlp_classifier.predict(X_test)
yproba = mlp_classifier.predict_proba(X_test)[::,1]
fpr,tpr,_ = roc_curve(Y_test, yproba)
auc = roc_auc_score(Y_test, yproba)
result_table = result_table.append({'classifiers':"MLP", 'fpr':fpr, 'tpr':tpr, 'auc':auc}, ignore_index = True)
print('MLP Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('MLP Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('MLP Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('MLP FPR:', fpr)
print('MLP TPR:', tpr)
print('MLP AUC:', auc)
print("\n")



# Linear Ddiscriminant Analysis (LDA)
Y_pred = lda_classifier.predict(X_test)
yproba = lda_classifier.predict_proba(X_test)[::,1]
fpr,tpr,_ = roc_curve(Y_test, yproba)
auc = roc_auc_score(Y_test, yproba)
result_table = result_table.append({'classifiers':"LDA", 'fpr':fpr, 'tpr':tpr, 'auc':auc}, ignore_index = True)
print('LDA Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('LDA Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('LDA Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('LDA FPR:', fpr)
print('LDA TPR:', tpr)
print('LDA AUC:', auc)
print("\n")


#
# Build a ROC Curve plot image 
#
result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(8,6))

for i in result_table.index:
	plt.plot(result_table.loc[i]['fpr'], 
	result_table.loc[i]['tpr'], 
	label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(numpy.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(numpy.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.savefig('roc_curve2.png');


exit(0)

