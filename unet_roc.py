###############################################################################
#
# unet_roc.py
#
# sheneman@uidaho.edu
#
# A simple convenience tool for generating ROC curves and AUC metrics from 
# the probabalistic (i.e. not thresholded binary) output from the UNet CNN.
#
# You must specify the path to the unpadded (original dimensions) "true" 
# binary images and the unpadded probabilistic output from the UNet CNN.  From
# that, this tool will use scikit-learn functions to compute the ROC Curve and
# the AUC metric.
#
# Usage:
# python unet_roc.py [ --help | --verbose | --config=<YAML config file> ]
#
###############################################################################

import getopt
import yaml
import sys
import cv2
import pandas as pd
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy
import re
from pprint import pprint
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import dice
import matplotlib.pyplot as plt

numpy.set_printoptions(threshold=sys.maxsize)


pattern = re.compile(".*tif")


#################################################################################
#
# HANDLE Command line arguments
#
#
def usage():
	print("python unet_roc.py [ --help | --verbose | --config=<YAML config filename> ] ")

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
	configfile="unet_roc.yaml"

#################################################################################

#################################################################################
#
# Format and Example config YAML file:
#
# FORMAT:
# -------
#   bindir: <path to bin images>
#   inputdir: <input directory with non-binary (proba) classified images>
#
# EXAMPLE:
# --------
#   bindir:             "../images/binary"
#   inputdir:        	"./classified/classified_proba
#
#################################################################################

cf = open(configfile, "r")
config = yaml.load(cf, Loader=yaml.FullLoader)
cf.close()


####################################################################
#
# Iterate through all images and build one large proba array
#
filelist = [f for f in listdir(config["inputdir"]) if isfile(join(config["inputdir"], f))]
file_index = 0
y_test  = []
y_proba = []
for f in filelist:
	if(pattern.match(f)):

		#print(file_index)

		binary_fullpath = config["bindir"]   + '/' + f
		output_fullpath = config["inputdir"] + '/' + f

		binary_img = Image.open(binary_fullpath)
		output_img = Image.open(output_fullpath)

		binary_imgarray = numpy.array(binary_img).flatten() / 255
		output_imgarray = numpy.array(output_img).flatten() / 255

		print(output_imgarray)

		y_test  = numpy.append(y_test,  binary_imgarray)
		y_proba = numpy.append(y_proba, output_imgarray)

		file_index = file_index + 1



fpr,tpr,_ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# Create Results Table Here for ROC Curves
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
result_table = result_table.append({'classifiers':"UNET", 'fpr':fpr, 'tpr':tpr, 'auc':auc}, ignore_index = True)

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

plt.savefig('CNN_ROC.eps');

print(fpr)
print(tpr)
print(auc) 

exit(0)

