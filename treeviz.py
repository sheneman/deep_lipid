###############################################################################
# treeviz.py
#
# sheneman@uidaho.edu
#
# A silly tool for taking a single trained decision tree from a random forest
# model and visualizing it using the GraphViz library.  
#
# Usage:
# python treeviz.py [ --help | --verbose | --config=<YAML config file> ]
#
################################################################################

from PIL import Image, ImageFilter
import sys
import getopt
import yaml
import pickle
from os.path import isfile, join
import numpy
import pprint
from skimage import feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from subprocess import call

import preprocess


numpy.set_printoptions(threshold=sys.maxsize)


#################################################################################
#
# HANDLE Command line arguments
#
#
def usage():
	print("python treeviz.py [ --help | --verbose | --config=<YAML config filename> ] ")

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
	configfile="treeviz.yaml"

#################################################################################

#################################################################################
#
# Format and Example config YAML file:
#
# FORMAT:
# -------
#   model:	<path to the model to use for classification>
#   output:	<path to output image>
#
# EXAMPLE:
# --------
#   model:	"./models/foo.model"
#   output:	"./tree.png"
#
#################################################################################

cf = open(configfile, "r")
config = yaml.load(cf, Loader=yaml.FullLoader)
print("YAML CONFIG:")
for c in config:
	print("    [%s]:\"%s\"" %(c,config[c]))
print("\n")
cf.close()

# Load the classifier model
print("Loading classifier model = %s" %config["model"])
classifier = pickle.load(open(config["model"],'rb'))
print("Model %s Loaded!" %config["model"])

# Get the feature labels
feature_labels = preprocess.feature_labels()
print(feature_labels)

tree = classifier.estimators_[0]
export_graphviz(tree, out_file='./tree.dot', 
                feature_names = feature_labels,
                class_names = ("Non-Lipid", "Lipid"),
                rounded = True, proportion = False, 
                precision = 2, filled = True)


#call(['dot', '-Tpng', '/tmp/tree.dot', '-o', config["output"], '-Gdpi=600'])
print("Done!")

exit(0)

