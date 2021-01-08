###############################################################################
# diff.py
#
# sheneman@uidaho.edu
#
# A simple tool for helping visualize the differences between classified 
# binary segmentation maps from our models compared to the binary "true"
# labels.  Takes all of binary images in the input directory, compares
# them (pixel-by-pixel) to the corresponding binary "true" labels. 
# Determines the true/false positives/negatives and outputs a color-coded 
# RGB image showing the differences.
#
# Usage:
# python diff.py [ --help | --verbose | --config=<YAML config file> ]
#
###############################################################################


import getopt
import yaml
import sys
import cv2
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy
import re
from pprint import pprint



###############################################################################
#
# HANDLE Command line arguments
#
#
def usage():
	print("python diff.py [ --help | --verbose | --config=<YAML config filename> ] ")

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
	configfile="diff.yaml"

###############################################################################

###############################################################################
#
# Format and Example config YAML file:
#
# FORMAT:
# -------
#   bindir: <path to bin images>
#   inputdir: <input directory with classified images>
#   outputdir: <output directory to put diff images>
#
# EXAMPLE:
# --------
#   bindir:             "../images/binary"
#   inputdir:        	"./classified_images_directory"
#   outputdir: 		"./output_diff_images_directory"
#
#################################################################################

cf = open(configfile, "r")
config = yaml.load(cf, Loader=yaml.FullLoader)
print("YAML CONFIG:")
for c in config:
        print("    [%s]:\"%s\"" %(c,config[c]))
print("\n")
cf.close()


###############################################################################
# Iterate through all images in the inputdir, compare to binary images
# and output color-coded differences to a set of RGB images
# 

filelist = [f for f in listdir(config["inputdir"]) if isfile(join(config["inputdir"], f))]
for f in filelist:

	print(f)

	binary_fullpath = config["bindir"] + '/' + f
	input_fullpath  = config["inputdir"] + '/' + f
	output_fullpath = config["outputdir"] + '/' + f

	binary_img = Image.open(binary_fullpath)
	input_img  = Image.open(input_fullpath)

	binary_imgarray = numpy.array(binary_img)
	input_imgarray  = numpy.array(input_img)

	(width,height) = binary_imgarray.shape
	
	output_img = Image.new('RGB', (height, width), (255, 255, 255))
	output_arr = numpy.array(output_img)

	for i in range(width):
		for j in range(height):

			if(binary_imgarray[i][j] == 0 and input_imgarray[i][j]==0):
				color = (0,0,0)  # true negative
			elif(binary_imgarray[i][j] == 255 and input_imgarray[i][j] == 255):
				color = (255,255,255)  # true positive
			elif(binary_imgarray[i][j] == 0 and input_imgarray[i][j] == 255):
				color = (255,0,0)  # FALSE POSITIVE
			elif(binary_imgarray[i][j] == 255 and input_imgarray[i][j] == 0):
				color = (0,255,0)  # FALSE NEGATIVE
			else:
				print("ERROR Scoring file")
				exit(0)

			output_arr[i,j]=color

	output_img = Image.fromarray(output_arr);
	output_img.save(output_fullpath, "TIFF")

exit(0)
