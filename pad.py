###############################################################################
# pad.py
#
# sheneman@uidaho.edu
#
#
# The UNET CNN model requires all input images to be a consistent 256x256
# dimension.  This tool preprocesses images of smaller dimensions to conform
# to that requirement.
#
# pad.py takes a  set of grayscale input images and pad them evenly on top, 
# bottom, and sides with negative pixels (value = 0) to form a 256x256 image 
# with the original image properly centered.
#
# The related unpad.py tool performs the exact opposite operation in order to 
# result in a file with its original dimensions.
#
# Usage:
# python pad.py [ --help | --verbose | --config=<YAML config file> ]
#
###############################################################################

import sys, os, getopt, yaml
from os import listdir
from PIL import Image, ImageOps

# for now, this is hardcoded, should put in YAML config file
PAD_SIZE = 256


###############################################################################
#
# HANDLE Command line arguments
#
#
def usage():
	print("python pad.py [ --help | --verbose | --config=<YAML config filename> ] ")

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
	configfile="pad.yaml"

###############################################################################

###############################################################################
#
# Format and Example config YAML file:
#
# FORMAT:
# -------
#   rawdir: <path to input raw images>
#   outdir: <path to output directory for padded images>
#
# EXAMPLE:
# --------
#   rawdir:             "../images/raw"
#   outdir:             "../images/padded"
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
##  FUNCTION DEFINITIONS
###############################################################################


###############################################################################
#
# function:  pad_file()
#

def pad_file(filename,new_width,new_height):
	
	print("In pad_file():  file = %s" %filename)
	fullpath = config["rawdir"] + '/' + filename
	img = Image.open(fullpath)
	(width,height) = img.size
	if(width > PAD_SIZE or height > PAD_SIZE):
		return(-1)

	delta_w = new_width  - width
	delta_h = new_height - height

	padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
	new_img = ImageOps.expand(img, padding)	

	return(new_img)
	






###############################################################################
#
# read all files in the input directory and call pad_file() to pad them to
# PAD_SIZE.  Output the padded image to the specified output directory

filenames = listdir(config["rawdir"]) 

for filename in filenames:
	padded_image = pad_file(filename,PAD_SIZE,PAD_SIZE)
	if(padded_image == -1):
		print("skipping %s" %filename)
	else:
		savepath = config["outdir"] + '/' + filename
		padded_image.save(savepath, "TIFF")

exit(0)


