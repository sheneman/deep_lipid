###############################################################################
# unpad.py
#
# sheneman@uidaho.edu
#
# The counterpart to pad.py, this tool crops all of the padded images in the 
# specified input directory back to their original, unpadded dimensions.
#
# You must specify three parameters in the YAML config:
#
#  rawdir: the path to a directory containing the images in their original size
#  padded: the input directory of padded images
#  unpadded: the output directory in which to place the cropped images 
#
# unpadding the output from the UNet CNN back to its original size is a 
# critical postprocessing step prior to scoring (e.g. score.py)
#
# Usage:
# python unpad.py [ --help | --verbose | --config=<YAML config file> ]
#
###############################################################################

import sys, os, getopt, yaml
from os import listdir
from PIL import Image, ImageOps



###############################################################################
#
# HANDLE Command line arguments
#
#
def usage():
	print("python unpad.py [ --help | --verbose | --config=<YAML config filename> ] ")

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
	configfile="unpad.yaml"

#################################################################################

#################################################################################
#
# Format and Example config YAML file:
#
# FORMAT:
# -------
#   rawdir: <path to original raw images>
#   padded: <path to padded images to unpad>
# unpadded: <path to place the unpadded images>
#
# EXAMPLE:
# --------
#   rawdir:             "../images/raw"
#   padded:             "./output"
# unpadded:             "./unpadded"
#
#################################################################################

cf = open(configfile, "r")
config = yaml.load(cf, Loader=yaml.FullLoader)
print("YAML CONFIG:")
for c in config:
	print("    [%s]:\"%s\"" %(c,config[c]))
print("\n")
cf.close()



#################################################################################################
##  FUNCTION DEFINITIONS
#################################################################################################


##################################################################################################
#
# function:  unpad_file()
#
def unpad_image(image, filename):
	
	print("In unpad_image():  file = %s" %filename)

	(width,height) = image.size	

	# determine original image width and height
	fullpath = config["rawdir"] + '/' + filename
	img = Image.open(fullpath)
	(original_width,original_height) = img.size

	delta_w = width  - original_width
	delta_h = height - original_height

	border = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
	new_img = ImageOps.crop(image, border)

	return(new_img)

	

# Main code 
#
filenames = listdir(config["padded"]) 

for f in filenames:
	fullpath = config["padded"] + '/' + f
	img = Image.open(fullpath)
	unpadded_image = unpad_image(img, f)
	
	savepath = config["unpadded"] + '/' + f
	unpadded_image.save(savepath, "TIFF")

exit(0)

