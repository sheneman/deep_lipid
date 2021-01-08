###############################################################################
# split.py
#
# sheneman@uidaho.edu
#
# NOTE: ***This tool is currently deprecated and unused***
#
# A simple tool for helping to create a train/validate/test splits based on
# source directories of files and specified split ratios.
#
# Usage:
# python split.py [ --help | --verbose | --config=<YAML config file> ]
#
###############################################################################

import os
import getopt
import yaml
from os import listdir
from os.path import isfile, join
import pprint
import glob
import sys
from time import time
from random import seed, random, shuffle
from math import floor

#
# This script takes a directory of input files and partitions them into a list of
# input files to use for TRAINING, VALIDATION, and TESTING 
#


# set our random seed based on current time
now = int(time())
seed(now)



###############################################################################
#
# HANDLE Command line arguments
#
#
def usage():
        print("python split.py [ --help | --verbose | --config=<YAML config filename> ] ")

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
	configfile = "split.yaml"

#################################################################################

#################################################################################
#
# Format and Example config YAML file:
#
# FORMAT:
# -------
#   rawdir: <path to raw images>
#   bindir: <path to bin images>
#   train_fraction: <fraction between 0 and 1.0>
#   validation_fraction: <fraction between 0 and 1.0>
#   test_fraction: <fraction between 0 and 1.0>
#   file_filter: <posix file filter for selecting files>
#   trainlist_out: <output path for training images>
#   validationlist_out: <output path for validation images>
#   testlist_out: <output path for test images>
#
# EXAMPLE: 
# --------
#   rawdir:              "../images/raw"
#   bindir:              "../images/binary"
#   train_fraction:      0.60
#   validation_fraxtion: 0.20
#   test_fraction:       0.20
#   file_filter:         "Po1g_100_1*.tif"
#   trainlist_out:       "./trainlist.txt"
#   validationlist_out:  "./validationlist.txt"
#   testlist_out:        "./testlist.txt"
#
#################################################################################

cf = open(configfile, "r")
config = yaml.load(cf, Loader=yaml.FullLoader)
print("YAML CONFIG:")
for c in config:
	print("    [%s]:\"%s\"" %(c,config[c]))
print("\n")
cf.close()


# Set some paths for our image library of raw and binary labeled data
IMG_RAWPATH = config["rawdir"]
IMG_BINPATH = config["bindir"]


# Set output filenames
TRAIN_FILENAME      = config["trainlist_out"]
VALIDATION_FILENAME = config["validationlist_out"]
TEST_FILENAME       = config["testlist_out"]

FILE_FILTER         = config["file_filter"]

# The fraction of the image library that will be used for training, validation, and testing
# The totals must add up to 1.0
TRAIN_FRACTION      = config["train_fraction"]
VALIDATION_FRACTION = config["validation_fraction"]
TEST_FRACTION       = config["test_fraction"]




# Get all of the filenames that match the filter and shuffle them in place
cwd = os.getcwd()
os.chdir(IMG_RAWPATH)
filenames = glob.glob(FILE_FILTER)
os.chdir(cwd)
shuffle(filenames)

num_filenames = len(filenames)

if(TRAIN_FRACTION < 0 or TRAIN_FRACTION > 1.0):
	print("ERROR: TRAIN_FRACTION must be between 0,0 and 1,0")
	exit(-1)

train_partition_start = 0
train_partition_end = int(floor(num_filenames*TRAIN_FRACTION-1))
validation_partition_size = int(floor(num_filenames*VALIDATION_FRACTION))
if(train_partition_end == num_filenames-1):
	test_partition_start = -1
	test_partition_end   = -1
	validation_partition_start = -1
	validation_partition_end = -1
else:
	validation_partition_start = train_partition_end+1
	validation_partition_end = validation_partition_start + validation_partition_size
	test_partition_start = validation_partition_end + 1
	test_partition_end   = num_filenames-1

if(train_partition_start < 0 or train_partition_end < 0 or train_partition_end < train_partition_start):
	train_partition_start = -1
	train_partiiton_end = -1
	

print("TOTAL NUMBER OF FILENAMES: %d" %num_filenames)
print("TRAIN: %d thru %d" %(train_partition_start,train_partition_end))
print("VALIDATION: %d thru %d" %(validation_partition_start,validation_partition_end))
print("TEST: %d thru %d" %(test_partition_start,test_partition_end))


# Write the training set input file
if(train_partition_start >= 0):
	file = open(TRAIN_FILENAME, "w")
	for i in range(train_partition_start, train_partition_end+1):
		file.write(filenames[i] + "\n")
	file.close()

# Write the validation set input file
if(validation_partition_start >= 0):
	file = open(VALIDATION_FILENAME, "w")
	for i in range(validation_partition_start, validation_partition_end+1):
		file.write(filenames[i] + "\n")
	file.close()

# Write the test set input file
if(test_partition_start >= 0):
	file = open(TEST_FILENAME, "w")
	for i in range(test_partition_start, test_partition_end+1):
		file.write(filenames[i] + "\n")
	file.close()

exit(0)

