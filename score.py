###############################################################################
# score.py
#
# sheneman@uidaho.edu
#
# A tool for scoring the binary segmentation maps created by our trained
# machine learning classifiers during the classification step against the 
# binary "true" labels.   This will score an entire directory of classified
# image output and will report many metrics in a spreadsheet format with some
# summary statistics.
#
# Usage:
# python score.py [ --help | --verbose | --config=<YAML config file> ]
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
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from scipy.spatial.distance import dice
numpy.set_printoptions(threshold=sys.maxsize)


pattern = re.compile(".*tif")
NUM_METRICS = 13 # hardcoded for now!!


###############################################################################
#
# HANDLE Command line arguments
#
#
def usage():
	print("python score.py [ --help | --verbose | --config=<YAML config filename> ] ")

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
	configfile="score.yaml"

###############################################################################

###############################################################################
#
# Format and Example config YAML file:
#
# FORMAT:
# -------
#   bindir: <path to bin images>
#   inputdir: <input directory with classified images>
#
# EXAMPLE:
# --------
#   bindir:             "../images/binary"
#   inputdir:        	"./classified_images_directory"
#
###############################################################################

cf = open(configfile, "r")
config = yaml.load(cf, Loader=yaml.FullLoader)
cf.close()


###############################################################################
#
# Iterate through all images and compute all metrics
#

# the header for our csv-like output
header = "FILENAME,True Positives,False Positives,True Negatives,False Negatives,TM_CCORR_NORMED,DICE,Jaccard,F1,F0.5,Precision,Recall,ROC_AUC,Accuracy,Balanced_Accuracy"
print(header)

filelist = [f for f in listdir(config["inputdir"]) if isfile(join(config["inputdir"], f))]
scores = numpy.empty((len(filelist),NUM_METRICS), dtype=numpy.float32)
file_index = 0
for f in filelist:
	x = 0

	if(pattern.match(f)):

		binary_fullpath = config["bindir"] + '/' + f
		output_fullpath = config["inputdir"] + '/' + f

		binary_img = Image.open(binary_fullpath)
		output_img = Image.open(output_fullpath)

		binary_imgarray = numpy.array(binary_img)
		output_imgarray = numpy.array(output_img)

		# normalize to [0,1] range
		binary_imgarray = cv2.normalize(binary_imgarray,None,0,1,cv2.NORM_MINMAX,cv2.CV_8U)   
		output_imgarray = cv2.normalize(output_imgarray,None,0,1,cv2.NORM_MINMAX,cv2.CV_8U) 

		numrows=len(binary_imgarray)
		numcols=len(binary_imgarray[0])

		totalsize=numrows*numcols

		TP = 0 # true positives
		FP = 0 # false positives
		TN = 0 # true negatives
		FN = 0 # false negatives

		# raw calculation of true positives, false positives, true negatives, false negatives		
		for i in range(numrows):
			for j in range(numcols):
				#print("bin[%d][%d] = %d, output[%d][%d] = %d" %(i,j,i,j,binary_imgarray[i][j], output_imgarray[i][j]))
				if(binary_imgarray[i][j] == 0 and output_imgarray[i][j]==0):
					TN=TN+1
				elif(binary_imgarray[i][j] == 1 and output_imgarray[i][j] == 1):
					TP=TP+1
				elif(binary_imgarray[i][j] == 0 and output_imgarray[i][j] == 1):
					FP=FP+1
				elif(binary_imgarray[i][j] == 1 and output_imgarray[i][j] == 0):
					FN=FN+1
				else:
					print("ERROR Scoring file")
					exit(0)

		# For Debugging
		#print("TP = ", TP);
		#print("FP = ", FP);
		#print("TN = ", TN);
		#print("FN = ", FN);

		scores[file_index][x] = TP; x+=1  #0
		scores[file_index][x] = FP; x+=1  #1
		scores[file_index][x] = TN; x+=1  #2
		scores[file_index][x] = FN; x+=1  #3

		#4 OpenCV2 Template Cross-Correlation Normalized (TM_CCORR_NORMED)
		tm_ccorr = cv2.matchTemplate(output_imgarray,binary_imgarray,cv2.TM_CCORR_NORMED)[0][0]
		scores[file_index][x] = tm_ccorr; x+=1
		
		#5 DICE (same as F1)
		DICE = 1.0 - dice(binary_imgarray.flatten(), output_imgarray.flatten())
		scores[file_index][x] = DICE; x+=1

		#6 JACCARD
		JACCARD = jaccard_score(binary_imgarray.flatten(), output_imgarray.flatten())
		scores[file_index][x] = JACCARD; x+=1

		#7 F1 (same as DICE)
		F1 = f1_score(binary_imgarray.flatten(), output_imgarray.flatten())
		scores[file_index][x] = F1; x+=1

		#8 F05	
		F05 = fbeta_score(binary_imgarray.flatten(), output_imgarray.flatten(), beta=0.5)
		scores[file_index][x] = F05; x+=1

		#9 PRECISION
		PRECISION = precision_score(binary_imgarray.flatten(), output_imgarray.flatten())
		scores[file_index][x] = PRECISION; x+=1

		#10 RECALL
		RECALL = recall_score(binary_imgarray.flatten(), output_imgarray.flatten())
		scores[file_index][x] = RECALL; x+=1

		#11 ACCURACY
		ACCURACY = accuracy_score(binary_imgarray.flatten(), output_imgarray.flatten()) 
		scores[file_index][x] = ACCURACY; x+=1

		#12 BALANCED ACCURACY
		BALANCED_ACCURACY = balanced_accuracy_score(binary_imgarray.flatten(), output_imgarray.flatten()) 
		scores[file_index][x] = BALANCED_ACCURACY; x+=1

		# output the row
		output = f + ',' + str(TP) + ',' + str(FP) + ',' + str(TN) + ',' + str(FN) + ',' + str(tm_ccorr) + ',' + str(DICE) + ',' + str(JACCARD) + ',' + str(F1) + ',' + str(F05) + ',' + str(PRECISION) + ',' + str(RECALL) + ',' + str(ACCURACY) + ',' + str(BALANCED_ACCURACY)
		print(output);

		file_index += 1

scores = scores[0:file_index][:]


print("\n\n")
print(" Median TM_CCORR: %f" %(numpy.median(scores[:,4])))
print("     Median DICE: %f" %(numpy.nanmedian(scores[:,5])))
print("  Median JACCARD: %f" %(numpy.nanmedian(scores[:,6])))
print("       Median F1: %f" %(numpy.nanmedian(scores[:,7])))
print("     Median F.05: %f" %(numpy.nanmedian(scores[:,8])))
print("Median PRECISION: %f" %(numpy.nanmedian(scores[:,9])))
print("   Median RECALL: %f" %(numpy.nanmedian(scores[:,10])))
print(" Median ACCURACY: %f" %(numpy.nanmedian(scores[:,11])))
print("Med BAL ACCURACY: %f" %(numpy.nanmedian(scores[:,12])))

print("   Mean TM_CCORR: %f" %(numpy.mean(scores[:,4])))
print("       Mean DICE: %f" %(numpy.nanmean(scores[:,5])))
print("    Mean JACCARD: %f" %(numpy.nanmean(scores[:,6])))
print("         Mean F1: %f" %(numpy.nanmean(scores[:,7])))
print("       Mean F.05: %f" %(numpy.nanmean(scores[:,8])))
print("  Mean PRECISION: %f" %(numpy.nanmean(scores[:,9])))
print("     Mean RECALL: %f" %(numpy.nanmean(scores[:,10])))
print("   Mean ACCURACY: %f" %(numpy.nanmean(scores[:,11])))
print("Med BAL ACCURACY: %f" %(numpy.nanmean(scores[:,12])))

exit(0)

