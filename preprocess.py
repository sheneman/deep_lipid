###############################################################################
# preprocess.py
#
# sheneman@uidaho.edu
#
# A library of functions to extract features from images using image filtering
# functions.  Also constructs an input array of pixels and their corresponding
# features that can be passed as training or classification input to 
# scikit-learn model fit / classify functions.
#
###############################################################################

from PIL import Image, ImageFilter
from os import listdir
from os.path import isfile, join
import numpy
import scipy
import cv2
from scipy.ndimage.filters import gaussian_filter
from skimage import feature
from random import seed
from random import randint



# some hardcoded values
# the labels used to identify the cell image based primary on cell age.
AGE_CLASSES = [ "MTYL_17", "MTYL_28", "MTYL_52", "MTYL_76", "MTYL_100", "MTYL_124", "Po1g_100" ]

# the fixed set of sigma parameters to use with most of our image filter functions
SIGMAS = [ 0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0 ]

preprocess_counter = 0



###############################################################################
#
# apply_mask()
#
# NOTE:  ***CURRENTLY DEPRECATED AND UNUSED***
#
# Apply a binary image mask representing the cell of interest in the frame
# Value of 0 represents NOT A CELL
# Any Non-Zero value represents CELL.  (usually specified as 255)
#
def apply_mask(raw_image, mask_image):
	raw_array  = numpy.array(raw_image)
	mask_array = numpy.array(mask_image)
	(numrows,numcols) = raw_array.shape

	for c in range(numcols):
		for r in range(numrows):
			if(mask_array[r][c] == 0):
				raw_array[r][c] = 0

	masked_raw = Image.fromarray(raw_array);
	return(masked_raw)



###############################################################################
#
# auto_canny()
#
# NOTE: ***CURRENTLY DEPRECATED AND UNUSED***
#
# Perform canny edge detection with automatic threshold
#
# From Adrian Rosebrock
# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
#
#
def auto_canny(img, sigma=0.33):

	# preprocess to smooth out details
	#img = cv2.GaussianBlur(img, (3, 3), 0)

	# compute the median of the single channel pixel intensities
	v = numpy.median(img)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v)/4)
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(img, lower, upper)
 
	# return the edged image
	return edged


###############################################################################
#
# median_hood()
#
# NOTE:  ***CURRENTLY DEPRECATED AND UNUSED***
# 
# Set pixels to extreme value based on threshhold and comparison of median 
# neighborhood intensity relative to median intensity of overall image
#
#
def median_hood(img_array, radius=4, factor=1.0):

	# if the input image is an unsigned 8-bit int, this function may generate an overflow
	if(img_array[0][0].dtype == numpy.uint8):
		return(img_array)

	new_array = numpy.ndarray(img_array.shape,dtype=numpy.float32)
	(numrows,numcols) = img_array.shape
	for x in range(numcols):
		for y in range(numrows):

			x1 = x-radius
			y1 = y-radius

			x2 = x+radius
			y2 = y+radius

			if(x1<0):
				x1=0
			if(y1<0):
				y1=0
			if(x2>=numcols):
				x2=numcols-1
			if(y2>=numrows):
				y2=numrows-1

			neighborhood=img_array[y1:y2, x1:x2]
			if(numpy.median(neighborhood) >= factor*numpy.median(img_array)):
				new_array[y,x] = 1.0
			else:
				new_array[y,x] = -1.0

	return(new_array)




###############################################################################
#
# center_proximity()
#
# Determine the relative proximity of each pixel in an image to the center of 
# the image (scaled from [-1,1]) and use this proxity value as a feature for 
# that pixel. It is a way to encode relative spatial information for the pixel.
# 
def center_proximity(img_array):

	new_array = numpy.ndarray(img_array.shape,dtype=numpy.float32)
	(numrows,numcols) = img_array.shape
	for x in range(numcols):
		for y in range(numrows):

			distx = abs((float(x)/float(numcols)-0.5)*2.0)
			disty = abs((float(y)/float(numrows)-0.5)*2.0)
			final = 1.0-((distx+disty)/2.0)

			new_array[y,x] = final

	return(new_array)



###############################################################################
#
# feature_count()
#
# Return the size of the feature vector by just counting the number of feature
# labels we are tracking

def feature_count(feature_labels):
	return(len(feature_labels))



###############################################################################
#
# label_age()
#
# Given an image array, a specified age class label, and the filename of the 
# current image, return a new array where each pixel is set to a binary value
# based on the filename of the input image (our filenames contain the age 
# class string)
#
# The goal of this simple feature extraction function is to encode the known
# age class of the cell within the image in each pixel as a form of 
# possibly useful metadata.   This has been shown (by looking at 
# random forest feature importance) to be generally unhelpful.
#
def label_age(img_array, age, filename):
	new_array = numpy.ndarray(img_array.shape, dtype=numpy.uint8)

	if age in filename:
		label = 255
	else:
		label = 0

	new_array.fill(label)

	return(new_array)
	





##############################################################################
# 
# feature_labels()
#
# This is a brute-force function to manually build a set of feature labels 
# that correspond to the feature extraction functions called within the
# image_preprocess() function.
#
# The structure of this function must match the structure of the 
# image_preprocess() function exactly in order for feature labels to match.
#
# The feature labels are useful in diagnostics and constructing the 
# feature_importance telemetry from our ensemble methods
#
def feature_labels(sigmas = SIGMAS):

	labels = []

	labels.append("Original")	

	for age in AGE_CLASSES:
		labels.append(age)	

	for s in sigmas:
		# Gaussian Smoothing
		l = "Gaussian_Smoothing_" + str(s)
		labels.append(l)

		# Sobel Edge Detection
		l = "Sobel_Edge_Detection_" + str(s)
		labels.append(l)

		# Laplacian of Gaussian Edge Detection
		l = "Laplacian_of_Gaussian_Edge_Detection_" + str(s)
		labels.append(l)

		# Gaussian Gradient Magnitude Edge Detection
		l = "Gaussian_Gradient_Magnitude_Edge_Detection_" + str(s)
		labels.append(l)

		# Difference of Gaussians
		l = "Difference_of_Gaussians_" + str(s)
		labels.append(l)

		# Structure Tensor Eigenvalues
		l = "Structure_Tensor_Eigenvalues_Large_" + str(s)
		labels.append(l)
		l = "Structure_Tensor_Eigenvalues_Small_" + str(s)
		labels.append(l)

		# Hessian Matrix
		l = "Hessian_Matrix_Hrr_" + str(s)
		labels.append(l)
		l = "Hessian_Matrix_Hrc_" + str(s)
		labels.append(l)
		l = "Hessian_Matrix_Hcc_" + str(s)
		labels.append(l)

	labels.append("center_proximity")
	labels.append("Intensity_Threshold")

	return(labels)



##############################################################################
#
# image_preprocess()
#
# The main function of this file that takes a filename, the length of our 
# feature vector (e.g. nfeatures=80), the original image as an array, and an
# optional list of sigma parameters and performs feature extraction operations
# in a specific order (which must match the order of our feature_labels() 
# function.
#
# 
#

def image_preprocess(filename, nfeatures, original_image_array, sigmas = SIGMAS):

	global preprocess_counter
	x=0

	# For performance reasons, pre-allocate an empty 2D array of the right shape 
	#  and dimensions to contain all of the given image's pixels and features
	images = numpy.empty((nfeatures, original_image_array.size), dtype=numpy.uint8)

	# normalize our (32-bit grayscale) image to an 8-bit unsigned representation 
	#  to conserve memory and increase performance
	img = cv2.normalize(original_image_array,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

	# the first feature is just the original normalized pixel value
	images[x]=img.flatten(order='F'); x+=1  # insert original image 

	# a set of features per pixel based on the image's filename which 
	#  encodes metadata related to the age class of the cell
	for age in AGE_CLASSES:
		img = label_age(original_image_array, age, filename)
		images[x] = img.flatten(order='F'); x+=1

	# call a series of parameterized image filters to extract features
	for s in sigmas:
		# Gaussian Smoothing
		img = gaussian_filter(original_image_array, sigma=s)
		img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1

		# Sobel Edge Detection
		img = scipy.ndimage.sobel(original_image_array, mode='constant', cval=s)
		img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1

		# Laplacian of Gaussian Edge Detection
		img = scipy.ndimage.gaussian_laplace(original_image_array, sigma=s)
		img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1

		# Gaussian Gradient Magnitude Edge Detection
		img = scipy.ndimage.gaussian_gradient_magnitude(original_image_array, sigma=s)
		img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1

		# Difference of Gaussians
		k = 1.7  # determined by trial and error
		tmp1_array = scipy.ndimage.gaussian_filter(original_image_array, sigma=s*k) 
		tmp2_array = scipy.ndimage.gaussian_filter(original_image_array, sigma=s)
		img = (tmp1_array - tmp2_array)
		img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1
		
		# Structure Tensor Eigenvalues
		Axx,Axy,Ayy = feature.structure_tensor(Image.fromarray(original_image_array), sigma=s)
		large_array,small_array = feature.structure_tensor_eigvals(Axx,Axy,Ayy)
		img = cv2.normalize(large_array,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1
		img = cv2.normalize(small_array,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1

		# Hessian Matrix
		Hrr,Hrc,Hcc = feature.hessian_matrix(Image.fromarray(original_image_array), sigma=s, order='rc')
		img = cv2.normalize(Hrr,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1
		img = cv2.normalize(Hrc,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1
		img = cv2.normalize(Hcc,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1


	# Generate proximity map (to center of image)
	newimg = center_proximity(original_image_array)
	img = cv2.normalize(newimg,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	images[x] = img.flatten(order='F'); x+=1

	# Simple Intensity Threshholding
	img = cv2.GaussianBlur(original_image_array, (5,5), cv2.BORDER_WRAP)
	img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	ret,img = cv2.threshold(img,205,255,cv2.THRESH_BINARY)
	images[x] = img.flatten(order='F'); x+=1


	# Flip the orientation of the flattened 2D array
	images=numpy.transpose(images,(1,0))

	preprocess_counter += 1
	print("Processed raw image %d" %preprocess_counter)
	
	return(images)






##############################################################################
#
# output_preprocessed()
#
# NOTE:  ***CURRENTLY DEPRECATED AND UNUSED***
#
# Output preprocessed images into a folder for inspection and debugging 
# purposes only
#
def output_preprocessed(images, dir):
	index = 0
	for f in images:
		pathname = dir + "/" + str(index) + ".tif"
		x = Image.fromarray(f)
		x.save(pathname, "TIFF")
		index = index + 1
	
