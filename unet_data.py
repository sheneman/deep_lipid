###############################################################################
# unet_data.py
#
# sheneman@uidaho.edu
#
# Support and wrapper functions for streaming training and test data, loading
# binary labels, and saving classification output.
#
################################################################################

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle
import numpy as np 
import os
import sys
import glob
import skimage.io as io
from skimage import img_as_ubyte
import skimage.transform as trans

# for debugging
np.set_printoptions(threshold=sys.maxsize)


###############################################################################
# 
# trainGenerator()
#
# A wrapper function for constructing a Keras/TensorFlow training set by 
# streaming both the raw and mask (i.e. binary label) data from directories, 
# zipping them together and normalizing them as floats between [0,1] for use
# in the UNet CNN model.
#
###############################################################################
def trainGenerator(batch_size,train_path,image_folder,mask_folder,image_color_mode = "grayscale",
			mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
			flag_multi_class = False,num_class = 2,target_size = (256,256),seed = 1):

	image_datagen = ImageDataGenerator()
	image_generator = image_datagen.flow_from_directory(
		train_path,
		classes = [image_folder],
		class_mode = None,
		color_mode = image_color_mode,
		target_size = target_size,
		batch_size = batch_size,
		save_prefix  = image_save_prefix,
		seed = seed)

	mask_datagen = ImageDataGenerator()
	mask_generator = mask_datagen.flow_from_directory(
		train_path,
		classes = [mask_folder],
		class_mode = None,
		color_mode = mask_color_mode,
		target_size = target_size,
		batch_size = batch_size,
		save_prefix  = mask_save_prefix,
		seed = seed)

	# combine streams using zip
	train_generator = zip(image_generator, mask_generator)

	for (img,mask) in train_generator:
		yield (img/255,mask/255) # normalize to [0,1] floating point representation






###############################################################################
#
# testGenerator()
#
# A function for streaming test images during the classification step.
#
###############################################################################
def testGenerator(testdir_path, filenames):
	for f in filenames:
		fullpath = testdir_path + '/' + f
		img = io.imread(fullpath,as_gray = True)
		img = img / 255
		img = np.reshape(img,img.shape+(1,))
		img = np.reshape(img,(1,)+img.shape)

		yield img



###############################################################################
#
# loadMasks()
#
# Load just the masks (i.e. "true" binary labels) separately
#
###############################################################################
def loadMasks(maskdir_path, filenames):
	mask_vector = []
	for f in filenames:
		fullpath = os.path.join(maskdir_path, f)
		img = io.imread(fullpath) / 255; 
		mask_vector = np.append(mask_vector, img.flatten())

	return(mask_vector)




###############################################################################
#       
# saveResults()
#               
# Save classified image to disk, applying threshold to convert probabilistic 
# output from UNet CNN to a binary representation in unsigned 8-bit form.
#               
###############################################################################
def saveResult(save_path,thresh_integer,filenames,results):
	for i,item in enumerate(results):
		img = item[:,:,0]
		thresh = float(thresh_integer)/100.0
		bin_img = img > thresh  # this performs threshold to binary value
		print("saveResult().  thresh=%f" %thresh)
		io.imsave(os.path.join(save_path,filenames[i]),img_as_ubyte(bin_img))
