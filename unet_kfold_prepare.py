###############################################################################
# unet_kfold_prepare.py
#
# sheneman@uidaho.edu
#
# This is an auxillary tool to prepare a directory hierarchy populated with
# raw and binary training images for use in doing a "manual" k-fold cross
# validation of Keras/TensorFlow models.
#
# Usage:
#  python unet_kfold_prepare.py
#
################################################################################

import os, sys
from sklearn.model_selection import KFold
import numpy as np
import shutil
from shutil import copyfile

kfold_splits = 5
raw_source = "./runs/unet/padded_raw8"
bin_source = "./runs/unet/padded_binary"
kfold_directory = "./runs/unet/kfold"
filenames_file = "./runs/training_5000.txt"

# prep the destination directories
for i in range(0,kfold_splits):
	path = os.path.join(kfold_directory, str(i))
	if(os.path.isdir(path)):
		shutil.rmtree(path)
	os.mkdir(path)
	trainpath = os.path.join(path, "train")
	os.mkdir(trainpath)
	os.mkdir(os.path.join(trainpath, "padded_raw8"))
	os.mkdir(os.path.join(trainpath, "padded_binary"))
	testpath  = os.path.join(path, "test")
	os.mkdir(testpath)
	os.mkdir(os.path.join(testpath, "padded_raw8"))
	os.mkdir(os.path.join(testpath, "padded_binary"))


# get the list of filenames in our training set
train_filenames      = np.asarray([line.rstrip('\n') for line in open(filenames_file)])
print(len(train_filenames))
print(type(train_filenames))
print(train_filenames[200])


# do the k-fold split on the list of filenames
kf = KFold(n_splits = kfold_splits, shuffle=True)
kf.get_n_splits(train_filenames)
print(kf)

# copy the right files into the right places in our k-fold cross-validation
# folder hierarchy
fold = 0
for train_index, test_index in kf.split(train_filenames):
	print("TRAIN:", train_index, "TEST:", test_index)
	x_train, x_test = train_filenames[train_index], train_filenames[test_index]

	cnt = 0
	for file in x_train:		

		print("FOLD: %d, TRAIN COUNT: %d" %(fold,cnt))

		# padded_raw8
		src_file = os.path.join(raw_source, file)
		dst_file = os.path.join(kfold_directory, str(fold))
		dst_file = os.path.join(dst_file, "train")
		dst_file = os.path.join(dst_file, "padded_raw8")
		dst_file = os.path.join(dst_file, file)
		copyfile(src_file, dst_file)
		#print("COPY: %s -> %s" %(src_file, dst_file));

		# padded_binary
		src_file = os.path.join(bin_source, file)
		dst_file = os.path.join(kfold_directory, str(fold))
		dst_file = os.path.join(dst_file, "train")
		dst_file = os.path.join(dst_file, "padded_binary")
		dst_file = os.path.join(dst_file, file)
		copyfile(src_file, dst_file)
		#print("COPY: %s -> %s" %(src_file, dst_file));
	
		cnt = cnt + 1


	cnt = 0
	for file in x_test:		
		print("FOLD: %d, TEST COUNT: %d" %(fold,cnt))

		# padded_raw8
		src_file = os.path.join(raw_source, file)
		dst_file = os.path.join(kfold_directory, str(fold))
		dst_file = os.path.join(dst_file, "test")
		dst_file = os.path.join(dst_file, "padded_raw8")
		dst_file = os.path.join(dst_file, file)
		copyfile(src_file, dst_file)
		#print("COPY: %s -> %s" %(src_file, dst_file));

		# padded_binary
		src_file = os.path.join(bin_source, file)
		dst_file = os.path.join(kfold_directory, str(fold))
		dst_file = os.path.join(dst_file, "test")
		dst_file = os.path.join(dst_file, "padded_binary")
		dst_file = os.path.join(dst_file, file)
		copyfile(src_file, dst_file)
		#print("COPY: %s -> %s" %(src_file, dst_file));

		cnt = cnt + 1 


	fold = fold + 1

