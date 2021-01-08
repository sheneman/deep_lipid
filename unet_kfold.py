###############################################################################
# unet_kfold.py
#
# sheneman@uidaho.edu
#
# Perform a "manual" k-fold cross validation with the UNet CNN model.  This 
# assumes that a directory hierarchy has been configured and populated (see
# unet_kfold_prepare.py).   
#
# This program will save the trained models for each fold in the appropriate 
# place in the directory hierarchy.  Those trained models can then be loaded 
# by unet_classify.py to compute the predicted output (binary semantic 
# segmentation maps) against the hold-out (test) set within each fold and then 
# scored against the "true" binary labels using various # model 
# effectiveness metrics (score.py).
#
# Usage:
#  python unet_kfold.py
#
################################################################################

import tensorflow as tf
import pandas as pd
import pickle
from math import ceil
from time import sleep, time
from datetime import datetime
from random import seed
from random import random
import os
import sys
import numpy as np
from keras.models import load_model
import keras.callbacks
from keras.callbacks import History
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils.multiclass import type_of_target
import matplotlib.pyplot as plt

from unet_model import *   # local python file containing the UNet architecture
from unet_data import *    # local python file containing helper functions


# for debugging
np.set_printoptions(threshold=sys.maxsize)

# for history diagnostics and telemetry
history = History()

# the path to our k-fold hierarchy (see unet_kfold_prepare.py)
kfold_path = "./runs/unet/kfold"

print("TENSORFLOW VERSION: %s" %(tf.__version__))

# set our random seed based on current time
now = int(time())

seed(now)
np.random.seed(now)
tf.random.set_seed(now)
os.environ['PYTHONHASHSEED'] = str(now)

# Some basic training parameters.  These are manually derived for now
EPOCHS = 1000
BATCH_SIZE = 5
TRAIN_SIZE = 4000
STEPS_PER_EPOCH = ceil(TRAIN_SIZE/BATCH_SIZE)
NUM_FOLDS = 5

print("Number of Folds for k-fold Cross Validation: %d" %NUM_FOLDS)

for fold in range(0,NUM_FOLDS):

	print("FOLD %d: Building training generator..." %fold)
	trainpath = os.path.join(kfold_path, str(fold))
	trainpath = os.path.join(trainpath, "train")
	lipid_train = trainGenerator(BATCH_SIZE,trainpath,'padded_raw8','padded_binary',seed=now)

	model = unet()   # new blank UNet model

	# save checkpoints at the end of each epoch
	checkpoint_path = os.path.join(kfold_path, str(fold))
	checkpoint_path = os.path.join(checkpoint_path, "unet_lipid_checkpoint.h5")
	model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss',verbose=1, save_best_only=True)

	# perform the model training for this fold
	print("FOLD %d: Fitting model with training data..." %fold)
	model.fit_generator(lipid_train,steps_per_epoch=STEPS_PER_EPOCH,epochs=EPOCHS,callbacks=[model_checkpoint,history])
	print("Done...")

	# save the model
	print("FOLD %d: Saving model..." %fold)
	modelpath = os.path.join(kfold_path, str(fold))
	modelfile = os.path.join(modelpath, "unet.model.h5")
	model.save(modelfile)

	# save the training history telemetry
	print("FOLD %d: Saving training history..." %fold)
	historypath = os.path.join(kfold_path, str(fold))
	historypath = os.path.join(historypath, "unet.training_history.pickle")
	with open(historypath, 'wb') as histfile:
	    pickle.dump(history.history, histfile)

exit(0)

