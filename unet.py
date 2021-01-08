###############################################################################
# unet.py
#
# sheneman@uidaho.edu
#
#
# The primary tool for training a UNet CNN model given a directory of training
# images and a corresonding directory of masks (i.e. "true" binary labels).
#
# This tool combines a stream of unsigned 8-bit, pre-padded (256x256) training
# images and their corresponding pre-padded binary masks and trains a Unet CNN.
#
# 32-bit float raw images must be initially preprocessed:
#    1) Scaled to normalized 8-bit unsigned representation
#    2) Padded to be 256x256 (see pad.py)
#
# The binary masks must also be padded to a consistent 256x256 dimension.
#
# Most of the training parameters (Epochs, Batch size, etc.) are hardcoded.
#
# The output is a trained model and the model training history.
#
# Usage:
#  python unet.py
#
###############################################################################


import tensorflow as tf
import pickle
from math import ceil
from time import sleep, time
from datetime import datetime
from random import seed
from random import random
import os
import numpy as np
from keras.models import load_model
from keras.callbacks import History

from unet_model import *   # local python file containing the UNet architecture
from unet_data import *    # local python file containing helper functions

history = History()


print("TENSORFLOW VERSION: %s" %(tf.__version__))

# set our random seed based on current time
now = int(time())

seed(now)
np.random.seed(now)
tf.random.set_seed(now)
os.environ['PYTHONHASHSEED'] = str(now)

# Some basic training parameters
EPOCHS = 1000
BATCH_SIZE = 5
TRAIN_SIZE = 5000
STEPS_PER_EPOCH = ceil(TRAIN_SIZE/BATCH_SIZE)

# Create a streaming training set generator based on flows_from_directory()
lipid_train = trainGenerator(BATCH_SIZE,'./runs/unet/complete/train','padded_raw8','padded_binary',seed=now)

model = unet() # instantiate a blank UNet CNN model

# configure a checkpoint that will be saved at the end of every epoch (if improved)
model_checkpoint = ModelCheckpoint('unet_lipid_checkpoint.h5', monitor='loss',verbose=1, save_best_only=True)

# train the UNet CNN model 
model.fit_generator(lipid_train,steps_per_epoch=STEPS_PER_EPOCH,epochs=EPOCHS,callbacks=[model_checkpoint,history])

# save the trained model to disk
print("Saving model...")
model.save("./runs/unet/complete/unet.model.h5")

# save the training history for the model for diagnostics
print("Saving training history...")
with open('./runs/unet/complete/unet_training_history.pickle', 'wb') as histfile:
    pickle.dump(history.history, histfile)

