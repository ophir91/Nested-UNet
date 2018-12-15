import os # os specific actions
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # image processing
import matplotlib.pyplot as plt # plots


# deep learning framework
import keras.backend as K
from keras import utils
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *

# local imports
from model import *
from loss import *
from my_generator import train_generator



train_path2images = r'C:\Users\ophir\OneDrive\Ophir\University\Master\dimot\TrainData\ct\train'
train_path2masks = r'C:\Users\ophir\OneDrive\Ophir\University\Master\dimot\TrainData\seg\train'
train_gen = train_generator(train_path2images, train_path2masks, batch_size=2)

val_path2images = r'C:\Users\ophir\OneDrive\Ophir\University\Master\dimot\TrainData\ct\val'
val_path2masks = r'C:\Users\ophir\OneDrive\Ophir\University\Master\dimot\TrainData\seg\val'
valid_gen = train_generator(val_path2images, val_path2masks, batch_size=2)

model = Nest_Net(img_rows=512, img_cols=512, color_type=1, num_class=3)
model.summary()

# get the loss function
model_dice = dice_coef_loss()

model.compile(optimizer=Adam(2e-4), loss=model_dice, metrics=[dice_coef, precision]) # Compile model with optimizer, loss and metrics
# Define callbacks that act at the end of epoch
weight_saver = ModelCheckpoint('models/lung.h5', monitor='val_dice_coef',
                                              save_best_only=True, save_weights_only=True)

lr_scd = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)

early_stopping = EarlyStopping(monitor='val_dice_coef', mode="max", patience=5)
# Train the model
hist = model.fit_generator(generator=train_gen,  # Training data supplied by generator
                           steps_per_epoch=20,  # Number of mini-batches to run in one epoch
                           validation_data=valid_gen,  # Validation data supplied as generator
                           validation_steps=5,
                           epochs=10, verbose=2,
                           callbacks=[weight_saver, early_stopping, lr_scd])

plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['dice_coef'], color='b')
plt.plot(hist.history['val_dice_coef'], color='r')
plt.show()