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

# pretrained_model_path = r'models/liver_adam1e-3_OnlyProjectData_unet.h5'
pretrained_model_path = None
name = 'liver_adam1e-3_OnlyProjectData_unet_3_channels'
batch_size = 5

train_path2images = r'C:\Users\Maya.WG32128\Google Drive\imaging_advanced_course\project\TrainData\ct\train'
train_path2masks = r'C:\Users\Maya.WG32128\Google Drive\imaging_advanced_course\project\TrainData\seg\train'
train_gen = train_generator(train_path2images, train_path2masks, batch_size=batch_size)

val_path2images = r'C:\Users\Maya.WG32128\Google Drive\imaging_advanced_course\project\TrainData\ct\val'
val_path2masks = r'C:\Users\Maya.WG32128\Google Drive\imaging_advanced_course\project\TrainData\seg\val'
valid_gen = train_generator(val_path2images, val_path2masks, batch_size=batch_size)

train_size = len([x for x in sorted(os.listdir(train_path2images)) if x[-4:] == '.png'])  # Read all the images
val_size = len([x for x in sorted(os.listdir(val_path2images)) if x[-4:] == '.png'])  # Read all the images

# model = Nest_Net(img_rows=512, img_cols=512, color_type=1, num_class=2)
model = U_Net(img_rows=512, img_cols=512, color_type=1, num_class=3)
model.summary()

if pretrained_model_path:
    # load weights
    model.load_weights(pretrained_model_path)

# get the loss function
model_dice = dice_coef_loss

model.compile(optimizer=Adam(1e-3), loss=model_dice, metrics=[dice_coef, precision]) # Compile model with optimizer, loss and metrics
# Define callbacks that act at the end of epoch
weight_saver = ModelCheckpoint('models/{}.h5'.format(name), monitor='val_dice_coef',
                                              save_best_only=True, save_weights_only=True)

lr_scd = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)

early_stopping = EarlyStopping(monitor='val_dice_coef', mode="max", patience=10)

tensorboard = TensorBoard(log_dir='./logs/{}'.format(name), histogram_freq=0,
                          write_graph=True, write_images=True)
# Train the model
hist = model.fit_generator(generator=train_gen,  # Training data supplied by generator
                           steps_per_epoch=train_size/batch_size,  # Number of mini-batches to run in one epoch
                           validation_data=valid_gen,  # Validation data supplied as generator
                           validation_steps=val_size/batch_size,
                           epochs=1000, verbose=1,
                           callbacks=[weight_saver, early_stopping, lr_scd, tensorboard])

plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['dice_coef'], color='b')
plt.plot(hist.history['val_dice_coef'], color='r')
plt.show()