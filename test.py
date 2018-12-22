import matplotlib.pyplot as plt # plots

from keras.callbacks import *

# local imports
from model import *
from loss import *
from my_generator import train_generator
import cv2


pretrained_model_path = r'models/liver_adam1e-3_OnlyProjectData_unet_4_channels.h5'

# x_train = cv2.imread(r'C:\Users\Maya.WG32128\Google Drive\imaging_advanced_course\project\TrainData\ct\val\ct_8_472.png')
# y_train = cv2.imread(r'C:\Users\Maya.WG32128\Google Drive\imaging_advanced_course\project\TrainData\seg\val\seg_8_472.png')
# x_train = cv2.cvtColor(x_train, cv2.COLOR_BGR2GRAY)
# x_train = x_train/255
model = U_Net(img_rows=512, img_cols=512, color_type=1, num_class=4)
model.summary()

if pretrained_model_path:
    # load weights
    model.load_weights(pretrained_model_path)

path2images = r'C:\Users\Maya.WG32128\Google Drive\imaging_advanced_course\project\TrainData\ct\train'
path2masks = r'C:\Users\Maya.WG32128\Google Drive\imaging_advanced_course\project\TrainData\seg\train'
gen = train_generator(path2images, path2masks, batch_size=1)

fix, ax = plt.subplots(1,4)
# ax.grid(False)
for x_train, y_train in gen:
    output = model.predict(x_train)
    ax[0].imshow(x_train[0,:,:,0], cmap='gray')
    ax[1].imshow(y_train[0,:,:,0], cmap='gray')
    ax[2].imshow(output[0,:,:,0], cmap='gray')
    ax[3].imshow(output[0,:,:,1], cmap='gray')
    plt.show()
pass