import os
import numpy as np
import cv2
# import keras
# import keras.preprocessing
# import nibabel as nib
import matplotlib as plt

def get_image(path2image, input_size):
    img = None
    if path2image[-4:] == '.png':
        img = cv2.imread('{}'.format(path2image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.resize(img, (input_size, input_size))
    return img


def preprocess_input(image, mask):
    """
    :param image:
    :param mask:
    :return:
    """
    # TODO: write the preprocess: --- Rotate Image --- Flip Image
    # img, mask = randomShiftScaleRotate(img, mask,
    #                                    shift_limit=(-0.0625, 0.0625),
    #                                    scale_limit=(-0.1, 0.1),
    #                                    rotate_limit=(-0, 0))
    # img, mask = randomHorizontalFlip(img, mask)
    image = np.expand_dims(image, axis=2)
    # mask = np.expand_dims(mask, axis=2)
    new_mask=np.zeros((mask.shape[0],mask.shape[1],2), dtype=np.uint8)
    inds = np.where(mask==127)
    new_mask[inds[0], inds[1],0] = 1.
    inds2 = np.where(mask==255)
    new_mask[inds2[0], inds2[1],1] = 1.


    return image, new_mask


def train_generator(path2images, path2masks, batch_size, input_size=512):
    all_images = [x for x in sorted(os.listdir(path2images)) if x[-4:]=='.png'] # Read all the images
    all_masks = [x for x in sorted(os.listdir(path2masks)) if x[-4:]=='.png']  # Read all the masks

    while True:
        for start in range(0, len(all_images), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(all_images))
            ids_train_batch = all_images[start:end]
            ids_masks_batch = all_masks[start:end]
            for id_image, id_mask in zip(ids_train_batch, ids_masks_batch):
                img = get_image(os.path.join(path2images, id_image), input_size)
                mask = get_image(os.path.join(path2masks, id_mask), input_size)
                img, mask = preprocess_input(img, mask)

                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32)
            yield x_batch, y_batch

#
# path2images = r'C:\Users\Maya.WG32128\Google Drive\imaging_advanced_course\project\TrainData\ct\train'
# path2masks = r'C:\Users\Maya.WG32128\Google Drive\imaging_advanced_course\project\TrainData\seg\train'
# gen = train_generator(path2images, path2masks, batch_size=32)
# for x_batch, y_batch in gen:
#     print(y_batch.shape)
# pass
