import os
import numpy as np
import cv2

import nibabel as nib


path2nifiti = r'C:\Users\Maya.WG32128\Google Drive\imaging_advanced_course\project\nifiti_data\Training Batch 1'
path2savePng = r'C:\Users\Maya.WG32128\Google Drive\imaging_advanced_course\project\convert_data'
all_seg = [x for x in sorted(os.listdir(path2nifiti)) if x[4:]=='segm'] # Read all the segmention
all_volumes = [x for x in sorted(os.listdir(path2nifiti)) if x[4:]=='volu'] # Read all the volume
for volume_name, seg_name in zip(all_volumes, all_seg):
    volume_path = os.path.join(path2nifiti, volume_name)
    seg_path = os.path.join(path2nifiti, seg_name)
    volume = nib.load(volume_path)
    seg = nib.load(seg_path)
    volume_data = volume.get_fdata()
    seg_data = seg.get_fdata()
    for i in range(len(volume_data[0,0,:])):
        img, label = volume_data[...,i], seg_data[...,i]
        cv2.imwrite('{}\{}_{}.png'.format(path2savePng, volume_name, i), img)
        cv2.imwrite('{}\{}_{}.png'.format(path2savePng, seg_name, i), label)
        pass
    print('Finish working on: volume {}, seg {}'.format(volume_name, seg_name))
print('Done')

