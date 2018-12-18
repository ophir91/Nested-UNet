import os
import numpy as np
import cv2

import nibabel as nib

img = nib.load(example_filename)
data = img.get_fdata()
