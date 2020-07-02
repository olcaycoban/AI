# -*- coding: utf-8 -*-
"""AlzheimerTORCH.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BLErDiCk2ZRrYTvKzYW-lpkw9i-_iUrh
"""

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle

! chmod 666 ~/.kaggle/kaggle.json
!kaggle datasets download -d tourist55/alzheimers-dataset-4-class-of-images

from zipfile import ZipFile
file_name="alzheimers-dataset-4-class-of-images.zip"

with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print("Done")

import os
print(os.listdir('Alzheimer_s Dataset/test'))

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import os
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
from fastai import *
from fastai.vision import *

img = open_image(Path('Alzheimer_s Dataset/train/VeryMildDemented/verymildDem972.jpg'))
print(img.shape)
img

PATH = Path('Alzheimer_s Dataset/')

transform = get_transforms(max_rotate=7.5, max_zoom=1.15, max_lighting=0.15, max_warp=0.15, p_affine=0.8, p_lighting = 0.8, 
                           xtra_tfms= [pad(mode='zeros'), symmetric_warp(magnitude=(-0.1,0.1)), cutout(n_holes=(1,3), length=(5,5))])

data = ImageDataBunch.from_folder(PATH, train="train/",
                                  test="test/",
                                  valid_pct=.4,
                                  ds_tfms=transform,
                                  size=112,bs=32, 
                                  ).normalize(imagenet_stats)

data.show_batch(rows=3, figsize=(10,10))

Category.__eq__ = lambda self, that: self.data == that.data
Category.__hash__ = lambda self: hash(self.obj)
Counter(data.train_ds.y)

Counter(data.test_ds.y)

learn = cnn_learner(data, models.vgg16_bn, metrics=error_rate, wd=1e-1)
learn.fit_one_cycle(17, max_lr=5e-4)