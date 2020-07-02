import numpy
import keras.layers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import max_norm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import matplotlib.pyplot as plt
import keras.layers

from scipy import ndimage
from scipy import misc
import numpy
from matplotlib import pyplot
from scipy.misc import toimage
import scipy.misc

from PIL.Image import fromarray

(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()

X_train=X_train.astype('float32')
Y_train=Y_train.astype('float32')

#normalize etmek için 255'e bölüyrouz.
X_train=X_train/255
Y_train=Y_train/255

#normalize etmek için
Y_train=np_utils.to_categorical(Y_train)
X_train=np_utils.to_categorical(X_train)
num_classes=Y_test.shape[1]

for i in range(0,9):
    plt.subplot(330+i+1)
    plt.imshow(fromarray(X_train[i]))

plt.show()

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(3,32,32),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dense())