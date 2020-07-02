from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD

from  sklearn.preprocessing import  Imputer
import numpy as np
import pandas as pd

veri = pd.read_csv("breast-cancer-wisconsin.data")

veri.replace('?', -99999, inplace='true')
#veri.drop(['id'], axis=1)
veriyeni = veri.drop(['1000025'],axis=1)

giris = veriyeni[:,0:8]
cikis = veriyeni[:,9]