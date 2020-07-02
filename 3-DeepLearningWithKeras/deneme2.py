import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

veri = pd.read_csv("breast-cancer-wisconsin.data")
veri.replace('?', -99999, inplace=True)
veriyeni = veri.drop(['1000025'],axis=1)

imp = Imputer(missing_values=-99999, strategy="mean",axis=0)
veriyeni = imp.fit_transform(veriyeni)

giris = veriyeni[:,0:8]
cikis = veriyeni[:,9]

model=Sequential()
model.add(Dense(64,input_dim=8))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(giris,cikis,epochs=50,batch_size=32,validation_split=0.15)

tahmin = np.array([5,5,5,8,10,8,7,3]).reshape(1,8)
print(model.predict_classes(tahmin))

tahmin = np.array([7,9,3,5,6,8,7,7]).reshape(1,8)
print(model.predict_classes(tahmin))