from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras import regularizers ,optimizers
from keras.layers import Dropout

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_colwidth', 10)
pd.set_option('display.width', None)


df=pd.read_csv("housepricedata.csv")
#print(df)
dataset=df.values
X=dataset[:,0:10]
Y=dataset[:,10]

min_max_scaler=preprocessing.MinMaxScaler()
X_scale=min_max_scaler.fit_transform(X)

X_train , X_val_and_test ,Y_train , Y_val_and_test=train_test_split(X_scale,Y,test_size=0.3)
X_val,X_test,Y_val,Y_test=train_test_split(X_val_and_test,Y_val_and_test,test_size=0.5)

model=Sequential([
    Dense(1000,activation='relu',kernel_regularizer=regularizers.l2(0.01),input_shape=(10,)),
    Dropout(0,3),
    Dense(1000, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0,3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0,3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0,3),
    Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(0.01)),
])

opt=optimizers.SGD(learning_rate=0.05)
model.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

hist=model.fit(X_train,Y_train,
               epochs=15,batch_size=32,
               validation_data=(X_val,Y_val))

model.evaluate(X_test,Y_test)[1]

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('val_loss')
plt.legend(['Train','Val'],loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc='lower right')
plt.show()
