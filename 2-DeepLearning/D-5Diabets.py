from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import regularizers ,optimizers
from keras.layers import Dropout

df = pd.read_csv('diabetes.csv')
df.drop_duplicates(inplace = True)
df.shape
df.isnull().sum()
dataset = df.values

X = dataset[:,0:8]
Y = dataset[:,8]

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

X_train , X_val_and_test ,Y_train , Y_val_and_test=train_test_split(X_scale,Y,test_size=0.1)
X_val,X_test,Y_val,Y_test=train_test_split(X_val_and_test,Y_val_and_test,test_size=0.5)

model=Sequential([
    Dense(1024,activation='relu',input_shape=(8,)),
    Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0,3),
    Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0,3),
    Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0,3),
    Dense(1,activation='sigmoid'),
])

optimizer=optimizers.Adamax(lr=0.5)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val,Y_val))

"""plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('val_loss')
plt.legend(['Train','Val'],loc='upper right')
plt.show()"""

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc='lower right')
plt.show()

prediction = model.predict(X_test)
prediction  = [1 if y>=0.5 else 0 for y in prediction] #Threshold
print(prediction)
print(Y_test)

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = model.predict(X_train)
pred  = [1 if y>=0.5 else 0 for y in pred] #Threshold
print(classification_report(Y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(Y_train,pred))
print()
print('Accuracy: ', accuracy_score(Y_train,pred))
print()

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = model.predict(X_test)
pred  = [1 if y>=0.5 else 0 for y in pred] #Threshold
print(classification_report(Y_test ,pred ))
print('Confusion Matrix: \n',confusion_matrix(Y_test,pred))
print()
print('Accuracy: ', accuracy_score(Y_test,pred))
print()

model.evaluate(X_test,Y_test)[1]