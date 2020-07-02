import pandas as pd
import quandl
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split


df=quandl.get('WIKI/GOOGL')
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100
df['PCT_Change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100

df=df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

forecast_col='Adj. Close'
df.fillna(-99999,inplace=True)

forecast_out=int(math.ceil(0.01*len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X=np.array(df.drop(['label'],1))
y=np.array(df['label'])
X=preprocessing.scale(X)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

clf=LinearRegression(n_jobs=-1)
clf.fit(X_train,y_train)
accurancy=clf.score(X_test,y_test)
print(accurancy)