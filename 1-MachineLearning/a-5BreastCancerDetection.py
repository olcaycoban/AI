import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

veri = pd.read_csv("breast-cancer-wisconsin.data")
veri.replace('?', -99999, inplace=True)
veri=veri.drop(['id'], axis=1)

y=veri.benormal
x=veri.drop(['benormal'],axis=1)

imp=Imputer(missing_values=-99999,strategy='mean',axis=0)
x=imp.fit_transform(x)
"""
for i in range(3,100,2):
    tahmin=KNeighborsClassifier(n_neighbors=i,weights='uniform',algorithm='auto',leaf_size=30,p=2,metric='euclidean',metric_params=None,n_jobs=1)
    tahmin.fit(x,y)
    ytahmin=tahmin.predict(x)
    basari_rank=accuracy_score(y,ytahmin,normalize=True,sample_weight=None)
    print("Basari Puani {0}".format(basari_rank))
"""
tahmin=KNeighborsClassifier(n_neighbors=3,weights='uniform',algorithm='auto',leaf_size=30,p=2,metric='euclidean',metric_params=None,n_jobs=1)
tahmin.fit(x,y)
ytahmin=tahmin.predict(x)
basari = accuracy_score(y, ytahmin, normalize=True, sample_weight=None)
print("Yüzde",basari*100," oranında:" )

#print(tahmin.predict([1,2,2,2,3,2,1,2,3,2]))