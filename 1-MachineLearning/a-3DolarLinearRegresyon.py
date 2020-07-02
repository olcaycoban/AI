import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data=pd.read_csv('2016dolaralis.csv')
print(data)

x = data["Gun"]
y = data["Fiyat"]
x=np.array(x)
y=np.array(y)
x=x.reshape(251,1)
y=y.reshape(251,1)
plt.scatter(x,y)


tLinear=LinearRegression()
tLinear.fit(x,y)
tLinear.predict(x)
plt.plot(x,tLinear.predict(x),c="red")

tahminPolinom=PolynomialFeatures(degree=3)
Xyeni=tahminPolinom.fit_transform(x)

tPolinom=LinearRegression()
tPolinom.fit(Xyeni,y)
tPolinom.predict(Xyeni)
plt.plot(x,tPolinom.predict(Xyeni),c="yellow")
plt.show()

hatakaresilinear=0
hatakaresipolinom=0

for i in range(len(x)):
    hatakaresilinear=hatakaresilinear+(float(y[i])-float(tLinear.predict(x)[i]))**2

for i in range(len(Xyeni)):
    hatakaresipolinom=hatakaresipolinom+(float(y[i])-float(tPolinom.predict(Xyeni)[i]))**2

print("hatakaresiLinerar {0}  \nhatakaresiPolinom {1}\n".format(hatakaresilinear,hatakaresipolinom))

minhatakaresi = 100
mindegree = 0
hatakaresipolinom = 0

for a in range(100):

    tahminpolinom = PolynomialFeatures(degree=a + 1)
    Xyeni = tahminpolinom.fit_transform(x)

    polinommodel = LinearRegression()
    polinommodel.fit(Xyeni, y)
    polinommodel.predict(Xyeni)
    for i in range(len(Xyeni)):
        hatakaresipolinom = hatakaresipolinom + (float(y[i]) - float(polinommodel.predict(Xyeni)[i])) ** 2
    print(a + 1, "inci dereceden fonksiyonda hata,", hatakaresipolinom)
    if hatakaresipolinom<minhatakaresi:
        minhatakaresi=hatakaresipolinom
        mindegree=a+1
    hatakaresipolinom = 0

print("{0}. derecedeki hata en azdır.\nHata oranı ise : {1} 'dir.".format(mindegree,minhatakaresi))

"""tahminPolinom2=PolynomialFeatures(degree=2)
Xyeni=tahminPolinom2.fit_transform(x)

tPolinom2=LinearRegression()
tPolinom2.fit(Xyeni,y)
tPolinom2.predict(Xyeni)
plt.plot(x,tPolinom2.predict(Xyeni),c="black")"""