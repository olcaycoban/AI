import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = pd.read_csv("linear.csv")
print(data)
x = data["metrekare"]
y = data["fiyat"]
x = pd.DataFrame.as_matrix(x)
y = pd.DataFrame.as_matrix(y)
#print(x)
#print(y)
plt.scatter(x,y)

m,b = np.polyfit(x,y,1)
a = np.arange(150)
plt.scatter(x,y)
plt.plot(m*a+b)


z = int(input("Kaç metrekare?"))
tahmin = m*z+b
print(tahmin)
plt.scatter(z,tahmin,c="red",marker=">")
plt.show()
print("y=",m,"x+",b)
hatakaresi=0
for i in range(int(len(y))):
    hatakaresi+=(y[i]-m*x[i]+b)**2
print(hatakaresi)
#24286.758970965493