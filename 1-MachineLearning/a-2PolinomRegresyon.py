import numpy as np
import pandas as pd
from  sklearn.preprocessing import PolynomialFeatures as pf
import matplotlib.pyplot as plt
from sklearn.linear_model import perceptron
from sklearn import linear_model

data = pd.read_csv("linear.csv")

x = data["metrekare"]
y = data["fiyat"]


x = np.array(x)
y= np.array(y)

#a,b,c,d,e =  np.polyfit(x,y,4)
#a,b,c,d =  np.polyfit(x,y,3)
#a,b=  np.polyfit(x,y,1)
a,b,c =  np.polyfit(x,y,2)
z = np.arange(150)
plt.scatter(x,y)
plt.plot(z,a*z**2+b*z,+c)
#plt.plot(z,a*z**3+b*z**2,c*z+d)
#plt.plot(z,a*z**4+b*z**4+c*z**2,+d*z+e)
plt.show()
print(a,"x +",b)
hatakaresi=0
for i in range(int(len(y))):
    hatakaresi+=(y[i]-(a*x[i]**2+b*y[i]+c))**2
print(hatakaresi)

#162623.11714037473