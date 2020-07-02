import numpy as np 
import pandas as pd 
from sklearn import preprocessing

# Data Visualization
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6)
plt.style.use('fivethirtyeight')

from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('penguins_size.csv')
data.head(10)

data.info()
data.isna().sum()

data.describe()

imputer = SimpleImputer(strategy='most_frequent')
data.iloc[:,:] = imputer.fit_transform(data)

import matplotlib.pyplot as plt

species_percentage = data['species'].value_counts()

labels = species_percentage.index
values = species_percentage.values

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
ax.pie(values, labels = labels,autopct='%1.8f%%')
plt.show()

sns.countplot(data['species'], data=data, palette='bone')
plt.title('Count - Species Distribution for Palmer Archipelago Penguins')
plt.show()

import matplotlib.pyplot as plt

species_percentage = data['island'].value_counts()

labels = species_percentage.index
values = species_percentage.values

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
ax.pie(values, labels = labels,autopct='%1.8f%%')
plt.show()

sns.countplot(data['island'], data=data, palette='bone')
plt.title('Count - Island Distribution in Palmer Archipelago')
plt.show()

data['island'].value_counts()

data['sex'].value_counts()

sex = data['sex'].value_counts()

labels = sex.index
values = sex.values

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
ax.pie(values, colors=["red","blue","green"],labels = labels,autopct='%1.8f%%')
plt.show()

print('Flipper Length Distribution')
sns.violinplot(data=data, x="species", y="flipper_length_mm", size=8)
plt.show()

print('Flipper Length Distribution')
sns.boxplot(data=data, x="species", y="body_mass_g")
plt.show()

print('culmen_length vs culmen_depth')
sns.scatterplot(data=data, x='body_mass_g', y='flipper_length_mm', hue='species')
plt.show()

print('Flipper Length Distribution')
sns.violinplot(data=data, x="species", y="culmen_length_mm", size=8)
plt.show()

print('Flipper Length Distribution')
sns.boxplot(data=data, x="species", y="culmen_depth_mm")
plt.show()

print('culmen_length vs culmen_depth')
sns.scatterplot(data=data, x='culmen_depth_mm', y='culmen_length_mm', hue='species')
plt.show()

sns.stripplot(data['species'], data['culmen_length_mm'], palette = 'Reds')
plt.title("Species vs Culmen Length (mm) - Plot 2")
plt.show()

sns.stripplot(data['species'], data['culmen_depth_mm'], palette = 'Reds')
plt.title("Species vs Culmen Length (mm) - Plot 2")
plt.show()

sns.FacetGrid(data, hue="species", size=8) \
   .map(plt.scatter, "culmen_length_mm", "culmen_depth_mm") \
   .add_legend()

print('Flipper Length Distribution')
sns.boxplot(data=data, x="species", y="culmen_depth_mm", hue="sex")
plt.show()

sns.distplot(data['culmen_length_mm'], color = 'blue')
plt.title('Culmen Length Distribution - Plot 1', fontsize = 20)
plt.xlabel('Culmen Length (mm)', fontsize = 16)
plt.ylabel('Count', fontsize = 16)
plt.show()

sns.distplot(data['culmen_length_mm'], bins = 58, kde = False, color = 'purple')
plt.title('Culmen Length Distribution - Plot 2', fontsize = 20)
plt.xlabel('Culmen Length (mm)', fontsize = 16)
plt.ylabel('Count', fontsize = 16)
plt.show()

x = pd.crosstab(data['island'], data['sex'])
color = plt.cm.Spectral(np.linspace(0, 1, 8))
x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, color = color)
plt.title("Sex vs Culmen Length - Plot 1", fontweight = 30, fontsize = 20)
plt.show()

data['sex'] = data['sex'].astype('category')
data['species'] = data['species'].astype('category')
data['island'] = data['island'].astype('category')

data["sex"] = data["sex"].cat.codes
data["species"] = data["species"].cat.codes
data["island"] = data["island"].cat.codes

data.head()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x=data['culmen_length_mm']
y=data['culmen_depth_mm']
z=data['flipper_length_mm']

ax.scatter( x, y, z,c=z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z,
                cmap='viridis', edgecolor='none');

print('Pairplot')
sns.pairplot(data=data[['species','culmen_length_mm','culmen_depth_mm','flipper_length_mm', 'body_mass_g']], hue="species", height=3, diag_kind="hist")
plt.show()

df = data.copy()
encode = ['species','island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'MALE':0, 'FEMALE':1}
def target_encode(val):
    return target_mapper[val]

df['sex'].unique()

X = df.drop('sex', axis=1)
y = df['sex']

# scaling the data

from sklearn import preprocessing
X = preprocessing.scale(X)

#splitting the data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=13)

# model fitting and prediction

from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X_train, y_train)
pred = model.predict(X_test)

# checking performance of model

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

print('CONFUSION MATRIX')
print(confusion_matrix(y_test, pred))

print('CLASSIFICATION REPORT\n')
print(classification_report(y_test, pred))