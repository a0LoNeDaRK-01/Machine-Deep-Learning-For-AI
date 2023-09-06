import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("multilinearregression.csv",sep=";")
df

features = df[['alan', 'odasayisi', 'binayasi']]
features.columns = ['alan','odasayisi','binayasi']
target = df['fiyat']

reg = linear_model.LinearRegression()
reg.fit(features,target)
reg.predict([[230,4,10]])
reg.predict([[230,6,0]])
reg.predict([[355,3,20]])
reg.predict([[230,4,10], [230,6,0], [355,3,20]])

reg.coef_
reg.intercept_
a = reg.intercept_
b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 230
x2 = 4
x3 = 10

y = a + b1*x1 + b2*x2 + b3*x3
print("Tahmin Edilen Y degeri: ",y)