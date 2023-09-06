import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv("DecisionTreesClassificationDataSet.csv")
print(df.head())

duzeltme_mapping = {"Y":1,"N":0}

df["IseAlindi"] = df["IseAlindi"].map(duzeltme_mapping)
df["StajBizdeYaptimi?"] = df["StajBizdeYaptimi?"].map(duzeltme_mapping)
df["Top10 Universite?"] = df["Top10 Universite?"].map(duzeltme_mapping)
df["SuanCalisiyor?"] = df["SuanCalisiyor?"].map(duzeltme_mapping)

egitim_mapping = {"BS":0,"MS":1,"PhD":2}
df["Egitim Seviyesi"] = df["Egitim Seviyesi"].map(egitim_mapping)
print(df.head())

y = df["IseAlindi"]
x = df.drop(["IseAlindi"],axis=1)
print(x.head())

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)

##Artık Tahmin Yapabiliriz :)
print(clf.predict([[2,1,2,0,0,0]]))
print(clf.predict([[1,1,1,0,0,0]]))