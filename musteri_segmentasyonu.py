import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os

os.environ["OMP_NUM_THREADS"] = "1"

df = pd.read_csv("Avm_Musterileri.csv")
print(df.head())

plt.scatter(df["Annual Income (k$)"],df["Spending Score (1-100)"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

df.rename(columns = {"Annual Income (k$)":"income"},inplace = True)
df.rename(columns = {"Spending Score (1-100)":"score"},inplace = True)

scaler = MinMaxScaler()
scaler.fit(df[["income"]])
df["income"] = scaler.transform(df[["income"]])

scaler.fit(df[["score"]])
df["score"] = scaler.transform(df[["score"]])
print(df.head())
print(df.tail())

k_range = range(1,11)

list_dist = []

for k in k_range:
    kmeans_modelim = KMeans(n_clusters=k)
    kmeans_modelim.fit(df[["income","score"]])
    list_dist.append(kmeans_modelim.inertia_)

plt.xlabel("K")
plt.ylabel("Distortion Degeri (inertia)")
plt.plot(k_range,list_dist)
plt.show()

###"Yukaridaki algoritmamizin ciktisindan aldigimiz sonuca gore "Diz" kuralindan 5'i en yakin rakam olarak aldik.
kmeans_modelim = KMeans(n_clusters=5)
y_predicted = kmeans_modelim.fit_predict(df[["income","score"]])
print(y_predicted)

df["cluster"] = y_predicted
print(df.head())

kmeans_modelim.cluster_centers_

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
df4=df[df.cluster==3]
df5=df[df.cluster==4]

plt.xlabel("income")
plt.ylabel("score")
plt.scatter(df1["income"],df1["score"],color="green")
plt.scatter(df2["income"],df2["score"],color="red")
plt.scatter(df3["income"],df3["score"],color="black")
plt.scatter(df4["income"],df4["score"],color="orange")
plt.scatter(df5["income"],df5["score"],color="purple")
plt.scatter(kmeans_modelim.cluster_centers_[:,0],kmeans_modelim.cluster_centers_[:,1],color = "blue",marker="X",label="Centroid")
plt.legend()
plt.show()