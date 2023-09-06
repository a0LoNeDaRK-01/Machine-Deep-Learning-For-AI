import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "pca_iris.data"
df = pd.read_csv(url,names=["sepal length","sepal width","petal length","petal width","target"])
print(df)

features = ["sepal length","sepal width","petal length","petal width"]
x = df[features]
y = df[["target"]]

x = StandardScaler().fit_transform(x)
print(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents,columns=["Principal Components 1","Principal Components 2"])
print(principalDf)

final_df = pd.concat([principalDf,df[["target"]]],axis=1)
print(final_df.head())

dfSetosa = final_df[df.target == "Iris-setosa"]
dfVirginica = final_df[df.target == "Iris-virginica"]
dfVersicolor = final_df[df.target == "Iris-versicolor"]
plt.xlabel("Principal Components 1")
plt.ylabel("Principal Components 2")
plt.scatter(dfSetosa["Principal Components 1"],dfSetosa["Principal Components 2"],color="green")
plt.scatter(dfVirginica["Principal Components 1"],dfVirginica["Principal Components 2"],color="red")
plt.scatter(dfVersicolor["Principal Components 1"],dfVersicolor["Principal Components 2"],color="blue")
plt.show()

targets = ["Iris-setosa","Iris-virginica","Iris-versicolor"]
colors = ["g","r","b"]
plt.xlabel("Principal Components 1")
plt.ylabel("Principal Components 2")
for target,col in zip(targets,colors):
    dftemp = final_df[df.target==target]
    plt.scatter(dftemp["Principal Components 1"],dftemp["Principal Components 2"],color=col)
plt.show()

pca.explained_variance_ratio_
pca.explained_variance_ratio_.sum()