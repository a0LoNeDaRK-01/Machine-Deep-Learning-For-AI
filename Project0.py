#Rakam Tanima!

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml #mnist datasini yuklemek icin gerekli
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


mnist = fetch_openml("mnist_784")
print(mnist.data.shape)

def showimage(df,index):
    some_digit = df.to_numpy()[index]
    some_digit_image = some_digit.reshape(28,28)
    plt.imshow(some_digit_image,cmap="binary")
    plt.axis("off")
    plt.show()

showimage(mnist.data,4)

train_img,test_img,train_lbl,test_lbl = train_test_split(mnist.data,mnist.target,test_size=1/7.0,random_state=0)
type(train_img)

test_img_copy = test_img.copy()
showimage(test_img_copy,4)

scaler = StandardScaler()
scaler.fit(train_img)

train_img = scaler.transform(train_img)
test_img = scaler.transform(train_img)

pca = PCA(0.95)

pca.fit(train_img)
print(pca.n_components_)

train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)
logisticRegr = LogisticRegression(solver="lbfgs",max_iter=10000)

logisticRegr.fit(train_img,train_lbl)

logisticRegr.predict(test_img[0].reshape(1,-1))

showimage(test_img_copy,4)
logisticRegr.predict(test_img[4].reshape(1,-1))

showimage(test_img_copy,0)
logisticRegr.predict(test_img[10000].reshape(1,-1))

showimage(test_img_copy,0)
logisticRegr.predict(test_img[10000].reshape(1,-1))

logisticRegr.score(test_img,test_lbl)