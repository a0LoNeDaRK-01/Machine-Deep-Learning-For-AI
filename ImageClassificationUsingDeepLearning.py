import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()

x_train.shape
x_test.shape
y_train[:3]

y_test = y_test.reshape(-1,)
y_test

resim_siniflari = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def plot_sample(x,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(x[index])
    plt.xlabel(resim_siniflari[y[index]])
    plt.show()

plot_sample(x_test,y_test,0)
plot_sample(x_test,y_test,1)
plot_sample(x_test,y_test,3)

x_train = x_train/255
x_test = x_test/255

deep_learning_model = models.Sequential([
    layers.Conv2D(filters =32,kernel_size=(3,3),activation="relu",input_shape=(32,32,3)),
    layers.Maxpooling2D((2,2)),

    layers.Conv2D(filters =64,kernel_size=(3,3),activation="relu"),
    layers.Maxpooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64,activation="relu"),
    layers.Dense(10,activation="softmax")
])

deep_learning_model.compile(optimizer="adam",
                            loss = "sparse_categorical_crossentropy",
                            metrics = ["accuracy"])

deep_learning_model.fit(x_train,y_train,epochs=5)
deep_learning_model.evaluate(x_test,y_test)
y_pred = deep_learning_model.predict(x_test)
y_pred[:3]

y_pred_siniflari = [np.argmax(element) for element in y_pred]
y_pred_siniflari[:3]
y_test[:3]

plot_sample(x_test,y_test,0)
resim_siniflari = y_pred_siniflari[0]
plot_sample(x_test,y_test,1)
resim_siniflari = y_pred_siniflari[1]
plot_sample(x_test,y_test,2)
resim_siniflari = y_pred_siniflari[2]