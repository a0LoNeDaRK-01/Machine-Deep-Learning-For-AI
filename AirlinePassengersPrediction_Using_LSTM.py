import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.models import Sequential

data = pd.read_csv("AirPassengers.csv")
print(data.head())

data.rename(columns={"#Passengers":"passengers"},inplace=True)
data = data["passengers"]
print(type(data))
print(data)

data = np.array(data).reshape(1,-1)
print(type(data))

plt.plot(data)
plt.show()

scaler = MinMaxScaler()
scalizing = scaler.fit_transform(data)
print(len(data))

train = data[0:100,:]
test = data[100:,:]

def get_data(data,steps):
    datax = []
    datay = []
    for i in range(len(data)-steps-1):
        a = data[i:(i+steps),0]
        datax.append(a)
        datay.append(data[i+steps,0])
    return np.array(datax),np.array(datay)

steps = 2

x_train,y_train = get_data(train,steps)
x_test,y_test = get_data(test,steps)

x_train = np.reshape(x_train,(x_train.shape[0],1,x_train.shape[1]))
x_test = np.reshape(x_test,(x_test.shape[0],1,x_test.shape[1]))

model = Sequential()
model.add(LSTM(128,input_shape=(1,steps)))
model.add(Dense(64))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer = "adam")
model.summary()

model.fit(x_train,y_train,epochs=25,batch_size=1)

y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = y_test.reshape(1,-1)
y_test = scaler.inverse_transform(y_test)

plt.plot(y_test,label = "real number of passengers")
plt.plot(y_pred,label = "predicted number of passengers")
plt.xlabel("Months")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()