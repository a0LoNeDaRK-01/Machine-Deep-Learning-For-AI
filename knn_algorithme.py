import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

#Outcome = 1 > Sick of Diabet 
#Outcome = 0 > Healthy

data = pd.read_csv("diabetes.csv")
data.head()

sick_people = data[data.Outcome == 1]
healthy_people = data[data.Outcome == 0]

plt.scatter(sick_people.Age,sick_people.Glucose,color="red",label="Diabet",alpha=0.4)
plt.scatter(healthy_people.Age,healthy_people.Glucose,color="green",label="Healthy",alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()

y = data.Outcome.values
x_ham_veri = data.drop(["Outcome"],axis=1)
x = (x_ham_veri - np.min(x_ham_veri))/(np.max(x_ham_veri)-np.min(x_ham_veri))

print("Eğitim Oncesi Veriler")
print(x_ham_veri)
print("Egitim Sonrasi Veriler")
print(x.head())

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=1)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("K=6 icin Test Verilerimizin Dogrulama Testi Sonucu: ",knn.score(x_test,y_test))

sayac=1
for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors=k)
    knn_yeni.fit(x_train,y_train)
    print(sayac," icin ","Dogruluk Orani: % ", knn_yeni.score(x_test,y_test)*100)
    sayac+=1

sc = MinMaxScaler()
sc.fit_transform(x_ham_veri)

new_prediction = []
new_prediction = knn.predict(sc.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))
new_prediction[0]