from asyncore import dispatcher_with_send
from atexit import register
from cProfile import label
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("polynomial.csv",sep=";")

plt.scatter(df["deneyim"],df["maas"])
plt.xlabel("Deneyim(Yil)")
plt.ylabel("Maas")
plt.savefig("1.png", dpi = 300)
plt.show()

reg = LinearRegression()
reg.fit(df[["deneyim"]],df["maas"])
plt.xlabel("Deneyim(Yil)")
plt.ylabel("Maas")
plt.scatter(df["deneyim"],df["maas"])

x_ekseni = df["deneyim"]
y_ekseni = reg.predict(df[["deneyim"]])
plt.plot(x_ekseni,y_ekseni,color="green",label="linear_regression")
plt.legend()
plt.show()

polynomial_regression = PolynomialFeatures(degree = 4)
x_polinomial = polynomial_regression.fit_transform(df[["deneyim"]])

reg = LinearRegression()
reg.fit(x_polinomial,df["maas"])

y_head = reg.predict(x_polinomial)
plt.plot(df["deneyim"],y_head,color="red",label="Polynomial Regression")
plt.legend()

plt.scatter(df["deneyim"],df["maas"])
plt.show()

x_polinomial1 = polynomial_regression.fit_transform([[7.5]])
print(reg.predict(x_polinomial1))