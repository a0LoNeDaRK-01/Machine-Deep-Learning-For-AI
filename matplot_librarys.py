from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.plot([1,2,3,4],[1,4,9,16])

x = [1,2,3,4]
y = [1,4,9,16]
plt.plot(x,y)
plt.show()

plt.title("My First Graphic")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.plot(x,y)
plt.show()

plt.xticks([1,2,3,4])
plt.yticks([1,4,9,16])
plt.title("My First Graphic")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.plot(x,y)
plt.show()

plt.plot(x,y,label="x^2",color="blue")
plt.xticks([1,2,3,4])
plt.yticks([1,4,9,16])
plt.title("My First Graphic")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.legend()
plt.show()

plt.plot(x,y,label="x^2",color="blue",linewidth=2,linestyle="--",marker=".")
plt.xticks([1,2,3,4])
plt.yticks([1,4,9,16])
plt.title("My First Graphic")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.legend()
plt.show()



plt.plot(x,y,label="x^2",color="blue",linewidth=2,linestyle="--",marker=".")
plt.xticks([1,2,3,4])
plt.yticks([1,4,9,16])
plt.title("My First Graphic")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.legend()
"""47:59"""
xnd = np.arange(0,5,0.5)
plt.plot(xnd,xnd*2,color="red",linewidth=2,marker=".",label="2*xnd")
plt.legend()
plt.savefig("myfirstaigraphic.png",dpi=300)
plt.show()

x=["Adana","Mersin","Antalya"]
y=[188,178,168]
plt.bar(x,y)
plt.show()

x=["Adana","Mersin","Antalya"]
y=[188,178,168]
bars = plt.bar(x,y)
bars[1].set_hatch("/")
bars[0].set_hatch("*")
plt.show()

gas = pd.read_csv("petrolfiyatlari.csv")
plt.title("Petrol Fiyatlari")
plt.plot(gas["Year"],gas["USA"],'b-',label="USA")
plt.xlabel("Yil")
plt.ylabel("Dolar")
plt.legend()
plt.show()

gas = pd.read_csv("petrolfiyatlari.csv")
plt.title("Petrol Fiyatlari")
plt.plot(gas["Year"],gas["USA"],"b-",label="USA")
plt.plot(gas["Year"],gas["France"],"g-",label="France")
plt.plot(gas["Year"],gas["Canada"],"r.-",label="Canada")
plt.plot(gas["Year"],gas["South Korea"],"y.-",label="South Korea")
plt.xlabel("Yil")
plt.ylabel("Dolar")
plt.legend()
plt.show()

plt.figure(figsize=(10,10))
plt.title("Petrol Fiyatlari")
plt.plot(gas["Year"],gas["USA"],"b-",label="USA")
plt.plot(gas["Year"],gas["France"],"g-",label="France")
plt.plot(gas["Year"],gas["Canada"],"r.-",label="Canada")
plt.plot(gas["Year"],gas["South Korea"],"y.-",label="South Korea")
plt.xlabel("Yil")
plt.ylabel("Dolar")
plt.legend()
plt.savefig("mysecondaigraphic.png",dpi=300)
plt.show()

"""Numpy Using"""
a = np.array([1,2,3])
print(a)
b = np.array([[1,2,3],[1,4,6]])
print(b)
print(a.ndim)
print(b.ndim)
print(b.shape)
print(a.dtype)
print(b[0,2])
print(b[0])
print(b[0,1:])
b[1,0] = 2

dizeros = np.zeros((3,3))
print(dizeros)

dizi_full = np.full((4,4),1)
print(dizi_full)

dizi_rand = np.random.rand(2,3)
print(dizi_rand)

dizi_randint = np.random.randint(0,44,size=(3,4))
print(dizi_randint)

#b=a
b = a.copy()
b[0] = 44
print(a)
print(b)

a = np.array([2,4,6,8])
print(a)
print(a+4)
print(a*2)
print(a/2)
b = np.array([3,6,9,12])
print(a+b)
print(a**2)

example_array = np.array([[1,2,3,4],[5,6,7,8]])
print(np.min(example_array))
print(np.max(example_array))
print(np.sum(example_array))

filedata = np.genfromtxt("example.txt",delimiter=",")
filedata = filedata.astype("int32")
print(filedata)
print(filedata[1,0])