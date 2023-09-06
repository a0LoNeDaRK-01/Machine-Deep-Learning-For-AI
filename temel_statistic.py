from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


gelirler = np.random.normal(10,100,50)
print(gelirler)

print(np.mean(gelirler))
plt.hist(gelirler,10)
plt.show()

gelirler = np.append(gelirler,78)
print("Ortalama: " , np.mean(gelirler))
print("Medyan: " , np.median(gelirler))

ages = np.random.randint(7,15,size=300)


mode_result = stats.mode(ages,keepdims=True)
print("Mode (Mod):" , mode_result.mode[0])
print("Medyan: " , np.median(ages))
print("Ortalama: " , np.mean(ages))
print("Varyans: " , np.var((gelirler)))
print("Standart Sapma: " , np.std((gelirler)))
