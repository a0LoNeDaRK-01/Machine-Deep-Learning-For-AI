import pandas as pd


df = pd.read_csv("outlier_ornek_veriseti.csv",sep=';')
print(df)

df.describe()

Q1 = df.boy.quantile(0.25)
print(Q1)

Q3 = df.boy.quantile(0.75)
print(Q3)

Q2 = df.boy.quantile(0.50)
print(Q2)

IQR_Degeri = Q3-Q1
print(IQR_Degeri)

alt_limit = Q1 -1.5*IQR_Degeri
print(alt_limit)

ust_limit = Q3 + 1.5*IQR_Degeri
print(ust_limit)

print(df[(df.boy<alt_limit) | (df.boy>ust_limit)])

### Altta "|" isaretini koymamamizin sebebi ise butun datayi almasini istemis oluruz.Ancak istedigimizse filtreleme yapmaktı.
df_outlier_filter = df[(df.boy<ust_limit) & (df.boy>alt_limit)]
print(df_outlier_filter)