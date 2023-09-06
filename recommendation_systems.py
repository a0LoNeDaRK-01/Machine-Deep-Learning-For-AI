import numpy as np
import pandas as pd

column_names = ["user_id","item_id","rating","timestamp"]
df = pd.read_csv("users.data",sep="\t",names=column_names)
print(df.head())
print(len(df))

movie_titles = pd.read_csv("movie_id_titles.csv")
print(movie_titles.head())
print(len(movie_titles))

df = pd.merge(df,movie_titles,on="item_id")
print(df.head())

movieMat = df.pivot_table(index="user_id",columns="title",values="rating")
print(movieMat.head())
print(type(movieMat))

toystory_ratings = movieMat["Toy Story (1995)"]
print(toystory_ratings.head())

similar_to_toystory = movieMat.corrwith(toystory_ratings)
print(similar_to_toystory)
type(similar_to_toystory)

corr_toystory = pd.DataFrame(similar_to_toystory,columns=["Correlation"])
corr_toystory.dropna(inplace=True)
print(corr_toystory.head())

corr_toystory.sort_values("Correlation",ascending=False).head(10)
print(df.head())

df.drop(["timestamp"],axis=1)

ratings = pd.DataFrame(df.groupby("title")["rating"].mean())
ratings.sort_values("rating",ascending=False).head()

ratings["rating_oy_sayisi"] = pd.DataFrame(df.groupby("title")["rating"].count())
print(ratings.head())

ratings.sort_values("rating_oy_sayisi",ascending=False).head()
corr_toystory.sort_values("Correlation",ascending=False).head(10)

corr_toystory = corr_toystory.join(ratings["rating_oy_sayisi"])
print(corr_toystory.head())

print(corr_toystory[corr_toystory["rating_oy_sayisi"]>100].sort_values("Correlation",ascending=False).head())