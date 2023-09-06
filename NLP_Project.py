#Duygu Analizi Prjct.0

import nltk
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

df = pd.read_csv("NLPlabeledData.tsv",delimiter="\t",quoting=3)
print(df.head())
print(len(df))
print(len(df["review"]))
nltk.download("stopwords")

sample_review = df.review[0]
print(sample_review)
sample_review = BeautifulSoup(sample_review).get_text()
print(sample_review)
sample_review = re.sub("[^a-zA-Z]","",sample_review)

sample_review = sample_review.lower
print(sample_review)

sample_review = sample_review.split()
print(sample_review)
print(len(sample_review))

stpwords = set(stopwords.words("english"))
sample_review = [w for w in sample_review if w not in stpwords]
print(sample_review)
print(len(sample_review))

def process(review):
    
    review = BeautifulSoup(review).get_text()
    review = re.sub("[^a-zA-Z]","",review)
    review = review.lower()
    review = review.split()
    stpwords = set(stpwords.words("english"))
    review = [w for w in review if w not in review]
    return("".join(review))

train_x_tum = []
for r in range(len(df["review"])):
    if(r+1)%1000 == 0:
        print("No of reviews processed: ",r+1)
    train_x_tum.append(process(df["review"][r]))
x = train_x_tum
y = np.array(df["sentiment"])

train_x,test_x,y_train,y_test = train_test_split(x,y,test_size=0.1)

vectorizer = CountVectorizer( max_features = 5000)
train_x = vectorizer.fit_transform(train_x)
print(train_x)

train_x = train_x.toarray()
train_y = y_train

print(train_x.shape , train_y.shape)
print(train_y)

model = RandomForestClassifier(n_estimators=100 , random_state=42)
model.fit(train_x,train_y)

test_xx = vectorizer.transform(test_x)
print(test_xx)
test_xx.toarray()
print(test_xx.shape)

test_predict = model.predict(test_xx)
dogruluk = roc_auc_score(y_test,test_predict)
print("Dogruluk Orani: ",dogruluk*100)