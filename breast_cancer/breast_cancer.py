# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:51:10 2020

@author: Wolverine
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('data.csv')
df.info()
df.isna().sum()
df=df.dropna(axis=1)

#count of malignment and benaginant
df['diagnosis'].value_counts() 

sns.countplot(df['diagnosis'],label="count")

#categorical data
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df.iloc[:,1]=le.fit_transform(df.iloc[:,1].values)
df.head()
df.corr()
#heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),fmt='.0%',annot=True)


#split datasets
from sklearn.model_selection import train_test_split
X=df.drop(['diagnosis'],axis=1)
Y=df.diagnosis.values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Model looking best fit for datasets
#SVM
from sklearn.svm import SVC
sv=SVC(random_state=1)
sv.fit(X_train,Y_train)
print("SVC Accuracy : {:.2f}%".format(sv.score(X_test,Y_test)*100))


#naive Bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,Y_train)
print("Naive Bayes Accuracy : {:.2f}%".format(nb.score(X_test,Y_test)*100))


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=3000,random_state=32)  #tune your data
rf.fit(X_train,Y_train)
print("Random forest  Accuracy : {:.2f}%".format(rf.score(X_test,Y_test)*100))
