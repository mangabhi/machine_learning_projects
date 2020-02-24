# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:50:46 2020

@author: Wolverine
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib.inline
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('pima-data.csv')
df.shape
df.head(5)
df.info()

sns.barplot(x=df.num_preg.value_counts().index,y=df.num_preg.value_counts().values)
plt.xlabel('Pregnant')
plt.ylabel('Count')
plt.show()



sns.barplot(x=df.glucose_conc.value_counts().index,y=df.glucose_conc.value_counts().values)
plt.xlabel('Glucose')
plt.ylabel('Count')
plt.show()


sns.barplot(x=df.diastolic_bp.value_counts().index,y=df.diastolic_bp.value_counts().values)
plt.xlabel('diastolic_bp')
plt.ylabel('Count')
plt.show()


sns.barplot(x=df.thickness.value_counts().index,y=df.thickness.value_counts().values)
plt.xlabel('thickness')
plt.ylabel('Count')
plt.show()


sns.barplot(x=df.insulin.value_counts().index,y=df.insulin.value_counts().values)
plt.xlabel('insulin')
plt.ylabel('Count')
plt.show()


sns.barplot(x=df.bmi.value_counts().index,y=df.bmi.value_counts().values)
plt.xlabel('bmi')
plt.ylabel('Count')
plt.show()

sns.barplot(x=df.age.value_counts().index,y=df.age.value_counts().values)
plt.xlabel('age')
plt.ylabel('Count')
plt.show()

df.info()


sns.countplot(df.diabetes)
plt.xlabel('diabetes')
plt.ylabel('Count')
plt.show()



#correlation
corr_rel=df.corr()
corr_lop=corr_rel.index
plt.figure(figsize=(15,15))
sns.heatmap(df[corr_lop].corr(),annot=True,cmap='RdYlGn')


#target values bool to int
dib_map={True:1,False:0}
df['diabetes']=df['diabetes'].map(dib_map)
df.head()
df.info()

P_true=len(df.loc[df['diabetes']==True])
F_false=len(df.loc[df['diabetes']==False])

print("Total number of diabetes paitent: {0}({1:2.2f}%)".format(P_true,(P_true/(P_true+F_false))*100))

print("Total number of not diabetes paitent: {0}({1:2.2f}%)".format(F_false,(F_false/(P_true+F_false))*100))

#train model
#splitting in x and y

x=df.drop(['diabetes'],axis=1)
y=df.diabetes.values

#if there is missing values the imputer take care of that
from sklearn.preprocessing import Imputer
miss_val=Imputer(missing_values=0,strategy='mean',axis=0)
x_train=miss_val.fit_transform(x_train)
x_test=miss_val.fit_transform(x_test)

#building models

#svm
from sklearn.svm import SVC
svn=SVC(random_state=1)
svn.fit(x_train,y_train)
print("Accuarcy on SVM is: {:.2f}%".format(svn.score(x_test,y_test)*100))


#for naive bayes
from sklearn.naive_bayes import GaussianNB
nav=GaussianNB()
nav.fit(x_train,y_train)
print("Accuarcy on Naive Bayes is: {:.2f}%".format(nav.score(x_test,y_test)*100))


#KNN model
from sklearn.neighbors import KNeighborsClassifier
kn= KNeighborsClassifier(n_neighbors=3)
kn.fit(x_train,y_train)
print("Accuarcy on KNN is: {:.2f}%".format(kn.score(x_test,y_test)*100))

#Random Forest
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=1000,random_state=1)
rf.fit(x_train,y_train)
print("Random forest accuray is {:.2f}%".format(rf.score(x_test,y_test)*100))