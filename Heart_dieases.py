
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('heart.csv')
df.head()
df.info()
df.describe()

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,fmt='.1f')
plt.show()
df.age.value_counts()[:10]

sns.barplot(x=df.age.value_counts()[:10].index,y=df.age.value_counts()[:10].values)
plt.xlabel('Age')
plt.ylabel('Age_count')
plt.title("Age Estimation")
plt.show()

df.target.value_counts()
countNoDiseas=len(df[df.target==0])
countHaveDiseas=len(df[df.target==1])
print("Paients having no heart diease:{:.2f}%".format((countNoDiseas/(len(df.target)))*100))
print("Paients having have heart diease:{:.2f}%".format((countHaveDiseas/(len(df.target)))*100))


df.sex.value_counts()
countMale=len(df[df.sex==1])
countFemale=len(df[df.sex==0])
print("% of male Patients:{:.2f}%".format((countMale/(len(df.sex))*100)))

print("% of female Patients:{:.2f}%".format((countFemale/(len(df.sex))*100)))


young_age=df[(df.age>29)&(df.age<40)]
middle_age=df[(df.age>40)&(df.age<55)]
old_age=df[(df.age>55)]
print("Young age",len(young_age))
print("Middle age",len(middle_age))
print("Old age",len(old_age))

colors=['red','green','purple']
explode=[0.1,0.1,0.1]
plt.figure(figsize=(5,5))
plt.pie([len(young_age),len(middle_age),len(old_age)],labels=['Young age','Middle age','Old age'])
plt.show()

#chest pain analysis
df.cp.value_counts()
df.target.unique()
sns.countplot(df.target)
plt.xlabel("Target")
plt.ylabel('count')
plt.title('Target 1 & 0')
plt.show()

df.corr()

from sklearn.linear_model import LogisticRegression
x_data=df.drop(['target'],axis=1)
y=df.target.values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_data,y,test_size=0.2,random_state=0)

logic_reg=LogisticRegression()
logic_reg.fit(x_train,y_train)
print("Test accuarcy: {:.2f}%".format(logic_reg.score(x_test,y_test)*100))


#for Knn model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
print("Test accuracy of knn is {:.2f}%".format(knn.score(x_test,y_test)*100))


'''#for Svm model   not working
from sklearn.svm import SVC
sps=SVC(random_state=1,kernel='rbf')
sps.fit_transpose(x_train,y_test)
print("SVM Accuracy report {:.2f}%".format(sps.score(x_test,y_test)*100))
'''
#naive bayes
from sklearn.naive_bayes import GaussianNB
nai=GaussianNB()
nai.fit(x_train,y_train)
print("Naive Bayes Accuracy report {:.2f}%".format(nai.score(x_test,y_test)*100))

#Random forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=1000,random_state=1)
rf.fit(x_train,y_train)
print("Random Forest Accuracy report {:.2f}%".format(rf.score(x_test,y_test)*100))
