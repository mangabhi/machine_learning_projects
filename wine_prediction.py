import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df=pd.read_csv('wine.csv')
df.info()


df['grade'] = 1 # good
df.grade[df.quality < 7] = 0 # not good

plt.figure(figsize = (8,8))
labels = df.grade.value_counts().index
plt.pie(df.grade.value_counts(), autopct='%1.1f%%')
plt.legend(labels, loc="best")
plt.axis('equal')
plt.title('Quality Pie Chart')
plt.show()
print('The good quality wines count for ',round(df.grade.value_counts(normalize=True)[1]*100,1),'%.')

sns.pairplot(df, hue='grade')
plt.show()


mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (12,12))
sns.heatmap(df.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20)
plt.show()

good = df[df.grade == 1]
notgood = df[df.grade == 0]

drop_items = ['quality','grade']
g1 = pd.DataFrame(good.drop(drop_items, axis=1).mean(), columns=['Good']).T
g2 = pd.DataFrame(notgood.drop(drop_items, axis=1).mean(), columns=['Not Good']).T
total = pd.DataFrame(df.drop(drop_items, axis=1).mean(), columns=['Total Average']).T
data = g1.append([g2, total])

# Set standard
temp1 = data.values.reshape((3, 11))
standard = data.loc['Total Average'].values.reshape((1, 11))
temp = 100* temp1 / standard
data_percentage = pd.DataFrame(temp, columns = data.columns.values.tolist())


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost
from sklearn.metrics import accuracy_score

df_train_features = df.drop(['quality','grade'], axis =1)
n = 11

x_train, x_test, y_train, y_test = train_test_split(df_train_features, df['grade'], test_size=0.1, random_state=7)

x_train_mat = x_train.values.reshape((len(x_train), n))
x_test_mat = x_test.values.reshape((len(x_test), n))


print('Start Predicting...')

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train_mat,y_train)
tree_pred = decision_tree.predict(x_test_mat)

rf = RandomForestClassifier()
rf.fit(x_train_mat,y_train)
rf_pred = rf.predict(x_test_mat)

KN = KNeighborsClassifier()
KN.fit(x_train_mat,y_train)
KN_pred = KN.predict(x_test_mat)

Gaussian = GaussianNB()
Gaussian.fit(x_train_mat,y_train)
Gaussian_pred = Gaussian.predict(x_test_mat)

svc = SVC()
svc.fit(x_train_mat,y_train)
svc_pred = svc.predict(x_test_mat)

xgb = xgboost.XGBClassifier()
xgb.fit(x_train_mat,y_train)
xgb_pred = xgb.predict(x_test_mat)

print('...Complete')



print('Decision Tree:', accuracy_score(y_test, tree_pred)*100,'%')
print('Random Forest:', accuracy_score(y_test, rf_pred)*100,'%')
print('KNeighbors:',accuracy_score(y_test, KN_pred)*100,'%')
print('GaussianNB:',accuracy_score(y_test, Gaussian_pred)*100,'%')
print('SVC:',accuracy_score(y_test, svc_pred)*100,'%')
print('XGB:',accuracy_score(y_test, xgb_pred)*100,'%')

k = [10,20,30,40,50]
for i in k:
    rf_tune = RandomForestClassifier(n_estimators=50, random_state=i)
    rf_tune.fit(x_train_mat,y_train)
    y_pred = rf_tune.predict(x_test_mat)
    print(accuracy_score(y_test, y_pred)*100,'%')


x_train_check = df_train_features.values.reshape((len(df_train_features), n))
x_test_check = df['grade'].values.reshape((len(df['grade']), 1))

k = [10,20,30,40,50]
for i in k:
    rf_tune = RandomForestClassifier(n_estimators=50, random_state=i)
    rf_tune.fit(x_train_mat,y_train)
    yy_pred = rf_tune.predict(x_train_check)
    print(accuracy_score(x_test_check, yy_pred)*100,'%')    
    
 
plt.figure(figsize = (20,8))
domain = np.linspace(1,100,len(y_pred)) 
plt.plot(domain, rf_pred,'o')
plt.plot(domain, y_test,'o')
plt.legend(('Prediction','Actual value'))
plt.show()    
    
