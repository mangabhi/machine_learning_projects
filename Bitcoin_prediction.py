
#Bitcoin Prediction 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('BitcoinPrice.csv')
df.info()
df.describe()
df.head(5)
df.tail(5)
df.drop(['Date'],1,inplace=True)
#prediction of 30 days data
p_days=30
df['Prediction']=df[['Price']].shift(-p_days)

#Separate x and y
X=np.array(df.drop(['Prediction'],1))
X=X[:len(df)-p_days]
X.shape

y=np.array(df['Prediction'])
y=y[:-p_days]
y.shape

#train datasets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

P_days_array=np.array(df.drop(['Prediction'],1))[-p_days:]


#Random forest model
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=1000,random_state=4)
rf.fit(X_train,y_train)
print("Random forest model accuarcy for bitcoin prediction is {:.2f}%".format(rf.score(X_test,y_test)*100))
rf_predict=rf.predict(X_test)
print(rf_predict)

rd_predict_30=rf.predict(P_days_array)
