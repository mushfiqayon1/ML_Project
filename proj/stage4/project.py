import numpy as np
import pandas as pd
import sys as sy
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


dataset= sy.argv[1]
    
print (dataset)
df = pd.read_csv(dataset)
df= np.array(df)
X = df[:,0:df.shape[1]-1]
y= df[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


krr = KernelRidge(alpha=1.0)
krr.fit(X_train, y_train)

print ("Training accuracy for KRR")
print (krr.score(X_train,y_train))

y_train_pred= krr.predict(X_train)
y_test_pred= krr.predict(X_test)


r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
error_train= mean_squared_error(y_train, y_train_pred)
error_test= mean_squared_error(y_test, y_test_pred) 
print('MSE train: %.3f, test:%.3f' % (error_train, error_test)) 
print('R^2 value train: %.3f, test:%.3f' % (r2_train, r2_test))


forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=10)
forest.fit(X_train, y_train)

print ("Training accuracy for RFR")
print (forest.score(X_train,y_train))

y_train_pred = forest.predict(X_train)

y_test_pred = forest.predict(X_test)


r2_train = r2_score(y_train, y_train_pred) 
r2_test = r2_score(y_test, y_test_pred)
error_train= mean_squared_error(y_train, y_train_pred)
error_test= mean_squared_error(y_test, y_test_pred) 
print('MSE train: %.3f, test:%.3f' % (error_train, error_test)) 
print('R^2 value train: %.3f, test:%.3f' % (r2_train, r2_test))


sv = SVR(C=1.0, epsilon=0.2)
sv.fit(X_train, y_train)
print ("Training accuracy for SVR")
print (sv.score(X_train,y_train))
y_train_pred = sv.predict(X_train)

y_test_pred = sv.predict(X_test)


r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
error_train= mean_squared_error(y_train, y_train_pred)
error_test= mean_squared_error(y_test, y_test_pred) 
print('MSE train: %.3f, test:%.3f' % (error_train, error_test)) 
print('R^2 value train: %.3f, test:%.3f' % (r2_train, r2_test))

