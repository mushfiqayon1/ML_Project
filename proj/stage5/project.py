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
import seaborn as sns
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor



# dataset= sy.argv[1]
dataset='PWM_Thrust_dataset.csv'
# dataset='PWM_Thrust_dataset_with_header.csv'

print (dataset)
df = pd.read_csv(dataset,header=None)
# df = pd.read_csv(dataset)
df.columns=['PWM1', 'PWM2', 'PWM3', 'PWM4', 'Thrust']
df.head()
# df= np.array(df)

# visualize the pair-wise correlations between the different features in this dataset
cols = ['PWM1', 'PWM2', 'PWM3', 'PWM4', 'Thrust']
# sns.pairplot(df[cols],kind="reg", height=2.5)
# plt.tight_layout()
# plt.savefig('pairwise_correlation.png')
# plt.show()
# cm = np.corrcoef(df[cols].values.T)
# sns.set(font_scale=1.5)
# hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
# plt.savefig('correlation_heatmap.png')
# plt.show()


#Calculate and print the number of rows and columns that this dataset contains.
print('the number of rows and coloumns are:',df.shape)


X = df[['PWM1', 'PWM2', 'PWM3', 'PWM4']]#.values
y= df[['Thrust']]#.values
print('the x data are:\n',X)
print('The y data are:\n',y)

# plt.plot(df['PWM1'],'b',label="PWM1")
# plt.plot(df['PWM2'],'g',label="PWM2")
# plt.plot(df['PWM3'],'r',label="PWM3")
# plt.plot(df['PWM4'],'c',label="PWM4")
#
# plt.title('PWM inuput commands to the Motors of the ARDrone')
# plt.legend(loc="best")
# plt.xlabel('# of samples')
# plt.ylabel('PWM Percentage')
# plt.grid()
# plt.savefig('PWM_Inputs.png')
# plt.show()
#
#
#
# plt.plot(df['Thrust'],'m',label="Thrust")
# plt.title('Thrust produced by the drone motors')
# plt.xlabel('# of samples')
# plt.ylabel('Thrust Force [N]')
# plt.legend(loc='best')
# plt.grid()
# plt.savefig('Thrust.png')
# plt.show()



# X = df.iloc[:,:-1].values
X = df[['PWM1', 'PWM2', 'PWM3', 'PWM4']]#.values
y = df['Thrust'].values
print('the x data are:\n',X)
print('The y data are:\n',y)

print('starting with Random Forest Regressor:\n')
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.333,random_state=30)
forest = RandomForestRegressor(n_estimators=10000,criterion='mse',random_state=10,n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print('MSE train: %.3f, test: %.3f\n' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f\n' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

plt.scatter(y_train_pred,y_train_pred - y_train,c='steelblue',edgecolor='white',marker='o',s=35,alpha=0.9,label='Training data')
plt.scatter(y_test_pred,y_test_pred - y_test,c='limegreen',edgecolor='white',marker='s',s=35,alpha=0.9,label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.xlim([-10, 50])
plt.show()
print('Done with Random Forest Regressor:\n')

#Decision tree regression
# X = df['PWM1'].values
# y = df['Thrust'].values

# tree = DecisionTreeRegressor(max_depth=3)
# tree.fit(X, y)
# sort_idx = X.flatten().argsort()
# lin_regplot(X[sort_idx], y[sort_idx], tree)
# plt.xlabel('% lower status of the population [LSTAT]')
# plt.ylabel('Price in $1000s [MEDV]')
# plt.show()


# X = df[:,0:df.shape[1]-1]
# y= df[:,-1]

print('the x data are:\n',X)
print('The y data are:\n',y)

# X = df[:,0]
# y= df[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(PWM1, Thrust, test_size=0.3, random_state=1)


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

