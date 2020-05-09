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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor



dataset= sy.argv[1]
# dataset='PWM_Thrust_dataset.csv'
# dataset='PWM_Thrust_dataset_with_header.csv'

print (dataset)
df = pd.read_csv(dataset,header=None)
# df = pd.read_csv(dataset)
df.columns=['PWM1', 'PWM2', 'PWM3', 'PWM4', 'Thrust']
df.head()
# df= np.array(df)

# visualize the pair-wise correlations between the different features in this dataset
def viz_correlation_heatmap():
    cols = ['PWM1', 'PWM2', 'PWM3', 'PWM4', 'Thrust']
    sns.pairplot(df[cols],kind="reg", height=2.5)
    plt.tight_layout()
    plt.savefig('pairwise_correlation.png')
    plt.show()
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
    plt.savefig('correlation_heatmap.png')
    plt.show()
    return


#Calculate and print the number of rows and columns that this dataset contains.
print('the number of rows and coloumns are:',df.shape)


X = df[['PWM1', 'PWM2', 'PWM3', 'PWM4']]#.values
y= df[['Thrust']]#.values
print('the x data are:\n',X)
print('The y data are:\n',y)

def viz_plotting_dataset():
    plt.plot(df['PWM1'],'b',label="PWM1")
    plt.plot(df['PWM2'],'g',label="PWM2")
    plt.plot(df['PWM3'],'r',label="PWM3")
    plt.plot(df['PWM4'],'c',label="PWM4")

    plt.title('PWM inuput commands to the Motors of the ARDrone')
    plt.legend(loc="best")
    plt.xlabel('# of samples')
    plt.ylabel('PWM Percentage')
    plt.grid()
    plt.savefig('PWM_Inputs.png')
    plt.show()
    plt.plot(df['Thrust'],'m',label="Thrust")
    plt.title('Thrust produced by the drone motors')
    plt.xlabel('# of samples')
    plt.ylabel('Thrust Force [N]')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('Thrust.png')
    plt.show()
    return


# X = df.iloc[:,:-1].values
# X = df[['PWM1', 'PWM2', 'PWM3', 'PWM4']].values
X = df[['PWM1']].values
y = df['Thrust'].values
print('the x data are:\n',X)
print('The y data are:\n',y)

##############################################################
# Splitting the dataset into the Training set and Test set
##############################################################
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.3,random_state=1)



##############################################################
# Fitting Linear Regression to the dataset
##############################################################
def viz_linear_regressor():
    print('starting with linear Regressor:\n')
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_train_pred = lin_reg.predict(X_train)
    y_test_pred = lin_reg.predict(X_test)
    print('MSE train: %.3f, test: %.3f\n' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f\n' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
    plt.plot(y_test,'b',label='testing data')
    plt.plot(y_test_pred,'r',label='predicted data')
    plt.legend(loc='best')
    plt.title('PWM-Thrust (Linear Regression)')
    plt.xlabel('# of samples')
    plt.ylabel('Thrust Force[N]')
    plt.savefig('linear_regression.png')
    plt.show()
    print('Done with linear Regressor:\n')
    return


##############################################################
# Fitting Polynomial Regression to the dataset
##############################################################
def viz_polynomial_regression():
    print('starting with Polynomial Regression (deg=4):\n')
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    y_train_pred=pol_reg.predict(poly_reg.fit_transform(X_train))
    y_test_pred=pol_reg.predict(poly_reg.fit_transform(X_test))
    print('MSE train: %.3f, test: %.3f\n' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f\n' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
    plt.plot(y_test,'b',label='testing data')
    plt.plot(y_test_pred,'r',label='predicted data')
    plt.legend(loc='best')
    plt.title('PWM-Thrust (Linear Regression)')
    plt.xlabel('# of samples')
    plt.ylabel('Thrust Force[N]')
    plt.savefig('polynomial_regression_deg4.png')
    plt.show()
    print('Done with Polynomial Regression:\n')
    return

def viz_polynomial_regression6():
    print('starting with Polynomial Regression (deg=6):\n')
    poly_reg = PolynomialFeatures(degree=10)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    y_train_pred=pol_reg.predict(poly_reg.fit_transform(X_train))
    y_test_pred=pol_reg.predict(poly_reg.fit_transform(X_test))
    print('MSE train: %.3f, test: %.3f\n' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f\n' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
    plt.plot(y_test,'b',label='testing data')
    plt.plot(y_test_pred,'r',label='predicted data')
    plt.legend(loc='best')
    plt.title('PWM-Thrust (Linear Regression)')
    plt.xlabel('# of samples')
    plt.ylabel('Thrust Force[N]')
    plt.savefig('polynomial_regression_deg6.png')
    plt.show()
    print('Done with Polynomial Regression:\n')
    return

def viz_Random_Forest():
    print('starting with Random Forest Regressor:\n')
    # forest = RandomForestRegressor(n_estimators=1000,criterion='mse',random_state=10,n_jobs=-1)
    forest = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
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
    plt.hlines(y=0, xmin=3, xmax=5, lw=2, color='black')
    plt.xlim([3.1, 4.2])
    plt.grid()
    plt.savefig('residuals_random_forest.png')
    plt.show()
    # Plot the results:
    s = 50
    a = 0.4
    plt.plot(y_test,'b',label='testing data')
    plt.plot(y_test_pred,'r',label='predicted data')
    plt.legend(loc='best')
    plt.grid()
    plt.title('Results plot of the Random forest')
    plt.xlabel('# of samples')
    plt.ylabel('Thrust Force [N]')
    plt.savefig('output_results_random_forst.png')
    plt.show()
    print('Done with Random Forest Regressor:\n')
    return

def viz_Kernal_Ridge():
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
    plt.plot(y_test,'b',label='testing data')
    plt.plot(y_test_pred,'r',label='predicted data')
    plt.legend(loc='best')
    plt.title('PWM-Thrust (Kernel Ridge)')
    plt.xlabel('# of samples')
    plt.ylabel('Thrust Force[N]')
    plt.savefig('Kernel_ridge.png')
    plt.show()
    print('Done with Kernel Ridge:\n')
    return
#################################
##### All functions are here
#################################
viz_plotting_dataset()
viz_correlation_heatmap()
viz_linear_regressor()
viz_polynomial_regression()
viz_polynomial_regression6()
viz_Kernal_Ridge()
viz_Random_Forest()



