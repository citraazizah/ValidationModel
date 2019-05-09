#%%stage
import pandas as pd  
import numpy as np 
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold


def count_error(X_train, X_test, y_train, y_test):
    knn.fit(X_train, y_train)
    prediksi = knn.predict(X_test)
    if prediksi != y_test:
        return True

    return False

def loo_errorratio(data):

    loo = LeaveOneOut()

    X = np.array(data)[:,:4]
    Y = np.array(data)[:,4]

    error = 0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
       
        if (count_error(X_train, X_test, y_train, y_test)):
            error += 1

    print('Error with LOO : ', (error / len(data)) * 100, '%')

def kfold_errorratio(data):
    kfold = KFold(n_split = 10)
    
    X = np.array(data)[:,:4]
    Y = np.array(data)[:,4]

    error = 0
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
       
        if (count_error(X_train, X_test, y_train, y_test)):
            error += 1

    print('Error with KFOLD : ', (error / len(data)) * 100, '%')

    

iris_data = pd.read_csv('iris.csv').tolist()
print(iris_data)
knn = KNeighborsClassifier(n_neighbors=3)
loo_errorratio(iris_data)
kfold_errorratio(iris_data)

