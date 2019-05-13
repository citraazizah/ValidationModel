#%%stage
import pandas as pd  
import numpy as np 
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

    X = data[:, :4]
    y = data[:, 4]

    error = 0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
       
        if (count_error(X_train, X_test, y_train, y_test)):
            error += 1

    print('Error with LOO : ', (error / len(data)) * 100, '%')

def kfold_errorratio(data):
    kfold = KFold(n_splits = 10)
    
    X = data[:, :4]
    y = data[:, 4]

    error = 0
    total_error = 0
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for i, test_data in enumerate(X_test):
          if (count_error(X_train, [test_data.tolist()], y_train, y_test[i])):
            error += 1

        total_error += (error / len(X_test)) * 100
        error = 0

    print('Error with KFOLD : ', (total_error / kfold.get_n_splits()), '%')

iris_data = np.array(pd.read_csv('iris.csv'))

knn = KNeighborsClassifier(n_neighbors=3)
kfold_errorratio(iris_data)
loo_errorratio(iris_data)