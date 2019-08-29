#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:46:22 2019

@author: subhash
"""

import numpy as np
import pandas as pd

from csv import reader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score


dataset = pd.read_csv('dataset.csv')
labelencoder = LabelEncoder()
data = dataset.iloc[:,:].values
data[:,12] = labelencoder.fit_transform(data[:,12])

#did not consider dummy variable trap here

data = np.array(data).astype('float')

X_data = data[:,0:11]
y_data = data[:,12].astype('int')


def SVM(X_train,X_test,y_train,y_test):
    c_range = np.outer(np.logspace(-1,1,3),np.array([1,5]))
    c_range = c_range.flatten()
    gamma_range = np.outer(np.logspace(-3,0,4),np.array([1,5]))
    gamma_range = gamma_range.flatten()
    parameters = {'kernel' :['rbf'],'C':c_range, 'gamma': gamma_range}
    classifier =svm.SVC()
    grid_classifier = GridSearchCV(estimator=classifier, param_grid=parameters, n_jobs=1, verbose=2)
    grid_classifier.fit(X_train,y_train)
    classifier = grid_classifier.best_estimator_
    
    train_predictions = classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train,train_predictions)
    
    test_predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test,test_predictions)
    
    precision = precision_score(y_test,test_predictions,average='weighted')
    recall = recall_score(y_test,test_predictions,average='weighted')
    f1 = 2.0 * (precision*recall)/(precision+recall)
    
    print("Precision : %.4f" % (precision))
    print("Recall : %.4f" % (recall))
    print("F1 Score : %.4f" % (f1))
    
    return accuracy,precision,recall,f1


sss = StratifiedShuffleSplit(n_splits=5,test_size=0.125)
scores = []

for train_index,test_index in sss.split(X_data,y_data):
    X_train,X_test = X_data[train_index],X_data[train_index]
    y_train,y_test = y_data[train_index],y_data[train_index]
    scores.append(SVM(X_train,X_test,y_train,y_test))
    
    
accuracy = 0.00
precision = 0.00
recall = 0.00
f1 = 0.00

for i in scores:
    accuracy+=i[0]
    precision+=i[1]
    recall+=i[2]
    f1+=i[3]

accuracy = accuracy/5.0
precision = precision/5.0
recall = recall/5.0
f1 = f1/5.0
    



