# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:45:19 2017

@author: Rafiya
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


a=pd.read_csv('sampledata.csv')


X = a.values[:, 2:20]
Y = a.values[:,21]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

print(X_train)
print (y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 4,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

y_pred_en = clf_entropy.predict(X_test)
y_pred_en

print(y_pred_en)
