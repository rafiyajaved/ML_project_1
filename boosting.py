# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:30:02 2017

@author: Rafiya
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve



a=pd.read_csv('EDdata.csv', encoding="ISO-8859-1")
b=pd.read_csv('HPSAdata.csv', encoding="ISO-8859-1")


Xa = a.values[:, 2:26]
ya = a.values[:,28]

Xb = b.values[:, 2:26]
yb = b.values[:,27]

Xa_train, Xa_test, ya_train, ya_test = train_test_split( Xa, ya, test_size = 0.3, random_state = 100)
Xb_train, Xb_test, yb_train, yb_test = train_test_split( Xb, yb, test_size = 0.3, random_state = 100)


cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
[-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
params= [1,2,5,10,20,30,45,60,80,100,150]
                    
               
boosterA = AdaBoostClassifier(algorithm='SAMME',learning_rate=1)
boosterB = AdaBoostClassifier(algorithm='SAMME',learning_rate=1)


train_scores, test_scores = validation_curve(boosterA, Xa_train, ya_train.astype(int), "n_estimators",params, cv=3)
train_scoresB, test_scoresB = validation_curve(boosterB, Xb_train, yb_train.astype(int), "n_estimators",params, cv=3)


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

train_scores_meanB = np.mean(train_scoresB, axis=1)
train_scores_stdB = np.std(train_scoresB, axis=1)
test_scores_meanB = np.mean(test_scoresB, axis=1)
test_scores_stdB = np.std(test_scoresB, axis=1)


print(params)
plt.figure(0)
plt.title("Data 1: Validation curve vs. Number of estimators")
plt.xlabel("n_estimators")
plt.ylabel("Score")

lw=2

plt.plot(params, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(params, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(params, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(params, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")

plt.savefig('validation_Boosting_A.png')

plt.figure(1)
plt.title("Data 2: Validation curve vs. Number of estimators")
plt.xlabel("n_estimators")
plt.ylabel("Score")

plt.plot(params, train_scores_meanB, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(params, train_scores_meanB - train_scores_stdB,
                 train_scores_meanB + train_scores_stdB, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(params, test_scores_meanB, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(params, test_scores_meanB - test_scores_stdB,
                 test_scores_meanB + test_scores_stdB, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")

plt.savefig('validationcurves_Boosting_B.png')


clf=AdaBoostClassifier(algorithm='SAMME',learning_rate=1,n_estimators=140)
clf.fit(Xa_train, ya_train.astype(int))
predictions = clf.predict(Xa_test)

print(accuracy_score(ya_test.astype(int),predictions))
print(classification_report(ya_test.astype(int),predictions))
print(confusion_matrix(ya_test.astype(int),predictions))

clf.fit(Xb_train, yb_train.astype(int))
predictions = clf.predict(Xb_test)

print(accuracy_score(yb_test.astype(int),predictions))
print(classification_report(yb_test.astype(int),predictions))
print(confusion_matrix(yb_test.astype(int),predictions))