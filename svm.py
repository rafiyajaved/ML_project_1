# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:29:35 2017

@author: Rafiya
"""

import numpy as np
from sklearn.svm import SVC
from sklearn import datasets, svm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn import preprocessing



a=pd.read_csv('EDdata.csv', encoding="ISO-8859-1")
b=pd.read_csv('HPSAdata.csv', encoding="ISO-8859-1")


Xa = a.values[:, 2:26]
ya = a.values[:,28]

Xa= preprocessing.scale(Xa)
ya=preprocessing.scale(ya)

Xb = b.values[:, 2:26]
yb = b.values[:,27]

Xb= preprocessing.scale(Xb)
yb=preprocessing.scale(yb)

Xa_train, Xa_test, ya_train, ya_test = train_test_split( Xa, ya, test_size = 0.3, random_state = 100)
Xb_train, Xb_test, yb_train, yb_test = train_test_split( Xb, yb, test_size = 0.3, random_state = 100)


params = np.logspace(-7, 3, 20)
  
clf=SVC(C=0.5)
train_scores, test_scores = validation_curve(clf, Xa_train, ya_train.astype(int), param_name="gamma",scoring="accuracy",param_range=params, cv=3)
train_scoresB, test_scoresB = validation_curve(clf, Xb_train, yb_train.astype(int), param_name="gamma",scoring="accuracy",param_range=params, cv=3)



train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

train_scores_meanB = np.mean(train_scoresB, axis=1)
train_scores_stdB = np.std(train_scoresB, axis=1)
test_scores_meanB = np.mean(test_scoresB, axis=1)
test_scores_stdB = np.std(test_scoresB, axis=1)



plt.figure(0)
plt.title("Data A: Validation curve vs. Gamma")
plt.xlabel("gamma")
plt.ylabel("Score")

lw=2

plt.semilogx(params, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(params, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(params, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(params, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")

plt.savefig('validation_SVM_A.png')


plt.figure(1)
plt.title("Data B: Validation curve vs. Gamma")
plt.xlabel("gamma")
plt.ylabel("Score")


plt.semilogx(params, train_scores_meanB, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(params, train_scores_meanB - train_scores_stdB,
                 train_scores_meanB + train_scores_stdB, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(params, test_scores_meanB, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(params, test_scores_meanB - test_scores_stdB,
                 test_scores_meanB + test_scores_stdB, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")


plt.savefig('validation_SVM_B.png')



clf=SVC(gamma=1, C=12)
clf.fit(Xa_train, ya_train.astype(int))
predictions = clf.predict(Xa_test)

print(accuracy_score(ya_test.astype(int),predictions))
print(classification_report(ya_test.astype(int),predictions))
print(confusion_matrix(ya_test.astype(int),predictions))

print(clf.n_support_ )

clf.fit(Xb_train, yb_train.astype(int))
predictions = clf.predict(Xb_test)

print(accuracy_score(yb_test.astype(int),predictions))
print(classification_report(yb_test.astype(int),predictions))
print(confusion_matrix(yb_test.astype(int),predictions))

print(clf.n_support_ )