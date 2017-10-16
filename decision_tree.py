# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:45:19 2017

@author: Rafiya
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
estimator=DecisionTreeClassifier(criterion = "entropy")

params=np.linspace(1,30,30).astype(int)
train_scores, test_scores = validation_curve(DecisionTreeClassifier(criterion = "entropy"), Xa_train, ya_train.astype(int), "max_depth",params, cv=3)
train_scoresB, test_scoresB = validation_curve(DecisionTreeClassifier(criterion = "entropy"), Xb_train, yb_train.astype(int), "max_depth",params, cv=3)

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
plt.title("Data A: Validation curve vs. Depth of Tree")
plt.xlabel("Depth")
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

plt.savefig('validation_DecisionTree_A.png')

plt.figure(1)
plt.title("Data B: Validation curve vs. Depth of Tree")
plt.xlabel("Depth")
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

plt.savefig('validationcurves_DecisionTree_B.png')


clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth=13, min_samples_leaf=5)
clf_entropy.fit(Xa_train, ya_train.astype(int))
predictions = clf_entropy.predict(Xa_test)

print(accuracy_score(ya_test.astype(int),predictions))
print(classification_report(ya_test.astype(int),predictions))
print(confusion_matrix(ya_test.astype(int),predictions))

clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth=13, min_samples_leaf=5)
clf_entropy.fit(Xb_train, yb_train.astype(int))
predictions = clf_entropy.predict(Xb_test)

print(accuracy_score(yb_test.astype(int),predictions))
print(classification_report(yb_test.astype(int),predictions))
print(confusion_matrix(yb_test.astype(int),predictions))
