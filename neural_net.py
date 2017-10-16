# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:29:00 2017

@author: Rafiya
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

a=pd.read_csv('EDdata.csv', encoding="ISO-8859-1")


Xa = a.values[:, 2:26].astype(int)
ya = a.values[:,28].astype(int)

Xa_train, Xa_test, ya_train, ya_test = train_test_split( Xa, ya, test_size = 0.3, random_state = 100)


pipeline = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),])
pipeline.fit(Xa_train, ya_train)



gs = GridSearchCV(MLPClassifier(max_iter=5000,early_stopping=True,random_state=55), param_grid={
    'hidden_layer_sizes': [(30, 70,70,70)]})
gs.fit(Xa_train, ya_train)

gs=gs.best_estimator_

predictions = gs.predict(Xa_test)

print(accuracy_score(ya_test.astype(int),predictions))
print(classification_report(ya_test.astype(int),predictions))
print(confusion_matrix(ya_test.astype(int),predictions))