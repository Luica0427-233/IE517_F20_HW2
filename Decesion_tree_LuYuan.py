# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 02:11:48 2020

@author: Lucia
"""

import pandas as pd
df = pd.read_csv("D:\Desktop\ML in Fin Lab\Module2\HW2\Treasury squeeze test - DS1.csv")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X = df.iloc[:,2:-1]
y = df['squeeze']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
sc.fit(X_test)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

dt = DecisionTreeClassifier(max_depth = 6, random_state = 1)
dt.fit(X_train_std, y_train)
y_pred = dt.predict(X_test_std)

#evaluate
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))

print("-----------")
print("My name is Lu Yuan")
print("My NetID is: luyuan3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")