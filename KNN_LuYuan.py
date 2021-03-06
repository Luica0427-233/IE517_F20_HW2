# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 02:11:48 2020

@author: Lucia
"""

import pandas as pd
df = pd.read_csv("D:\Desktop\ML in Fin Lab\Module2\HW2\Treasury squeeze test - DS1.csv")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

X = df.iloc[:,2:-1]
y = df['squeeze']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
sc.fit(X_test)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Test multiple different K inputs
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 20)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k)

    # Fit the classifier to the training data
    knn.fit(X_train_std,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train_std,y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test_std, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

#According to the picture, The best choice of K is 7.
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train_std,y_train)
test_accuracy = knn.score(X_test_std, y_test)
print("Test set accuracy for k=7: {:.2f}".format(test_accuracy))

print("-----------")
print("My name is Lu Yuan")
print("My NetID is: luyuan3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")