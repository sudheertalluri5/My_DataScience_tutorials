# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:05:29 2019

@author: sudhe
"""

import numpy as np

features_train=np.array([[-1,1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
labels_train=np.array([1,1,1,2,2,2])
features_test=np.array([[-0.8,-1],[-2,-2],[1,2],[1,3]])
labels_test=np.array([1,1,2,2])#given all the labels according to model only so accuracy is 100 percent in practical data as shown in udacity we got 0.884 as accuracy
from sklearn.naive_bayes import GaussianNB
 ### create classifier
clf =GaussianNB()    ### fit the classifier on the training features and labels
    #TODO
clf.fit(features_train,labels_train)
    ### use the trained classifier to predict labels for the test features
pred = clf.predict(features_test)


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred,True,None)
print(accuracy)