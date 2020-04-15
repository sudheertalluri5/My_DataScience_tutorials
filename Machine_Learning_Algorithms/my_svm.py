# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:52:38 2019

@author: sudhe
"""

import numpy as np
from sklearn import svm
feature_train=np.array([[-1,1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
feature_test=np.array([[-0.8,-1],[1,2]])
label_train=np.array([1,1,1,2,2,2])
clf=svm.SVC(gamma='scale')
clf.fit(feature_train,label_train)
print(clf.predict(feature_test))