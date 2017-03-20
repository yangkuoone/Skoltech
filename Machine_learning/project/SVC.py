
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from time import time
import copy as copy

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.multiclass import OneVsRestClassifier, OutputCodeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, roc_auc_score
from sklearn.neighbors.kde import KernelDensity


# In[ ]:

data = pd.read_csv(open("diabetic_data.csv"))
# print('Columns', data.columns)
data.drop('weight', axis=1, inplace=True)
data.drop('payer_code', axis=1, inplace=True)
data.drop('medical_specialty', axis=1, inplace=True)
n_samples = data.shape[0]


# In[18]:

LE = LabelEncoder()

s = 1
for column in data.columns[:-1]:
    LE.fit(data[column])
    data[column] = s * LE.transform(data[column])


med_col = data.columns[9:-1]
drug_col = data.columns[21:-3]
X_med = np.asarray(data[med_col])
X_drug = np.asarray(data[drug_col])

LE.fit(data['readmitted'])
data['readmitted'] = LE.transform(data['readmitted'])

X, y = np.asarray(data[data.columns[2:-1]]), np.asarray(data['readmitted'])


# In[ ]:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)


def binarization (y, y_train, y_test, param=2):
    
    y_bin = np.zeros(len(y))
    y_bin_train = np.zeros(len(y_train))
    y_bin_test = np.zeros(len(y_test))

    for i in range(len(y)):
        if y[i] < param:
            y_bin[i] = 1
        else:
            y_bin[i] = 0
    
    for i in range(len(y_train)):
        if y_train[i] < param:
            y_bin_train[i] = 1
        else:
            y_bin_train[i] = 0

    for i in range(len(y_test)):
        if y_test[i] < param:
            y_bin_test[i] = 1
        else:
            y_bin_test[i] = 0

    return y_bin, y_bin_train, y_bin_test


y_bin, y_bin_train, y_bin_test = binarization(y, y_train, y_test, param=1)

svc = SVC()
svc.fit(X_train, y_bin_train)
y_pred = svc.predict(X_test)


file = open(file='SVCAccuracy1.txt', mode='w')
roc_auc = roc_auc_score(y_score=y_pred, y_true=y_bin_test)
np.savetxt('SVCAccuracy1.txt', roc_auc)
file.close()

file = open(file='SVCprecision1.txt', mode='w')
np.savetxt('SVCprecision1.txt', y_pred)
file.close()

y_bin, y_bin_train, y_bin_test = binarization(y, y_train, y_test, param=2)

svc = SVC()
svc.fit(X_train, y_bin_train)
y_pred = svc.predict(X_test)

file = open(file='SVCAccuracy2.txt', mode='w')
roc_auc = roc_auc_score(y_score=y_pred, y_true=y_bin_test)
np.savetxt('SVCAccuracy2.txt', roc_auc)
file.close()

file = open(file='SVCprecision2.txt', mode='w')
np.savetxt('SVCprecision2.txt', y_pred)
file.close()

