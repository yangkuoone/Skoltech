
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
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
from imblearn.under_sampling import RandomUnderSampler


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


# In[ ]:

n_components = 2

t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("PCA is done in %0.3fs" % (time() - t0))

X_train_pca = pca.transform(X_train)
file = open(file='PCA.txt', mode='w')
np.savetxt('PCA.txt', X_train_pca)
file.close()


# In[ ]:



