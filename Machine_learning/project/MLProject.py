
# coding: utf-8

# In[1]:

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

get_ipython().magic('matplotlib inline')


# # Working with data

# In[2]:

data = pd.read_csv(open("dataset_diabetes/diabetic_data.csv"))
print('Columns', data.columns)
data.drop('weight', axis=1, inplace=True)
data.drop('payer_code', axis=1, inplace=True)
data.drop('medical_specialty', axis=1, inplace=True)
n_samples = data.shape[0]


# In[3]:

print ('Number of samples', n_samples)


# ### Encoding of categorical features

# In[4]:

LE = LabelEncoder()


# In[5]:

# Transform the categorical features

s = 1
for column in data.columns[:-1]:
    LE.fit(data[column])
    data[column] = s * LE.transform(data[column])
#     s += 1
    


# #### All data and relevant medical data

# In[12]:

med_col = data.columns[9:-1]
drug_col = data.columns[21:-3]
X_med = np.asarray(data[med_col])
X_drug = np.asarray(data[drug_col])

LE.fit(data['readmitted'])
data['readmitted'] = LE.transform(data['readmitted'])
print ('Label encoding for readmitted', LE.classes_)

X, y = np.asarray(data[data.columns[2:-1]]), np.asarray(data['readmitted'])


# ### Data split

# In[13]:

X_train_med, X_test_med, y_train, y_test = train_test_split(X_med, y, 
                                                                    test_size=0.5, random_state=1, stratify=y)


# In[14]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)


# In[15]:

X_train_drug, X_test_drug, y_train, y_test = train_test_split(X_drug, y, 
                                                                    test_size=0.1, random_state=1, stratify=y)


# ### Label binarization
# Here we use binariztion to predict readmission in any time 

# In[16]:

# LE.fit(data['readmitted'])
# data['readmitted'] = LE.transform(data['readmitted'])
# print ('Label encoding for readmitted', LE.classes_)


# In[17]:

def binarization (y_train, y_test, param=2):
   
    y_bin_train = np.zeros(len(y_train))
    y_bin_test = np.zeros(len(y_test))

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

    return y_bin_train, y_bin_test


# In[18]:

y_bin_train, y_bin_test = binarization(y_train, y_test, param=1)


# --------------------------------------------------------------------------

# ## Emergency visits

# In[22]:

#Number of emergency visits of the patient in the year preceding the encounter

n_emergency = np.asarray(data['number_emergency'])
emergency_values = np.unique(n_emergency)


# In[23]:

#Age intervals, for example [10, 20)

ages = np.asarray(data["age"])
ages_types = np.unique(ages)


# In[24]:

probability = []
for age in ages_types:
    emergency = n_emergency[np.where(ages == age)]
    probability.append(emergency.sum() / len(emergency))
    print(age, emergency.sum() / len(emergency))


# In[25]:

fig = plt.figure(figsize=(10,7))
plt.plot(np.arange(len(ages_types)), np.log(probability), marker='o')
plt.xticks(np.arange(len(ages_types)), ages_types)
plt.title("Dependency of age and the probability of emergency visit")
plt.xlabel("Age")
plt.ylabel("Probability of emergency visit")


# -------------------

# # Visualization

# ### PCA to visualize

# In[27]:

n_components = 2

t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("PCA is done in %0.3fs" % (time() - t0))

X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test)
print("Transformation is done in %0.3fs" % (time() - t0))


# In[28]:

x0 = [x[0] for x in X_train_pca]
x1 = [x[1] for x in X_train_pca]
# colors = [cm.Vega10((y_) / max(y_train)) for y_ in y_bin_train]
plt.figure(figsize=(10, 7))
ax = plt.axes(frameon=False)
plt.scatter(x0, x1, c=y_train, edgecolor = 'none', s = 5)
plt.xlabel('x0')
plt.ylabel('x1')
# ax.set_xlim([-2e14, 2e14])
# ax.set_ylim([-0.25e14, 0.25e14])
plt.title('Two prinicipal components visualisation')


# ### KernelPCA

# Аккуратно с запуском! Здесь перепутаны train и test. Запуск опасен для жизни компа!

# In[128]:

n_components = 3

t0 = time()
kpca = KernelPCA(n_components=n_components, kernel='cosine').fit(X_train_med)
print("KernelPCA is done in %0.3fs" % (time() - t0))

X_train_kpca = kpca.transform(X_train_med)
X_test_kpca = kpca.transform(X_test_med)
print("Transformation is done in %0.3fs" % (time() - t0))


# In[165]:

x0 = [x[0] for x in X_test_kpca]
x1 = [x[1] for x in X_test_kpca]
# colors = [cm.Vega10((y_) / max(y_train)) for y_ in y_bin_train]
plt.figure(figsize=(10, 7))
ax = plt.axes(frameon=False)
plt.scatter(x0, x1, c=y_test, edgecolor = 'none', s = 5)
plt.xlabel('x0')
plt.ylabel('x1')
# ax.set_xlim([-2e14, 2e14])
# ax.set_ylim([-0.25e14, 0.25e14])
plt.title('Two prinicipal components visualisation')


# In[132]:

colors = [cm.Vega10((y_) / max(y_train)) for y_ in y_train]

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X_test_kpca[:, 0], X_test_kpca[:, 1], X_test_kpca[:, 2], c=y_test)
plt.show()


# ### t-SNE

# In[ ]:

n_components = 3
t0 = time()
print('Fitting t-SNE...............')
tsne = TSNE(n_components=n_components)
X_train_tsne = tsne.fit_transform(X_train)
print('Fitting is done in %0.3s' % (time() - t0))


# In[ ]:




# --------

# # Classification
# ## Binary classification

# ### AdaBoost with random under-sampling

# We have an imbalanced data, so we want to use RUS

# In[21]:

rus = RandomUnderSampler()
X_train_med_rus, y_bin_train_rus = rus.fit_sample(X_train_med, y_bin_train)
X_test_med_rus, y_bin_test_rus = rus.fit_sample(X_test_med, y_bin_test)


# In[23]:

AB = AdaBoostClassifier(base_estimator=LogisticRegression(), n_estimators=200)
print('Fitting classifier')
t0 = time()
AB.fit(X_train_med_rus, y_bin_train_rus)
print('fitting is done in %s ' % (time() - t0))
print('Accuracy', accuracy_score(y_pred=AB.predict(X_test_med_rus), y_true=y_bin_test_rus))


# In[26]:

y_bin_test.shape


# In[278]:

y_pred_proba = AB.predict_proba(X_test_med)[:, 0]
precision_AB_rus, recall_AB_rus, _ = precision_recall_curve(y_true=y_bin_test, probas_pred=y_pred_proba)


# In[279]:

plt.title('Precision Recall curve for AdaBoost fitting over undersampling training data')
plt.plot(recall_AB_rus, precision_AB_rus)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


# ### AdaBoost without RUS

# In[101]:

AB = AdaBoostClassifier(base_estimator=LogisticRegression(), n_estimators=200)


# In[104]:

print('Fitting classifier')
t0 = time()
AB.fit(X_train_med, y_bin_train)
print('fitting is done in %s ' % (time() - t0))
print('Accuracy', accuracy_score(y_pred=AB.predict(X_test_med), y_true=y_bin_test))


# In[105]:

y_pred_proba = AB.predict_proba(X_test_med)[:, 0]


# In[107]:

precision_AB, recall_AB, _ = precision_recall_curve(y_true=y_bin_test, probas_pred=y_pred_proba)


# In[108]:

plt.title('Precision Recall curve for AdaBoost fitting over undersampling training data')
plt.plot(recall_AB, precision_AB)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


# ### MLPClassifier

# In[287]:

mlp = MLPClassifier(hidden_layer_sizes=(35, 35, 35, 35, 35), max_iter=1000, activation='logistic')


# In[289]:

print('Fitting classifier')
t0 = time()
mlp.fit(X_train_med, y_bin_train)
print('fitting is done in %s ' % (time() - t0))


# In[291]:

print('Accuracy', list(mlp.predict(X_test_med) == y_bin_test).count(True) / len(y_bin_test))


# In[292]:

# print ('Number of ones in test prediction', np.sum(mlp.predict(X_test_med)))
# print ('Number of ones in y_test', np.sum(y_bin_test))


# ### Naive Bayes

# In[121]:

NB = BernoulliNB()


# In[122]:

print('Fitting classifier')
t0 = time()
NB.fit(X_train_med_rus, y_bin_train_rus)
print('fitting is done in %s ' % (time() - t0))


# In[124]:

print ('Accuracy', accuracy_score(y_pred=NB.predict(X_test_med), y_true=y_bin_test))


# ### Discriminant Analysis

# In[296]:

QDA = QuadraticDiscriminantAnalysis()
LDA = LinearDiscriminantAnalysis()


# In[297]:

QDA.fit(X_train, y_bin_train)
LDA.fit(X_train, y_bin_train)


# In[300]:

print ('QDA accuracy', accuracy_score(y_pred=QDA.predict(X_test), y_true=y_bin_test))
print ('LDA accuracy', accuracy_score(y_pred=LDA.predict(X_test), y_true=y_bin_test) )


# In[ ]:

# list(QDA.predict(X_test_med)).count(0)


# In[ ]:

# list(LDA.predict(X_test_med)).count(0)


# In[ ]:

# list(y_bin_test_med).count(0)


# ## Multiclass classification

# ### Random forest classifier

# In[293]:

forest = RandomForestClassifier(n_estimators=100, random_state=1)
t0 = time()
print("Fitting in forest..")
forest.fit(X_train, y_train)
print("Done in %s seconds" %(time() - t0))


# In[295]:

list(forest.predict(X_test) == y_test).count(True) / len(y_test)


# In[ ]:

plt.plot(range(X_train_med.shape[1]), forest.feature_importances_)


# In[285]:

np.argmax(forest.feature_importances_)


# In[286]:

data.columns[9]


# Choosing the best parameter via grid search

# In[302]:

print("Fitting the classifier to the training set..")
t0 = time()
param_grid = {'n_estimators': np.arange(5, 150, 10)}
forest = GridSearchCV(RandomForestClassifier(random_state=1), param_grid)
forest.fit(X_train_med, y_bin_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(forest.best_estimator_)


# In[305]:

print ('Accuracy', list(forest.best_estimator_.predict(X_test_med) == y_bin_test).count(True) / len(y_bin_test))


# -----

# ### OneVsRestClassifier

# In[167]:

OVR = OneVsRestClassifier(estimator=LogisticRegression())


# In[174]:

print('Fitting classifier')
t0 = time()
OVR.fit(X_train, y_train)
print('Fitting is done in %0.3fs' % (time() - t0))


# In[175]:

y_pred = OVR.predict(X_test)


# In[177]:

print ('Accuracy', accuracy_score(y_true=y_test, y_pred=y_pred))


# ### OutoutCodeClassifier (without binarization)

# In[133]:

OCC = OutputCodeClassifier(code_size=50, estimator=LogisticRegression())


# In[9]:

OCC.fit(X_train, y_train)


# In[10]:

list(OCC.predict(X_test) == y_test).count(True) / len(y_test)


# In[ ]:



