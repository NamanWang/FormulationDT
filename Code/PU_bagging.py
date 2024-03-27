# %%
from __future__ import division, print_function
import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import pandas as pd

# %%
data_P = pd.read_csv("positive_0.csv")
data_P = data_P.iloc[:, 1:].values
data_U = pd.read_csv("unlabeled_0.csv")
data_U = data_U.iloc[:, 1:].values

# %%
NP = data_P.shape[0]
NU = data_U.shape[0]

T = 1000
K = NP
train_label = np.zeros(shape=(NP+K))
train_label[:NP] = 1.0
print(train_label)

# %%
n_oob = np.zeros(shape=(NU,))
f_oob = np.zeros(shape=(NU, 2))
print(n_oob)
print(f_oob)
pred_process = pd.DataFrame()

# %%
for i in range(T):
    # Bootstrap resample
    bootstrap_sample = np.random.choice(np.arange(NU), replace=True, size=K)
    # Positive set + bootstrapped unlabeled set
    data_bootstrap = np.concatenate((data_P, data_U[bootstrap_sample, :]), axis=0)
    # Train model
    #model = KNeighborsClassifier()
    #model = GaussianNB()
    #model = SVC()
    model = DecisionTreeClassifier()
    #model = LogisticRegression()
    model.fit(data_bootstrap, train_label)
    # Index for the out of the bag (oob) samples
    idx_oob = sorted(set(range(NU)) - set(np.unique(bootstrap_sample)))
    print(idx_oob)
    # Transductive learning of oob samples
    f_oob[idx_oob] += model.predict_proba(data_U[idx_oob])
    n_oob[idx_oob] += 1
    predict_proba = f_oob[:, 1]/n_oob
    predict_proba = pd.DataFrame(predict_proba)
    pred_process[f'col{i}'] = predict_proba

predict_proba.to_csv("t_predict_proba.csv", index=False)
pred_process.to_csv("pred_process.csv", index=False)

# %%



