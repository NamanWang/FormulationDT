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
s=2
n=51

# %%
def get_spy(positive, unlabeled):
  # Loop from 1 to i
  for j in range(1, n):
    # Randomly sample 14 rows from positive_0, excluding the first row
    spy_j = positive.iloc[1:].sample(s)
    # Save spy_j as a csv file
    spy_j.to_csv(f"spy_{j}.csv", index=False)
    # Drop the sampled rows from positive_0 and save the remaining rows as a new csv file
    positive_j = positive.drop(spy_j.index)
    positive_j.to_csv(f"positive_{j}.csv", index=False)
    # Append spy_j to the end of unlabeled_0 and save the resulting dataframe as a new csv file
    unlabeled_j = unlabeled.append(spy_j)
    unlabeled_j.to_csv(f"unlabeled_{j}.csv", index=False)

# %%
def PUbagging(data_P, data_U, j):
    data_P = data_P.iloc[:, 1:].values
    data_U = data_U.iloc[:, 1:].values

    NP = data_P.shape[0]
    NU = data_U.shape[0]

    T = 1000
    K = NP
    train_label = np.zeros(shape=(NP+K))
    train_label[:NP] = 1.0

    n_oob = np.zeros(shape=(NU,))
    f_oob = np.zeros(shape=(NU, 2))

    pred_process = pd.DataFrame()

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
        #print(idx_oob)
        # Transductive learning of oob samples
        f_oob[idx_oob] += model.predict_proba(data_U[idx_oob])
        n_oob[idx_oob] += 1
        predict_proba = f_oob[:, 1]/n_oob
        predict_proba = pd.DataFrame(predict_proba)
        pred_process[f'col{i}'] = predict_proba

    predict_proba.to_csv(f"predict_proba_{j}.csv", index=False)
    #pred_process.to_csv(f"pred_process.csv_{j}", index=False)

# %%
# Read the files as dataframes
positive_0 = pd.read_csv("positive_0.csv")
unlabeled_0 = pd.read_csv("unlabeled_0.csv")
get_spy(positive_0, unlabeled_0)

# %%
for j in range(1, n):
    data_P = pd.read_csv(f"positive_{j}.csv")
    data_U = pd.read_csv(f"unlabeled_{j}.csv")
    PUbagging(data_P, data_U, j)

# %%
T = pd.DataFrame()
for j in range(1, n):
    predict_proba = pd.read_csv(f"predict_proba_{j}.csv")
    lines = predict_proba.iloc[-s:]
    T=T.append(lines)
T.to_csv(f"positive_score_validation_points.csv", index=False)

# %%



