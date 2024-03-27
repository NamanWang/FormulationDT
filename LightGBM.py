# %%
from bayes_opt import BayesianOptimization
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

# %%
def evaluate_performance(best_model, X, y_true):
    
    y_pred = best_model.predict(X)
    y_prob = best_model.predict_proba(X)[:, 1] 

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    performance = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc,
        'MCC': mcc
    }
    return performance

# %%
# Load data
data = pd.read_csv("data_0.56.csv")
X = data.drop(columns=['target'])
y = data['target']

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store performance metrics
train_performance = []
val_performance = []
test_performance = []

# Loop through each fold
for train_index, test_index in skf.split(X, y):
    print("\nNew round")
    # Split data into train and test sets
    X_remain, X_test = X.iloc[train_index], X.iloc[test_index]
    y_remain, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Further split train data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_remain, y_remain, test_size=0.2, random_state=42)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_resampled_train, y_resampled_train = smote.fit_resample(X_train, y_train)

    # Define function to optimize
    def rf(num_leaves, max_depth, min_data_in_leaf, feature_fraction, bagging_fraction, bagging_freq, lambda_l1, lambda_l2):
        rf = LGBMClassifier( 
                                    num_leaves=int(num_leaves),
                                    max_depth=int(max_depth),
                                    min_data_in_leaf=int(min_data_in_leaf),
                                    feature_fraction=feature_fraction,
                                    bagging_fraction=bagging_fraction,
                                    bagging_freq=int(bagging_freq),
                                    lambda_l1=lambda_l1,
                                    lambda_l2=lambda_l2,
                                    random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        return matthews_corrcoef(y_val, y_pred)

    # Define parameter bounds for Bayesian optimization
    pbounds = {
        "num_leaves": (20, 100),
        "max_depth": (3, 10),
        "min_data_in_leaf": (10, 100),
        "feature_fraction": (0.5, 1.0),
        "bagging_fraction": (0.5, 1.0),
        "bagging_freq": (1, 10),
        "lambda_l1": (0.0, 10.0),
        "lambda_l2": (0.0, 10.0)
            }
    
    # Run Bayesian optimization
    optimizer = BayesianOptimization(
        f=rf,
        pbounds=pbounds,
        random_state=42, 
    )

    optimizer.maximize(init_points=10, n_iter=200)
    params = optimizer.max['params']
    print("Best MCC score: {:.3f}".format(optimizer.max['target']))

    params = optimizer.max['params']                 
    best_model=LGBMClassifier( 
                                    num_leaves=int(params['num_leaves']),
                                    max_depth=int(params['max_depth']),
                                    min_data_in_leaf=int(params['min_data_in_leaf']),
                                    feature_fraction=params['feature_fraction'],
                                    bagging_fraction=params['bagging_fraction'],
                                    bagging_freq=int(params['bagging_freq']),
                                    lambda_l1=params['lambda_l1'],
                                    lambda_l2=params['lambda_l2'], random_state=42
                                    )
    
    best_model_train=best_model.fit(X_resampled_train, y_resampled_train)
    
    performance_train = evaluate_performance(best_model_train, X_train, y_train)
    performance_val = evaluate_performance(best_model_train, X_val, y_val)

    X_resampled_remain, y_resampled_remain = smote.fit_resample(X_remain, y_remain)
    best_model_remain=best_model.fit(X_resampled_remain, y_resampled_remain)
    
    performance_test = evaluate_performance(best_model_remain, X_test, y_test)
    
    print("performance_train=")
    print(performance_train)
    print("performance_val=")
    print(performance_val)
    print("performance_test=")
    print(performance_test)

    train_performance.append(performance_train)
    val_performance.append(performance_val)
    test_performance.append(performance_test)

# Calculate average performance metrics
avg_train_performance = pd.DataFrame(train_performance).mean()
avg_val_performance = pd.DataFrame(val_performance).mean()
avg_test_performance = pd.DataFrame(test_performance).mean()
std_train_performance = pd.DataFrame(train_performance).std()
std_val_performance = pd.DataFrame(val_performance).std()
std_test_performance = pd.DataFrame(test_performance).std()

# Print average performance metrics
print("\n" +"Average Train Performance:" + str(avg_train_performance) + "+/-" + str(std_train_performance))
print("\n" +"Average Validation Performance:", avg_val_performance, "+/-", std_val_performance)
print("\n" +"Average Test Performance:", avg_test_performance, "+/-", std_test_performance)


