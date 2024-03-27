# %%
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import mannwhitneyu

data = pd.read_csv("output_p.csv")

features = data.columns[1:]
labels = data[data.columns[0]]

smote = SMOTE(random_state=42)
balanced_features, balanced_labels = smote.fit_resample(data[features], labels)

p_values = []
for feature in features:
    u_stat, p_value = mannwhitneyu(balanced_features[feature][balanced_labels == 0], 
                                   balanced_features[feature][balanced_labels == 1])
    p_values.append((feature, p_value))

alpha = 0.01
selected_features = [feature for feature, p_value in p_values if p_value < alpha]

selected_data = data[['target'] + selected_features]
selected_data.to_csv('output_pm.csv', index=False)
# %%



