# %%
import pickle
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

plt.rcParams["font.family"] = "Arial"

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

explainer = shap.Explainer(model)

input_data = pd.read_csv("train_data.csv")
feature_names = input_data.columns[1:]
X = input_data.iloc[:, 1:].values

shap_values = explainer(X)

shap_values = shap_values[:,:,1]
shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=10)
plt.savefig("shap_summary_plot.png", format='png', dpi=300, bbox_inches='tight')
plt.show()
shap.summary_plot(shap_values, feature_names=feature_names, plot_type='bar', show=False, max_display=10)
plt.savefig("shap_bar_plot.png", format='png', dpi=300, bbox_inches='tight')
plt.show()

data=abs(shap_values.values)

column_means = np.mean(data, axis=0)
df = pd.DataFrame({'Column Means': column_means})
df.to_csv('column_means.csv', index=False)