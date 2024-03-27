# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

input_file = "output_pm.csv"
data = pd.read_csv(input_file, header=0, index_col=0)

features = data.iloc[:, 0:]

scaler = StandardScaler()

scaler.fit(features)

joblib.dump(scaler, 'scaler.pkl')

scaled_features = scaler.fit_transform(features)

scaled_data = pd.DataFrame(data=scaled_features, index=data.index, columns=features.columns)

output_file = "output_pms.csv"
scaled_data.to_csv(output_file)

print("标准化完成，结果已保存到output.csv文件中。")

# %%



