# %%
import pandas as pd

input_file = "71.csv"
data = pd.read_csv(input_file, index_col=0)

correlation_matrix = data.corr()

columns_to_drop = set()

threshold = 0.8

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if correlation_matrix.iloc[i, j] > threshold:
            colname = correlation_matrix.columns[i]
            columns_to_drop.add(colname)

data_filtered = data.drop(columns=columns_to_drop)

output_file = "output_p.csv"
data_filtered.to_csv(output_file)
