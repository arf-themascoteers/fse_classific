import pandas as pd

src = "data/ghsi.csv"
df = pd.read_csv(src)
columns_with_nan = df.columns[df.isna().any()]
print(columns_with_nan)
print(len(df.columns))
unique_values = df['crop'].unique()
print(unique_values)
num_rows_with_nan = df.isna().any(axis=1).sum()
print(num_rows_with_nan)
df.dropna(inplace=True)
df.to_csv(src, index=False)