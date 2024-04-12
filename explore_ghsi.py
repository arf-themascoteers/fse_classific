import pandas as pd

d = "data/ghsi.csv"
df = pd.read_csv(d)
c = list(df.columns)
print(len(c))
print(len(df))
