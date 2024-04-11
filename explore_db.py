import pandas as pd

src = "data/ghsi.csv"
df = pd.read_csv(src)
df['crop'], class_labels = pd.factorize(df['crop'])
print(df['crop'])
print(class_labels)