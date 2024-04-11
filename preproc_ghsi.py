import pandas as pd
import pywt

ghsi = pd.read_csv("data/ghsi.csv")
ghsi_min = ghsi.sample(frac=0.04, random_state=10)
ghsi_min.to_csv("data/ghsi_min.csv", index=False)
