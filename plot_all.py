import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_ghsi():
    df = pd.read_csv(r"data/ghsi_min.csv")
    data = df.iloc[0,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

    data = df.iloc[20,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()

    data = df.iloc[30,1:].to_numpy()
    x = list(range(data.shape[0]))
    plt.plot(x,data)
    plt.show()


def plot_ghsi_many():
    df = pd.read_csv(r"data/ghsi.csv")
    df = df.sample(frac=0.0007, random_state=2)
    data = df.iloc[:,1:].to_numpy()
    x = list(range(data.shape[1]))
    for i in range(data.shape[0]):
        plt.plot(x,data[i])
    data = np.mean(data, axis=0)
    plt.plot(x, data, linestyle='--')
    plt.title("GHSI")
    plt.show()

if __name__ == "__main__":
    plot_ghsi_many()
