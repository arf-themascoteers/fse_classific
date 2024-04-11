import pandas as pd
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    plot_ghsi()
