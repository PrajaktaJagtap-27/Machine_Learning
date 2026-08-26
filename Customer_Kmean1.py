import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def main():
    # step 1

    df = pd.read_csv("Mall_Customers.csv")
    print("Dataset Loaded with values")
    print(df.head())

    print("Missing values")

    print(df.null().sum())

if __name__ == "__main__":
    main()