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

    # step 2 : feature selection

    X = df[["AnnualIncome","SpendingScore"]]

    print("Feature selection done")
    print(X.head())

    #step 3 scaloe the data

    scalar = StandardScaler()

    X_scaled = scalar.fit_transform(X)
    print(" scaling Data")
    print(X_scaled[:5])

if __name__ == "__main__":
    main()