
import numpy as np
import pandas as pd 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#----------------------------------------------------------------
#   Function Name : LoadData
#   Description   : Load the data from csv
#   Input         :Name of csv file
#   output        : Data frame
#   Author        : Prajakta Sharad Jagtap
#   Date          : 16-08-26  
#----------------------------------------------------------------


def LoadData(filename):
    df = pd.read_csv(filename)

    print("dataset loaded successfully")
    print(df.head())

    return df

#step : 2   

#----------------------------------------------------------------
#   Function Name : Data Preprocessing
#   Description   : It perform data analystics
#   Input         : Data frame
#   output        : updated Data frame
#   Author        : Prajakta Sharad Jagtap
#   Date          : 16-08-26  
#----------------------------------------------------------------
def  PreProcessData(df):
    df = df.drop([
        "Passengerid",
        "zero",
        "name" ,
    ],
    errors = "ignore"
    )

    #Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    #convert catagorical to numeric data
    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first= True,
        dtype=int
    )

    print(df.head())
    print("Data preprocessing completed")
        
    return df

#----------------------------------------------------------------
#   Function Name : main
#   Description   : Entry point function
#   Input         : none
#   output        : none
#   Author        : Prajakta Sharad Jagtap
#   Date          : 16-08-26  
#----------------------------------------------------------------


def main():
    #step 1
    df = LoadData("MarvellousTitanicDataset.csv")

    # step 2
    df = PreProcessData(df)

if __name__ == "__main__":
    main()
