
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
#   Function Name : Split Data
#   Description   : spliting activity
#   Input         : Data frame
#   output        : 4 subset for traning and testing
#   Author        : Prajakta Sharad Jagtap
#   Date          : 16-08-26  
#----------------------------------------------------------------
def SplitData(df):
    X =df.drop("Survived",axis = 1)
    Y = df["Survived"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )
    print("data set spliting completed successfully")
    return X_train, X_test, Y_train, Y_test
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

    X_train, X_test, Y_train, Y_test = SplitData(df)

if __name__ == "__main__":
    main()
