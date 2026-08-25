
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

#----------------------------------------------------------------
#   Function Name : LoadData
#   Description   : Entry point function
#   Input         :Name of csv file
#   output        : Data frame
#   Author        : Prajakta Sharad Jagtap
#   Date          : 16-08-26  
#----------------------------------------------------------------


def main():
    LoadData("MarvellousTitanicDataset.csv")
    
if __name__ == "__main__":
    main()
