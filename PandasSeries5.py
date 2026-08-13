import pandas as pd

def main():
    sobj = pd.Series([11.2,21,51,101],index = [5,6,7,8])   #to change index 

    print(sobj)

    print(sobj[7])

if __name__ == "__main__":
    main()
