import pandas as pd
import numpy as np

def main():
    dataframe = pd.read_csv("cleaned_train.csv")
    
    X = dataframe.iloc[0:3,:].head()
    print(X)
if __name__ == "__main__":
    main()