import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm


def main():
    df = pd.read_csv("SentimentAnalysis.csv")
    #example = df['statement'][50659]
    #print(example)
    res_main = dict()
    res_main = polarity_sc(df)
    print(res_main)
    
    vaders = pd.DataFrame(res_main).T
    vaders = vaders.reset_index().rename(columns={'index': 'id'})
    vaders = vaders.merge(df,how='left')
    
    vaders.to_csv('SentimentAnalisisNew.csv')
    
    



def polarity_sc(df):
    res = {}
    sia = SentimentIntensityAnalyzer()
    for i, rows in tqdm(df.iterrows(), total=len(df)):
        feeling_rows = rows['statement']
        id_rows = rows['id']
        res[id_rows] = sia.polarity_scores(repr(feeling_rows))
    return res
if __name__ == '__main__':
    main()