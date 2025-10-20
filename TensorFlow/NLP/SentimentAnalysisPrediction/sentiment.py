import nltk 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

def main():
    df = pd.read_csv("SentimentAnalysis.csv")
    example = df['statement'][50659]
    
    """   
    tokens = nltk.word_tokenize(example)
    print(tokens[:10])


    tagged = nltk.pos_tag(tokens)
    print(tagged[:10])
    
    
    entities = nltk.chunk.ne_chunk(tagged)
    entities.pprint()
    """
    
    sia = SentimentIntensityAnalyzer()
    print(sia.polarity_scores(example))
    print(example)
    
    
    
    
    res = {}
    for i , row in tqdm(df.iterrows(),total=len(df)) :
        opinions = row['statement']
        perId = row['id']
        res[perId]= sia.polarity_scores(opinions)
    
    
    print(res)
    

if __name__ == '__main__':
    main()