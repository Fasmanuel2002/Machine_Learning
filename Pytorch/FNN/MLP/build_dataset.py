from typing import List, Dict, Tuple
import torch

def build_dataset(words_corpus : List[str], block_size : int,  stoi : Dict[int, str]) -> Tuple:
    X_inputs, Y_labels = [], []
    
    for word in words_corpus:
        #print(word)

        context = [0] * block_size
        #print(f"The context {context}")
        for ch in word + '.':
            ix = stoi[ch]
            #print(f"the ix: {ix}")
            X_inputs.append(context)
            Y_labels.append(ix)
            context = context[1:] + [ix]

    X_inputs = torch.tensor(X_inputs)
    Y_labels = torch.tensor(Y_labels)
    print(X_inputs.shape, Y_labels.shape)
    return (X_inputs, Y_labels) 
