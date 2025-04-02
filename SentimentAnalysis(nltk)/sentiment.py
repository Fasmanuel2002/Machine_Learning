import pandas as pd
import numpy as np
import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer

def main():
    df = pd.read_csv('SentimentAnalisisNew.csv')
    
    X = df.iloc[:, 2:6].values
    Y = df['status']
    
    status_map = {"Anxiety":[1,0,0,0,0,0,0], "Normal":[0,1,0,0,0,0,0], "Suicidal":[0,0,1,0,0,0,0],
                  "Depression":[0,0,0,1,0,0,0], "Stress":[0,0,0,0,1,0,0],
                  "Bi-Polar":[0,0,0,0,0,1,0], "Personality Disorder":[0,0,0,0,0,0,1]}
    Y_status_Map = pd.DataFrame(Y.map(status_map).tolist(), columns=["Anxiety","Normal","Suicidal","Depression","Stress","Bi-Polar","Personality Disorder"]).values()


## 1)Layers
def layer(X,Y_status_Map):
    n_x = X.shape[1]
    n_y = Y_status_Map.shape[1]

    return n_x, n_y


##2)Parameters Initialization
def parameters_initialization(n_x,n_y):
    W = np.random.rand(n_x,n_y)
    b = np.zeros((n_y,1))

    
    parameters = {
        "W":W,
        "b":b
    }
    

    return parameters

## 3) Activation "SoftMax"

def activation_softMax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)
    
## 4) Forward Propagation
def forward_propagation(X,parameters):
    W = parameters['W']
    b = parameters['b']
    
    Z = np.matmul(X, W) + b.T
    Z_softMax = activation_softMax(Z)

    return Z_softMax

## 5) Log Loss function in SoftMax Likehood(Maximize)

# y = Y_status_Map
# h(x) = ^y 
def cost_fuction(Y_status_Map,Z_softMax):

    m = Y_status_Map.shape[0]
    cost_fuctionSoftMax = -np.sum(Y_status_Map * np.log(Z_softMax)) / m
    return cost_fuctionSoftMax

## 6) Back propagation
def backPropagation(Z_softMax,X, Y_status_Map, iterations):
    m = Y_status_Map.shape[0]

    dW = (X.T @ (Z_softMax - Y_status_Map)) / m
    db = np.sum(Z_softMax - Y_status_Map, axis=0, keepdims=True).T / m

    cache = {
        "dW":dW,
        "db":db
    }
    
    return cache

def gradient_Descent(parameters,cache,learning_rate=1.2):
    
    #dW and dB
    dW = cache['dW']
    db = cache['db']
    
    #w and b
    W = parameters['W']
    b = parameters['b']
    
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    parameters = {
        "W":W,
        "b":b
    }
    return parameters
    
if __name__ == "__main__":
    main()