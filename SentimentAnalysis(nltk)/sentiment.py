import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer

def main():
    df = pd.read_csv('SentimentAnalisisNew.csv')
    
    X = df.iloc[:, 2:6].values
    Y = df['status']
    
    status_map = {"Anxiety":[1,0,0,0,0,0,0], "Normal":[0,1,0,0,0,0,0], "Suicidal":[0,0,1,0,0,0,0],
                  "Depression":[0,0,0,1,0,0,0], "Stress":[0,0,0,0,1,0,0],
                  "Bi-Polar":[0,0,0,0,0,1,0], "Personality Disorder":[0,0,0,0,0,0,1]}
    
    
    mapped_values = Y.map(status_map)
    
    mapped_values = mapped_values.apply(lambda x: x if isinstance(x, list) else [0, 0, 0, 0, 0, 0, 0])
    Y_status_Map = pd.DataFrame(mapped_values.tolist(), columns=["Anxiety", "Normal", "Suicidal", "Depression", "Stress", "Bi-Polar", "Personality Disorder"])
    
    array_Y = np.array(Y_status_Map)
    
    ##Normalize X
    X_normalize = ((X - np.mean(X, axis=0)/ np.std(X, axis=0)))
    
    
    
    nn_network(X_normalize, array_Y, 1000, True)
## 1)Layers
def layer(X,Y_status_Map):
    n_x = X.shape[1]
    n_h = 2
    n_y = Y_status_Map.shape[1]

    return n_x, n_y


##2)Parameters Initialization
def parameters_initialization(n_x,n_h,n_y):
    
    
    #First Layer before Output
    W1 = np.random.rand(n_h,n_x)
    b1 = np.zeros((n_h,1))
    
    #Out put layers
    W2 = np.random.rand(n_y, n_h)
    b2 = np.zerps((n_y, 1))

    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    
    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
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
    cost_fuctionSoftMax = -np.sum(np.array(Y_status_Map) * np.log(Z_softMax)) / m
    return cost_fuctionSoftMax

## 6) Back propagation
def backPropagation(Z_softMax,X, Y_status_Map):
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
    

#8 NN_network
def nn_network(X, Y_status_Map, number_of_iterations = 10, printCostFalse = False):
    n_x, n_y = layer(X, Y_status_Map)
    
    parameters = parameters_initialization(n_x, n_y)
    
    for iteration in range(number_of_iterations):
        Z_softMax = forward_propagation(X, parameters)
        
        ActivationSoftMax = activation_softMax(Z_softMax)
        
        cost = cost_fuction(Y_status_Map, ActivationSoftMax)
        cache = backPropagation(ActivationSoftMax, X, Y_status_Map)
        
        parameters = gradient_Descent(parameters, cache, 1)
        
        if printCostFalse:
            print(f"For iteration: {iteration}:{cost} this is the cost")
        
    return parameters
        
        
        
if __name__ == "__main__":
    main()