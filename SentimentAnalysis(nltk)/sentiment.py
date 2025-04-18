import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle
def main():
    df = pd.read_csv('SentimentAnalisisNew.csv')
    
    X = df.iloc[:, 2:6].values
    X = X[:, :100]
    Y = df['status']
    
    status_map = {"Anxiety":[1,0,0,0,0,0,0], "Normal":[0,1,0,0,0,0,0], "Suicidal":[0,0,1,0,0,0,0],
                  "Depression":[0,0,0,1,0,0,0], "Stress":[0,0,0,0,1,0,0],
                  "Bi-Polar":[0,0,0,0,0,1,0], "Personality Disorder":[0,0,0,0,0,0,1]}
    
    
    mapped_values = Y.map(status_map)
    
    mapped_values = mapped_values.apply(lambda x: x if isinstance(x, list) else [0, 0, 0, 0, 0, 0, 0])
    Y_status_Map = pd.DataFrame(mapped_values.tolist(), columns=["Anxiety", "Normal", "Suicidal", "Depression", "Stress", "Bi-Polar", "Personality Disorder"])
    
    
    
    ##Normalize X
    X_normalize = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    
    X_small = X_normalize[:5000]
    Y_small = Y_status_Map[:5000]
    parameters = nn_network(X_small, Y_small, 2000, True)

    true_labels = np.argmax(Y_status_Map.values, axis=1)
    pred_labels = predict(X_normalize, parameters)
    accuracy = np.mean(true_labels == pred_labels)
    print(f"Training accuracy: {accuracy * 100:.2f}%")



# Save the trained parameters
    with open("model_parameters.pkl", "wb") as f:
        pickle.dump(parameters, f)

## 1)Layers
def layer(X,Y_status_Map):
    n_x = X.shape[1]
    n_h1 = 20
    n_h2 = 20
    n_y = Y_status_Map.shape[1]

    return n_x, n_h1,n_h2, n_y


##2)Parameters Initialization
def parameters_initialization(n_x,n_h1,n_h2,n_y):
    
    
    #First Layer before Output
    W1 = np.random.rand(n_h1,n_x) * 0.01
    b1 = np.zeros((n_h1,1))
    
    
    
    #Out put layers
    W2 = np.random.rand(n_h2, n_h1) * 0.01
    b2 = np.zeros((n_h2, 1))
    
    W3 = np.random.rand(n_y, n_h2) * 0.01
    b3 = np.zeros((n_y, 1)) 

    
    assert (W1.shape == (n_h1, n_x))
    assert (b1.shape == (n_h1, 1))
    assert (W2.shape == (n_h2, n_h1))
    assert (b2.shape == (n_h2, 1))
    assert (W3.shape == (n_y, n_h2))
    assert (b3.shape == (n_y, 1))

    
    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2,
        "W3":W3,
        "b3":b3
    }
    

    return parameters

## 3) Activation "SoftMax"

def activation_softMax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0
## 4) Forward Propagation
def forward_propagation(X,parameters):
    
    #From the first layer(input -> hidden layer)
    W1 = parameters['W1']
    b1 = parameters['b1']
    
    #From the hidden layer one to the hidden layer two
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    #from the hidden layer two to output 
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = np.matmul(W1, X.T) + b1  
    A1 = relu(Z1)
    
    Z2 = np.matmul(W2 ,A1) + b2
    A2 = relu(Z2)
    
    Z3 = np.matmul(W3, A2) + b3
    A3 = activation_softMax(Z3)
    
    
    
    cache = {
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2": A2,
        "Z3":Z3,
        "A3":A3
    }
    
    
    

    return (A3,cache)

## 5) Log Loss function in SoftMax Likehood(Maximize)

# y = Y_status_Map
# h(x) = ^y 
def cost_fuction(Y_status_Map,A2):

    Y_T = np.array(Y_status_Map)
    m = Y_T.shape[0]
    cost_fuctionSoftMax = -np.sum(Y_T * np.log(A2.T)) / m
    return cost_fuctionSoftMax

## 6) Back propagation
def backPropagation(parameters, X, Y_status_Map,cache):
    
    m = X.shape[0]
    Y_T = np.array(Y_status_Map).T 

    #all Parameters
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    #all forward propagation
    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']
    A3 = cache['A3']
    Z3 = cache['Z3']

    #Layer from output -> second Layer
    dZ3 = A3 - Y_T #Output -> Hidden Layer
    dW_third_layer = (dZ3 @ A2.T) / m
    db_third_layer = np.sum(dZ3, axis=1, keepdims=True) / m



    #Its the difference between the hidden layer and W2 
    dA2 =  W3.T @ dZ3 
    dZ2 = dA2 * (1- np.power(A2, 2))
    dW_second_layer  = (dZ2 @ A1.T) / m
    db_second_layer = np.sum(dZ2, axis=1, keepdims=True) / m
    
    #First layer
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * (1 - np.power(A1, 2))
    dW_first_layer = (dZ1 @ X) / m
    db_first_layer = np.sum(dZ1, axis=1, keepdims=True)
    
    
    

    grads = {
        "dW3":dW_third_layer,
        "db3":db_third_layer,
        "dW2":dW_second_layer,
        "db2":db_second_layer,
        "dW1":dW_first_layer,
        "db1":db_first_layer
    }
    
    return grads

def gradient_Descent(parameters,grads,learning_rate=1.2):
    
    #dW2 and dB2 from the output -> hidden layer
    for l in [1, 2, 3]:
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
    return parameters
    

#8 NN_network need to change
def nn_network(X, Y_status_Map, number_of_iterations = 10, printCostFalse = False):
    n_x,n_h1,n_h2,n_y = layer(X, Y_status_Map)
    
    parameters = parameters_initialization(n_x,n_h1,n_h2, n_y)
    for iteration in range(number_of_iterations):
        A3, cache = forward_propagation(X, parameters)
        
        
        cost = cost_fuction(Y_status_Map, A3)
        cache = backPropagation(parameters, X, Y_status_Map, cache)
        
        parameters = gradient_Descent(parameters, cache, 0.500)
        
        if printCostFalse:
            print(f"Iteration {iteration} - Cost: {cost}")

    return parameters
        
def predict(X, parameters):
    A2, _ = forward_propagation(X, parameters)
    return np.argmax(A2, axis=0)

# Evaluate



        
if __name__ == "__main__":
    main()