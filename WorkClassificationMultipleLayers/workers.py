import pandas as pd
import numpy as np


#

def main():
    df = pd.read_csv("employee_promotion.csv")
    
    #Independent Variables
    X = df.iloc[:,1].values.reshape(1, -1)
    Y = df["Promotion"].values.reshape(1, -1)
    
    #Normalize data, problem was Axis == 0
    mean_X = np.mean(X)
    std_X = np.std(X)
    X_normalize = (X - mean_X) / std_X
    
    output = nn_network(X_normalize,Y,10000,True)
    
    evaluate_model(X_normalize,Y,output)
    
    
    

#1) Layers 
def layers_sizes(X,Y):
    #Numbers of inputs
    n_x = X.shape[0]
    #Numbers of hidden layers
    n_h = 2
    
    #Output Layer
    n_y = Y.shape[0]
    
    return n_x, n_h, n_y
#2) Parameter Initialization
def parameters_initialization(n_x,n_h, n_y):
    
    W1 = np.random.randn(n_h,n_x) 
    b1 = np.zeros((n_h,1))
    
    
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y,1))
    
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    paramters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return paramters
#3) Activation Function
def activation_fuction(Z):
    return 1 / ( 1 + np.exp(-Z) )
    
#4) Forward Propagation
def foward_Propagation(X,parameters, keep_dropout = 0.8):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    
    Z1 = np.matmul(W1, X) + b1
    A1 = activation_fuction(Z1)
    
    #Dropout
    D1 = np.random.rand(A1.shape[0], A1.shape[1]) < keep_dropout
    A1 = A1 * D1
    A1 = A1 / keep_dropout
    
    
    Z2 = np.matmul(W2, A1) + b2
    A2 = activation_fuction(Z2)
    
    
    
    cache = {
        "Z1":Z1,
        "A1": A1,
        "Z2": Z2,
        "A2":A2,
        "D1":D1
        
    }
    return (A2, cache)

#5) Cost Function
def cost_fuction(A2, Y):
    
    m = Y.shape[1]
    epsilon = 1e-10
    A2_clipped = np.clip(A2, epsilon, 1 - epsilon)
    cost = np.sum(-Y * np.log(A2_clipped) - (1 - Y) * np.log(1 - A2_clipped))
    finally_cost = cost / m

    return finally_cost
    


#6) Backward_propagation
def back_propagation(cache, parameters, X, Y, keep_dropout = 0.8):
    
    m = Y.shape[1]
    #All the parameters
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    #all foward propagation
    A1 = cache["A1"]
    A2 = cache["A2"]
    #Dropout
    D1 = cache["D1"]
    
    
    #The most closest to the ouput layer
    print(f"A2 shape {A2.shape}")
    print(f"Y shape {Y.shape}")
    Dz2 = A2 - Y  
    dw_second_layer = (1/m * np.dot(Dz2,A1.T)) 
    db_second_layer = (1/m * np.sum(Dz2, axis=1, keepdims=True))
    
    # (W * Dz2) * A * (1-y)
    #the second Layer, the most far in the layers
    Dz1 = np.dot(W2.T, Dz2) * A1 * (1 - A1)
    Dz1 = Dz1 * D1
    Dz1 = Dz1 / keep_dropout
    
    dw_first_layer = (1/m * np.dot(Dz1, X.T))
    db_first_layers = (1/m * np.sum(Dz1, axis=1, keepdims=True))
    
    grads = {
        "dW2":dw_second_layer,
        "dB2":db_second_layer,
        "dW1":dw_first_layer,
        "dB1":db_first_layers
    }
    return grads
    
#7) Update Parameters(Gradient Descent)
def update_parameters(grads, parameters, learning_rate=1.2):
    #Get grads
    # Near the output layers(Hidden Layers)
    dW2 = grads["dW2"]
    dB2 = grads["dB2"]
    
    # Near the input layers(Hidden Layers)
    dW1 = grads["dW1"]
    dB1 = grads["dB1"]
    
    #Get parameters
    #Near the output 
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    
    #Learning rate of second layer(near the output)
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * dB2
    
    #Learning rate of first layer(near the input)
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * dB1
    
    parameters = {
        "W2":W2,
        "b2":b2,
        "W1":W1,
        "b1":b1
    }
    
    return parameters

    
#8) Neuronal Network    

def nn_network(X, Y , numbers_of_iterations = 10, print_cost = False, keep_dropout = 0.8):
    #Input Layer
    n_x, n_h, n_y = layers_sizes(X,Y)
    parameters = parameters_initialization(n_x ,n_h ,n_y)
    
    for iteration in range(numbers_of_iterations):
        
        #Get all the formulas
        A2, cache = foward_Propagation(X, parameters, keep_dropout)
        
        cost = cost_fuction(A2,Y)
        
        
        grads = back_propagation(cache, parameters, X, Y, keep_dropout)
        
        parameters = update_parameters(grads, parameters, learning_rate=0.1)
        
        if print_cost:
            print(f"The cost fuction loss in this iteration is {iteration}:{cost}")
        
    return parameters
    
       
#9) Predict
def prediction(X, parameters):
    
    #get the new Foward Propation
    A, cache = foward_Propagation(X, parameters, keep_dropout=1)
    
    return (A > 0.5).astype(int) 

#10) Evaluate model
def evaluate_model(X,Y,parameters):
    predictions = prediction(X, parameters)
    true_labels = (Y > 0.5).astype(int)
    accuracy = np.mean(predictions == true_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
if __name__ == "__main__":
    main()