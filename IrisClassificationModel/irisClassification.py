import pandas as pd
import numpy as np


def main():
    data = pd.read_csv("iris2.csv")
    
    X = data.iloc[:,0:4].values
    Y = data["species"]

    species_map = {"setosa": [1, 0, 0], "versicolor": [0, 1, 0], "virginica": [0, 0, 1]}
    Y_one_hot = pd.DataFrame(Y.map(species_map).tolist(), columns=["setosa", "versicolor", "virginica"]).values
    
    #Normalize the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    ouput = nn_network(X, Y_one_hot, 10000, True)
    
    evaluate_model(X, Y_one_hot, ouput)
    

##1)Layer
def layer(X, Y_one_hot):
    n_x = X.shape[1]  
    n_y = Y_one_hot.shape[1]  
    return n_x, n_y


##2)Parameters Inialization

def parameters_initialization(n_x,n_y):
    W = np.random.rand(n_y, n_x) * 0.01
    b = np.zeros((n_y,1))

    parameters = {
        "W": W,
        "b": b
    }
    
    return parameters

##3.1)Activation fuction:
def sigmoid(Z):
    return 1/ (1 + np.exp(-Z))

##3.2)Forward propagation
def forward_propagation(parameters,X):
    W = parameters["W"]
    b = parameters["b"]
    Z = np.matmul(W, X.T) + b
    A = sigmoid(Z)
    return A

##4) Cost Fuction (Loss fuction):
def cost_fuction(A, Y_one_hot):
    
    m = Y_one_hot.shape[0]
    A = np.clip(A, 1e-10, 1 - 1e-10)
    loss_fuction = np.sum(-Y_one_hot * np.log(A.T) - (1- Y_one_hot) * np.log(1 - A.T))
    cost_fuction = loss_fuction / m
    return cost_fuction


##5) Back propagation:
def back_propagation(A, X, Y_one_hot):
    #M Para normalizar variables
    m = Y_one_hot.shape[0]
    dz = A - Y_one_hot.T
    
    dw = np.matmul(dz, X) / m
    db = np.sum(dz, axis=1, keepdims=True) / m
    
    grads = {
        "dW": dw,
        "db": db
    }
    return grads

##6) Gradient Descent:

def gradient_descent(grads, parameters,Learning_rate=1.2):
    #Define the grads  
    dw = grads["dW"]
    db = grads["db"]
    
    #Define Parameters
    W = parameters["W"]
    b = parameters["b"]
    
    #Update Parameters
    W = W - Learning_rate * dw 
    b = b - Learning_rate * db 
    
    parameters ={
        "W": W,
        "b": b
    }
    return parameters



##7) Neuronal Networl
def nn_network(X,Y_one_hot, numbers_of_iterations = 10, print_cost = False):
    
    n_x = layer(X, Y_one_hot)[0]
    n_y = layer(X, Y_one_hot)[1]
    
    parameters = parameters_initialization(n_x, n_y)
    for iterations in range(numbers_of_iterations):
        #Propagation
        A = forward_propagation(parameters, X)
        
        #Cost
        cost = cost_fuction(A, Y_one_hot)
        
        #Back Propagation
        grads = back_propagation(A,X,Y_one_hot)
        
        parameters = gradient_descent(grads, parameters, Learning_rate=0.1)
        
        if print_cost:
            print(f"The cost fuction in this iteration is {iterations}:{cost}")
        
    return parameters
        
#8 prediction
def prediction(X, parameters):
    # Forward propagation
    A = forward_propagation(parameters, X)
    
    # Get the index of the highest probability (class prediction)
    predictions = np.argmax(A, axis=0)
    
    return predictions

def evaluate_model(X, Y, parameters):
    predictions = prediction(X, parameters)
    true_labels = np.argmax(Y, axis=1)
    accuracy = np.mean(predictions == true_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

