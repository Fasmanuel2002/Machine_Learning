import pandas as pd
import numpy as np


#1) I need to train the machine Lerning model with the "Train.csv" for mathematical way
#2) Do the mathemathics 
#3) Program so IA learn
def main():
    modelForTraining = pd.read_csv("house-prices-advanced-regression-techniques/cleaned_train.csv") 
    
    
    X = modelForTraining.iloc[:, 0:38]  # Evidence
    Y = modelForTraining["SalePrice"]   # Label
    epsilon = 1e-8
    
    
    Y_orig_mean = Y.mean()
    Y_orig_std = Y.std()
    
    
    X_normalize = (X - X.mean()) / (X.std() + epsilon)
    
    Y_normalize = (Y - Y_orig_mean) / (Y_orig_std + epsilon)
    
    
    X_multi_norm = np.array(X_normalize).T
    Y_multi_norm = np.array(Y_normalize).reshape((1, len(Y_normalize)))
    
   
    FinalParameter = nn_model(X_multi_norm, Y_multi_norm, numberInterations=100, learning_rate=0.1, beta=0.9, printCost=True)
    print(f"Gradient descent result: W, b = {FinalParameter['W']}, {FinalParameter['b']}")
    
   
    X_pred = X_multi_norm[:, 0]  # Primera casa en los datos

   
    predicted_price = prediction(X_multi_norm, Y_multi_norm, FinalParameter, X_pred, Y_orig_mean, Y_orig_std)
    print(f"Predicted house price (denormalized): {predicted_price}")

#Layers
def layers(X, Y):
    
    n_x = X.shape[0]
    n_y = 1
    
    return n_x, n_y
    
#Intialize Parameters
def initializeParameters(n_x,n_y):
    #W = Weight
    W = np.random.rand(n_y, n_x) * 0.01
    # B = Bias
    B = np.zeros((n_y,1))
    
    parameters = {
        "W": W,
        "b": B
    }
    
    return parameters

def forward_propagation(parameters,X):
    #Weights
    W = parameters["W"]
    #Bias
    b = parameters["b"]
    #Output
    z = np.matmul(W,X) + b
    yhat = z
    return yhat

#Loss cost
def compute_Cost(Y_hat,Y):
    #Number of examples
    m = Y_hat.shape[1]
    #Cost Fuction
    l = (1/ (2 * m) ) * np.sum((Y_hat - Y)**2)
    return l

     
def back_Propagation(X , Y , Y_hat):
    #Number of examples
    m = Y_hat.shape[1]
    
    # 
    dz = Y_hat - Y
    dw = 1/m * np.dot(dz, X.T)
    db = 1/m * np.sum(dz, axis=1, keepdims=True)

    grads = {
        "dw": dw,
        "db": db
    }
    return grads

#Update parameters
def gradient_descent(grads, parameters, learning_rate=1.2, beta=0.9, v=None):
    # Inicializar Momentum si es la primera iteración
    if v is None:
        v = {
            "dW": np.zeros_like(parameters["W"]),
            "db": np.zeros_like(parameters["b"])
        }

   
    W = parameters["W"]
    b = parameters["b"]
    
  
    dW = grads["dw"]
    db = grads["db"]

   
    v["dW"] = beta * v["dW"] + (1 - beta) * dW
    v["db"] = beta * v["db"] + (1 - beta) * db

    # Actualizar parámetros con Momentum
    W -= learning_rate * v["dW"]
    b -= learning_rate * v["db"]

   
    return {"W": W, "b": b}, v
#Neuronal Model 
def nn_model(X, Y, numberInterations = 10, learning_rate = 1.2,beta = 0.9, printCost = False):
    #Call all fuctions
    n_x = layers(X,Y)[0] # First one of boths
    n_y = layers(X,Y)[1] # Second one 
    parameters = initializeParameters(n_x=n_x, n_y=n_y)
    
    v = None
    
    for iteration in range(numberInterations):
        #Foward progatarion (get y_hat  Y = X @ W + b)
        y_hat = forward_propagation(parameters, X)
        #Get compute cost of all the iteration(1/ 2 * m) * np.sum((y_hat - Y))**2
        cost = compute_Cost(y_hat,Y)
        #Back propagation(Get dw and db)
        grads = back_Propagation(X,Y,y_hat)
        
        #Gradient descent (w = w - learning rate * dw  and b = b - learning rate * db)
        parameters, v = gradient_descent(grads, parameters, learning_rate, beta, v)

        
        if printCost:
            print(f"This is the total cost {iteration}:{cost}")
        
    return parameters

def prediction(X, Y, parameters, X_pred, Y_orig_mean, Y_orig_std):
    W = parameters["W"]
    b = parameters["b"]

    # Calcuate the mean and std of X
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True) + 1e-8

    # reshape X_pred to be a column vector
    X_pred = X_pred.reshape(-1, 1)
    X_pred_norm = (X_pred - X_mean) / X_std

    # Make the prediction
    y_hat = np.matmul(W, X_pred_norm) + b

    # Desnormalizar using the original mean and std of Y
    y_pred = y_hat * Y_orig_std + Y_orig_mean
    return y_pred[0]
if __name__ == "__main__":
    main()
    
