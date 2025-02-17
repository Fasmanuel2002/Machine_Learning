import pandas as pd
import numpy as np
CSVPATH = "house_prices_one_variable.csv"

def main():
    csv = pd.read_csv(CSVPATH)
    #m == w
    
    # This is for making the                    
    X = csv['Square_Feet']
    Y = csv['House_Price']
    
    #Normalize the parameters
    X_normalize = (X - np.mean(X)) / np.std(X)
    Y_normalize = (Y - np.mean(Y)) / np.std(Y)
    
    
    m_initial = 0
    b_initial = 0
    num_iterations = 20
    learning_rate = 0.3
    m_gd, b_gd = gradient_descent(dEdm, dEdb, m_initial, b_initial, X_normalize, Y_normalize, learning_rate, num_iterations, print_cost=True)
    print(f"Gradient descent result: m_min, b_min = {m_gd}, {b_gd}")
    
    
    X_pred = np.array([800 , 2000 , 3500 ])
    
    #normalize the x_pred
    X_pred_norm = (X_pred - np.mean(X))/np.std(X)
    Y_pred_gd_norm = m_gd * X_pred_norm + b_gd #formula  y = mx + b
    
    # Use the same mean and standard deviation of the original training array Y
    Y_pred_gd = Y_pred_gd_norm * np.std(Y) + np.mean(Y)
    print(f"Predicted price for {X_pred} square feet: {Y_pred_gd}")

#Cost Fuction
def E(m,b,x,y):
    return 1/(2 * len(y)) * np.sum((m * x + b - y)**2)
# Partial derivate of wieght
def dEdm(m,b,x,y):
    res  = 1/len(y) * np.dot((m * x + b - y),x)
    return res
# Partial Derivate of bias
def dEdb(m,b,x,y):
    res = 1/len(y) * np.sum((m * x + b - y)) 
    return res

#Gradient Descent
def gradient_descent(dEdm, dEdb, w,b,x, y, LEARNIN_RATE=0.0001, iterations=1000,  print_cost=False):
    
    for iteration in range(iterations):
        new_weight = w - LEARNIN_RATE * dEdm(w,b,x,y)
        new_bias = b - LEARNIN_RATE * dEdb(w,b,x,y)
        
        w = new_weight
        b = new_bias
        
        if print_cost:
            print(f"The total cost in {iteration} : {E(w,b,x,y)}")    
    return w, b

if __name__ == "__main__":
    main()