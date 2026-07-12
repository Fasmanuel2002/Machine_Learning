from typing import Any

import torch
import torch.nn.functional as F
from torch import Generator

class Linear:
    def __init__(self, fan_in : int, fan_out : int, generator : Generator, bias : bool = True) -> None:
        #Innitalization the weights of the model and adding the fan_in for the gain or the wieght initialization see more https://docs.pytorch.org/docs/2.13/nn.init.html
        self.weight = (torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5)
        #Innitialization of the bias that if there is BatchNorm will be equal to None because of the BatchNorm bias
        self.bias = torch.zeros(fan_out) if bias else None
        
    def __call__(self,x):
        #This is the forward pass in Pytorch
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        #The total parameters of the model
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1D:
    #The class for the normalization of the data, converting the layer from mean = x and std = x to mean = 0 and std = 1, making the layer more Gaussian
    
    def __init__(self, dim : int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        #The variables from the BatchNorm Class see more https://docs.pytorch.org/docs/2.13/generated/torch.nn.BatchNorm1d.html
        self.eps = eps # a value added to the denominator so there is not an Zero division
        self.momentum = momentum #The value used so there is a differentation on average for the running_mean and variance
        self.training = True #to track the running_mean and variance statistics
        #parameters (trained with the backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        
        #buffers(trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        
    def __call__(self, x):
            if self.training:
                #For updating the training mean and variance
                xmean = x.mean(0, keepdim=True) #Batch mean
                xvar = x.var(0, keepdim=True) #Batch variance
            else:
                xmean = self.running_mean
                xvar = self.running_var

            xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance (x - mean) / sqrt(xvar + self.eps)
            self.out = self.gamma * xhat + self.beta
            
            if self.training:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                    
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
            
            return self.out

    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:
    #The class for Tanh for making the outputs from -1 to 1
    def __call__(self, x) -> Any:
        self.out = torch.tanh(x) # https://docs.pytorch.org/docs/2.13/generated/torch.nn.Tanh.html
        return self.out
    def parameters(self):
        return []