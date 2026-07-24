import torch.nn as nn
from models import Seq2Seq
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader


def init_weights(m, low_boundary : float = -0.08, high_boundary : float = 0.08):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, low_boundary, high_boundary)



def plot_weights_initialization(model : Seq2Seq):
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.append(param.cpu().flatten().detach().numpy())
    
    all_weights_concat = np.concatenate(all_weights)
    plt.hist(all_weights_concat.flatten(), bins=50)
    plt.title("Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.show()


def count_parameters_model(model : Seq2Seq) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_fuction(
    model : Seq2Seq, data_loader : DataLoader, optimizer : optim.Adam, criterion : nn.CrossEntropyLoss,  clip : float, teacher_forcing_ratio : float, device
) -> float:
    model.train() #Putting the model for training
    epoch_loss = 0.0
    for index, batch in enumerate(data_loader):
        src = batch['tokenized_input'].to(device) # shape -> (Batch size, src lenght)
        
        trg = batch['label'].to(device) # shape -> (Batch size, trg lenght)
        
        optimizer.zero_grad() #Making all the gradients 0 so it can update
        
        logits = model(src, trg, 0) # Shape -> (batch_size, trg_lenght, output_dim)
        
        output_dim = logits.shape[-1]
        
        logits = logits[1:].view(-1, output_dim) #Make the logits concat with the batch size * trg lenght
        
        trg = trg[1:].view(-1)
        
        loss = criterion(logits, trg)
        
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), clip) #For fighting Exploiting Gradients
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(data_loader)
    
def validation_fuction(
    model : Seq2Seq, data_loader : DataLoader, criterion : nn.CrossEntropyLoss,  device
) -> float:
    model.eval() #Putting the model for validation
    epoch_loss = 0.0
    for index, batch in enumerate(data_loader):
        src = batch['tokenized_input'].to(device) # shape -> (Batch size, src lenght)
        
        trg = batch['label'].to(device) # shape -> (Batch size, trg lenght)
        
        logits = model(src, trg, 0) # Shape -> (batch_size, trg_lenght, output_dim)
        
        output_dim = logits.shape[-1]
        
        logits = logits[1:].view(-1, output_dim) #Make the logits concat with the batch size * trg lenght
        
        trg = trg[1:].view(-1)
        
        loss = criterion(logits, trg)
        
        epoch_loss += loss.item()
        
        
    return epoch_loss / len(data_loader)
    