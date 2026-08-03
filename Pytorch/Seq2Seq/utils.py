import torch.nn as nn
from models.seq2seq_baseline import Seq2Seq
from models.seq2seq_attention import Seq2SeqAttention

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from typing import Tuple
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def init_weights(m, low_boundary : float = -0.08, high_boundary : float = 0.08):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, low_boundary, high_boundary)

def init_weights_attention(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data,  mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


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
    model : Seq2Seq | Seq2SeqAttention, data_loader : DataLoader, optimizer : optim.Adam, criterion : nn.CrossEntropyLoss,  clip : float, teacher_forcing_ratio : float, device
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
    model : Seq2Seq | Seq2SeqAttention, data_loader : DataLoader, criterion : nn.CrossEntropyLoss,  device
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

import torch

def predict_LSTM(
    sentence,
    model,
    tokenizer,
    device="cpu",
    max_output_length=50,
    repetition_penalty=1.2
):
    model.eval()
    
    
    input_ids = tokenizer.encode(sentence)
    src_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)  # Shape: [1, seq_len]
    
    
    eos_idx = tokenizer.text_to_token_ids.get("<|endoftext|>", 0)

    
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    
    current_input = torch.LongTensor([eos_idx]).to(device)
    generated_ids = []

    for _ in range(max_output_length):
        with torch.no_grad():
            output, hidden, cell = model.decoder(current_input, hidden, cell)
            
        logits = output[0] 
        
       
        for token_id in set(generated_ids):
            if logits[token_id] < 0:
                logits[token_id] *= repetition_penalty
            else:
                logits[token_id] /= repetition_penalty
                
        pred_token_idx = logits.argmax(0).item()
        
        if pred_token_idx == eos_idx:
            break
            
        generated_ids.append(pred_token_idx)
        current_input = torch.LongTensor([pred_token_idx]).to(device)

    
    return tokenizer.decode(generated_ids)



def predict_GRU_attention(
    sentence,
    model,
    tokenizer,
    device="cpu",
    max_output_length=50,
    repetition_penalty=1.2) -> Tuple:
    model.eval()
    
    input_ids = tokenizer.encode(sentence)
    src_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)  # Shape: [1, seq_len]
    
    sos_idx = tokenizer.sos_token_id
    eos_idx = tokenizer.eos_token_id

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    current_input = torch.LongTensor([sos_idx]).to(device)
    generated_ids = []
    
    # For saving the weights of attention 
    attentions = []

    for _ in range(max_output_length):
        with torch.no_grad():
            output, hidden, attention_weights = model.decoder(current_input, hidden, encoder_outputs)
            
       
        attentions.append(attention_weights.squeeze(1).cpu().numpy())
            
        logits = output[0] 
        
        for token_id in set(generated_ids):
            if logits[token_id] < 0:
                logits[token_id] *= repetition_penalty
            else:
                logits[token_id] /= repetition_penalty
                
        pred_token_idx = logits.argmax(0).item()
        
        if pred_token_idx == eos_idx:
            break
            
        generated_ids.append(pred_token_idx)
        current_input = torch.LongTensor([pred_token_idx]).to(device)

    # Devolvemos tanto el texto como las atenciones (por si quieres graficar)
    predicted_text = tokenizer.decode(generated_ids)
    
    return predicted_text, attentions




def plot_attention_matrix(prompt, predicted_text, tokenizer, attention_weights):
    
    input_ids = tokenizer.encode(prompt)
    x_labels = [tokenizer.token_ids_to_token.get(idx, "<unk>") for idx in input_ids]

    y_labels = predicted_text.split()
    

    attention_matrix = np.vstack(attention_weights)
    

    min_y = min(len(y_labels), attention_matrix.shape[0])
    min_x = min(len(x_labels), attention_matrix.shape[1])
    
    attention_matrix = attention_matrix[:min_y, :min_x]
    y_labels = y_labels[:min_y]
    x_labels = x_labels[:min_x]
    

    plt.figure(figsize=(12, 8))

    sns.heatmap(attention_matrix, xticklabels=x_labels, yticklabels=y_labels, 
                cmap='magma', annot=False)
    
    plt.title("What is the model looking", fontsize=14, pad=20)
    plt.xlabel("Original Prompt (Memory of Encoder)", fontsize=12)
    plt.ylabel("generated prompt (prediction of Decoder)", fontsize=12)
    
    
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.show()