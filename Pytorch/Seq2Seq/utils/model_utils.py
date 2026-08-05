import torch.nn as nn

from models.seq2seq_baseline import Seq2Seq
from models.seq2seq_attention import Seq2SeqAttention
from typing import Tuple
import torch



def init_weights(m, low_boundary : float = -0.08, high_boundary : float = 0.08):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, low_boundary, high_boundary)

def init_weights_attention(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data,  mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters_model(model : Seq2Seq | Seq2SeqAttention) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

    
    predicted_text = tokenizer.decode(generated_ids)
    
    return predicted_text, attentions




