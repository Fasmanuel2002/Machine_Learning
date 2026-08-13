from typing import Tuple, Dict 
from torch import Tensor
import torch

def get_batch(split : str , batch_size : int, sequence_length : int, train_data : Tensor, val_data : Tensor, device : torch.device) -> Tuple[Tensor, Tensor]:
    
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    
    #The random ints of the data, make the batch size for the number of total batches and sequence lenght for knowing the lenght
    ix = torch.randint(len(data) - sequence_length, (batch_size, ))
    
    #Creating the inputs for the decoder generation, all the previous tokens for making the prediction "this are the context"
    input_x = torch.stack([data[index : index + sequence_length ] for index in ix])
    #Creating the labels for the decoder generation, the token that is for the predicition of the model
    target_y = torch.stack([data[index + 1: index + sequence_length + 1] for index in ix])
    
    input_x, target_y = input_x.to(device), target_y.to(device)
    
    return input_x, target_y

@torch.no_grad()
def estimate_loss(model, eval_iters: int, batch_size: int, sequence_length: int, train_data: Tensor, val_data: Tensor, device: torch.device) -> Dict[str,float]:    
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split=split, batch_size=batch_size, sequence_length=sequence_length, train_data=train_data, val_data=val_data, device=device)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
        