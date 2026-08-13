from gpt import GPTLM
from utils import get_batch, estimate_loss
import torch
from EarlyStopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter # type: ignore

def main():
    
    # hyperparameters
    batch_size = 2 # how many independent sequences will we process in parallel?
    sequence_lenght = 8 # what is the maximum context length for predictions?
    max_iters = 10000
    eval_interval = 10
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 10
    n_embd = 16
    n_head = 2
    n_layer = 2
    dropout = 0.2
    
    
    torch.manual_seed(1337)
    
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocabulary_size = len(chars)
    
    #Encoder
    stoi = {character : index for index, character in enumerate(chars)}
    encode = lambda s : [stoi[character] for character in s] # encoder: take a string, output a list of integers


    #Decoder
    itos = {index : character for index, character in enumerate(chars)}
    decode = lambda l : ''.join([itos[index] for index in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    
    n_len = int(0.90 * len(data))
    train_data = data[:n_len]
    val_data = data[n_len:]
    
    
    gpt_model = GPTLM(vocab_size=vocabulary_size, 
                      n_embeddings=n_embd, 
                      sequence_lenght=sequence_lenght,
                      n_heads=n_head, 
                      n_layers=n_layer, 
                      dropout=dropout,
                      device=device)
    
    m = gpt_model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    print("Training is starting")
    
    optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=learning_rate)
    earlyStopping = EarlyStopping(patience=20, mode="min", path="best_gpt_model.pt")
    
    #Summary Writer for Tensorboard
    tensor_board_writer = SummaryWriter(log_dir=f"runs")
    
    for iter in range(max_iters):
        
        if (iter % eval_interval == 0) or (iter == max_iters - 1):
            losses = estimate_loss(model=gpt_model, eval_iters=eval_iters, batch_size=batch_size,
                                   sequence_length=sequence_lenght, train_data=train_data, val_data=val_data, device=device)
            earlyStopping(losses["val"], gpt_model)
            
            if earlyStopping.early_stop:
                print("Finished because of the early Stopping")
                break 
              
            print(f'Epoch {iter}: train loss: {losses["train"]:.4f}, val losses {losses["val"]:.4f}')
            tensor_board_writer.add_scalar("Train/Loss", losses['train'], iter)
            tensor_board_writer.add_scalar("Val/Loss", losses['val'], iter)
            
        xb, yb = get_batch('train',batch_size=batch_size,sequence_length=sequence_lenght,
                           train_data=train_data,val_data=val_data,device=device) # type: ignore
        
        logits, loss = gpt_model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    tensor_board_writer.close()
    
    
    # Generación de texto
    gpt_model.load_state_dict(torch.load("best_gpt_model.pt"))
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(gpt_model.generate(context, max_new_tokens=500)[0].tolist()))

if __name__ == "__main__":
    main()